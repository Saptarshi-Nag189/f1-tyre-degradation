"""GPU-accelerated data processing."""
import pandas as pd
import numpy as np
import torch
import logging
from typing import Dict, List, Optional
from contextlib import contextmanager

try:
    import cupy as cp
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from .gpu_memory import GPUMemoryManager


class GPUDataProcessor:
    """Process F1 data using GPU acceleration where available.
    
    Implements GPU processing for eligible features per Feature Contract:
    - Speed, acceleration, speed_ma_5, lateral_acceleration
    - Throttle, Brake, Distance
    
    Falls back to CPU when:
    - GPU not available
    - Dataset < 1000 rows (threshold per Feature Contract)
    - GPU memory exhausted
    
    Attributes:
        batch_size: Number of rows to process at once.
        use_gpu: Whether to use GPU acceleration.
        device: Current device ('cuda' or 'cpu').
    """
    
    def __init__(self, batch_size: int = 1000, use_gpu: bool = True):
        """Initialize the GPU data processor.
        
        Args:
            batch_size: Number of rows to process in each batch.
            use_gpu: Whether to attempt GPU processing.
        """
        self.batch_size = batch_size
        self.use_gpu = use_gpu and torch.cuda.is_available() and GPU_AVAILABLE
        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.memory_manager = GPUMemoryManager()
        self.logger = logging.getLogger(__name__)
        
        if self.use_gpu:
            self.logger.info(f"GPU processing on {torch.cuda.get_device_name()}")
        else:
            self.logger.info("Using CPU processing")
    
    @contextmanager
    def processing_context(self):
        """Context manager for safe GPU processing with OOM fallback.
        
        Catches OOM errors and falls back to CPU mode automatically.
        Note: yields exactly once. On OOM, sets use_gpu=False so the
        NEXT call will use CPU — the current block is NOT re-entered.
        
        Yields:
            None
        """
        try:
            with self.memory_manager.gpu_memory_context():
                yield
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            self.logger.warning(f"GPU OOM, falling back to CPU: {e}")
            self.use_gpu = False
            # Don't re-yield — the caller's `with` block has already exited
    
    def process_telemetry_batch(self, telemetry_data: Dict, features: List[str]) -> pd.DataFrame:
        """Process telemetry data with GPU acceleration.
        
        Args:
            telemetry_data: Dictionary of driver code -> telemetry DataFrame.
            features: List of feature names to calculate.
            
        Returns:
            pd.DataFrame: Concatenated DataFrame with all engineered features.
        """
        results = []
        
        with self.processing_context():
            for driver, tel_df in telemetry_data.items():
                if tel_df is None or tel_df.empty:
                    continue
                
                # GPU threshold per Feature Contract: 1000 rows
                if self.use_gpu and len(tel_df) > 1000:
                    driver_result = self._process_gpu(tel_df, driver, features)
                else:
                    driver_result = self._process_cpu(tel_df, driver, features)
                
                if driver_result is not None:
                    results.append(driver_result)
        
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def _process_gpu(self, tel_df: pd.DataFrame, driver: str, features: List[str]) -> Optional[pd.DataFrame]:
        """GPU-accelerated telemetry processing.
        
        Uses cuDF for DataFrame operations and CuPy for array computations.
        
        Args:
            tel_df: Telemetry DataFrame for a driver.
            driver: Driver code.
            features: List of features to calculate.
            
        Returns:
            pd.DataFrame: Processed DataFrame or None on failure.
        """
        try:
            gpu_df = cudf.from_pandas(tel_df)
            batch_results = []
            
            for i in range(0, len(gpu_df), self.batch_size):
                batch = gpu_df.iloc[i:i + self.batch_size]
                processed = self._calculate_gpu_features(batch, features, driver)
                if processed is not None:
                    batch_results.append(processed.to_pandas())
            
            return pd.concat(batch_results, ignore_index=True) if batch_results else None
        except Exception as e:
            self.logger.warning(f"GPU processing failed for {driver}: {e}")
            return self._process_cpu(tel_df, driver, features)
    
    def _process_cpu(self, tel_df: pd.DataFrame, driver: str, features: List[str]) -> Optional[pd.DataFrame]:
        """CPU fallback for telemetry processing.
        
        Uses pandas/numpy for all operations.
        
        Args:
            tel_df: Telemetry DataFrame for a driver.
            driver: Driver code.
            features: List of features to calculate.
            
        Returns:
            pd.DataFrame: Processed DataFrame or None on failure.
        """
        try:
            results = []
            cpu_batch_size = min(self.batch_size, 500)
            for i in range(0, len(tel_df), cpu_batch_size):
                batch = tel_df.iloc[i:i + cpu_batch_size].copy()
                processed = self._calculate_cpu_features(batch, features, driver)
                if processed is not None:
                    results.append(processed)
            return pd.concat(results, ignore_index=True) if results else None
        except Exception as e:
            self.logger.error(f"CPU processing failed for {driver}: {e}")
            return None
    
    def _calculate_gpu_features(self, gpu_df: 'cudf.DataFrame', features: List[str], driver: str) -> Optional['cudf.DataFrame']:
        """Calculate features using GPU operations.
        
        Implements DEFINED features per Feature Contract Section 1:
        - acceleration: Speed.diff()
        - speed_ma_5: 5-sample rolling mean of Speed
        - lateral_acceleration: curvature * speed^2
        
        Args:
            gpu_df: cuDF DataFrame batch.
            features: Feature names to calculate.
            driver: Driver code for tagging.
            
        Returns:
            cudf.DataFrame: DataFrame with calculated features.
        """
        result = gpu_df.copy()
        
        # acceleration (DEFINED): First-order difference of Speed
        if 'speed_acceleration' in features and 'Speed' in gpu_df.columns:
            speed_arr = cp.asarray(gpu_df['Speed'].values)
            accel = cp.diff(speed_arr, prepend=speed_arr[0])
            result['acceleration'] = cp.asnumpy(accel)
        
        # speed_ma_5 (DEFINED): 5-sample rolling mean
        if 'speed_moving_avg' in features and 'Speed' in gpu_df.columns:
            result['speed_ma_5'] = gpu_df['Speed'].rolling(window=5, min_periods=1).mean()
        
        # lateral_acceleration (DEFINED): curvature * speed^2
        if 'lateral_acceleration' in features and all(c in gpu_df.columns for c in ['Speed', 'Distance']):
            speed_ms = cp.asarray(gpu_df['Speed'].values) / 3.6
            dist = cp.asarray(gpu_df['Distance'].values)
            
            dx = cp.diff(dist, prepend=dist[0])
            dv = cp.diff(speed_ms, prepend=speed_ms[0])
            
            curvature = cp.where(dx != 0, cp.abs(dv / dx), 0)
            lateral_acc = speed_ms * curvature
            result['lateral_acceleration'] = cp.asnumpy(lateral_acc)
        
        result['Driver'] = driver
        return result
    
    def _calculate_cpu_features(self, df: pd.DataFrame, features: List[str], driver: str) -> Optional[pd.DataFrame]:
        """Calculate features using CPU operations.
        
        CPU fallback implementing same features as GPU path.
        
        Args:
            df: pandas DataFrame batch.
            features: Feature names to calculate.
            driver: Driver code for tagging.
            
        Returns:
            pd.DataFrame: DataFrame with calculated features.
        """
        result = df.copy()
        
        # acceleration (DEFINED)
        if 'speed_acceleration' in features and 'Speed' in df.columns:
            result['acceleration'] = df['Speed'].diff().fillna(0)
        
        # speed_ma_5 (DEFINED)
        if 'speed_moving_avg' in features and 'Speed' in df.columns:
            result['speed_ma_5'] = df['Speed'].rolling(window=5, min_periods=1).mean()
        
        # lateral_acceleration (DEFINED)
        if 'lateral_acceleration' in features and all(c in df.columns for c in ['Speed', 'Distance']):
            speed_ms = df['Speed'] / 3.6
            dx = df['Distance'].diff().fillna(1)
            dv = speed_ms.diff().fillna(0)
            curvature = np.where(dx != 0, np.abs(dv / dx), 0)
            result['lateral_acceleration'] = speed_ms * curvature
        
        result['Driver'] = driver
        return result
