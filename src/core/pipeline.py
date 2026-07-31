"""Modular F1 tyre degradation pipeline — UI-ready interface."""
from pathlib import Path
from typing import Callable, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class F1Pipeline:
    """Orchestrates data loading, feature engineering, training, and evaluation.
    
    Usage (script):
        pipeline = F1Pipeline(project_root=Path('d:/top_secret/f1_tyre_degradation_model'))
        results = pipeline.run_full_pipeline()
    
    Usage (future UI):
        pipeline = F1Pipeline(
            project_root=...,
            config_overrides={'model': {'epochs': 50, 'sequence_length': 5}}
        )
        results = pipeline.run_full_pipeline(callback=my_progress_fn)
    """

    STAGES = ['setup', 'load_data', 'build_features',
              'prepare_sequences', 'build_model', 'train',
              'evaluate', 'save']

    def __init__(self, project_root: Path, config_overrides: dict = None):
        self.project_root = Path(project_root)
        self.config_overrides = config_overrides or {}
        self.config = None
        self.gpu_config = None
        self.model = None
        self.results = None
        self._callback = None

    def run_full_pipeline(self, callback: Optional[Callable] = None) -> dict:
        """Run complete train + evaluate pipeline.
        
        Args:
            callback: Optional fn(stage: str, progress: float, message: str)
                      for UI progress. progress is 0.0–1.0.
        
        Returns:
            dict: {metrics, model_path, history}
        """
        import sys
        sys.path.insert(0, str(self.project_root / 'src'))
        
        from utils.setup import setup_environment, load_config
        from modeling.gpu_models import GPUTyreModel
        
        self._callback = callback
        self._report('setup', 0.0, 'Initializing environment...')
        
        setup_environment(project_root=self.project_root)
        self.config = load_config(
            str(self.project_root / 'config' / 'config.yaml'))
        self.gpu_config = load_config(
            str(self.project_root / 'config' / 'gpu_config.yaml'))
        self._apply_overrides()
        
        # Import training functions
        sys.path.insert(0, str(self.project_root / 'scripts'))
        from gpu_enhanced_training import (
            load_data, build_lap_features, prepare_lap_sequences,
            evaluate_model
        )
        import numpy as np
        from sklearn.preprocessing import StandardScaler
        
        # Load data
        self._report('load_data', 0.15, 'Loading data...')
        data = load_data()
        
        # Build features
        self._report('build_features', 0.25, 'Building lap features...')
        lap_df = build_lap_features(data)
        
        # Prepare sequences
        self._report('prepare_sequences', 0.35, 'Preparing sequences...')
        X, y, feature_names = prepare_lap_sequences(
            lap_df, self.config['model']['sequence_length'])
        
        if len(X) == 0:
            raise RuntimeError("No sequences created from data!")
        
        # Split (temporal)
        n = len(X)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]
        
        # Scale
        scaler = StandardScaler()
        X_train = scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = scaler.transform(
            X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        X_test = scaler.transform(
            X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Build model
        self._report('build_model', 0.45, 'Building models...')
        models_dir = str(self.project_root / 'results' / 'models')
        merged = {**self.gpu_config, 'model': self.config['model']}
        self.model = GPUTyreModel(merged, models_dir=models_dir)
        self.model.build_models(
            input_dim=X_train.shape[-1],
            sequence_length=X_train.shape[1])
        
        # Train
        self._report('train', 0.55, 'Training...')
        self.model.train(
            X_train, y_train, X_val, y_val,
            epochs=self.config['model']['epochs'])
        
        # Evaluate
        self._report('evaluate', 0.85, 'Evaluating...')
        self.results = evaluate_model(self.model, X_test, y_test)
        
        # Save
        self._report('save', 0.95, 'Saving models...')
        self.model.save(str(self.project_root / 'results' / 'models' / 'gpu_ensemble'))
        
        self._report('save', 1.0, 'Pipeline complete!')
        
        return {
            'metrics': self.results,
            'model_path': models_dir,
            'history': self.model.history,
            'feature_names': feature_names,
        }

    def predict_strategy(self, params: dict) -> dict:
        """End-user entry point: accept race params, return strategy.
        
        Args:
            params: {circuit, driver, laps, air_temp, track_temp,
                     conditions, compounds}
        
        Returns:
            dict: {best_strategy, alternatives, confidence,
                   wear_curves, lap_times, pit_windows}
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        raise NotImplementedError("Strategy prediction — Phase 4")

    @staticmethod
    def get_default_config() -> dict:
        """Return full config with defaults. UI can auto-populate from this."""
        return {
            'data': {'years': [2022, 2023],
                     'sessions': ['Practice 3', 'Qualifying', 'Race']},
            'model': {'sequence_length': 5, 'epochs': 100,
                      'batch_size': 32, 'random_state': 42},
            'feature_engineering': {'enable_gpu_processing': True,
                                    'gpu_batch_size': 2000},
        }

    def _apply_overrides(self):
        """Deep-merge config_overrides into loaded config."""
        def _merge(base, override):
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    _merge(base[k], v)
                else:
                    base[k] = v
        if self.config:
            _merge(self.config, self.config_overrides)

    def _report(self, stage: str, progress: float, message: str):
        """Report progress to callback if set."""
        logger.info(f"[{stage}] {message}")
        if self._callback:
            self._callback(stage, progress, message)
