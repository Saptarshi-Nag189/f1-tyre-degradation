"""GPU memory management for RTX 4050."""
import torch
import gc
import logging
from contextlib import contextmanager

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class GPUMemoryManager:
    """Manages GPU memory with automatic cleanup for 6GB VRAM constraint.
    
    Attributes:
        device: Target device ('cuda' or 'cpu').
        cuda_available: Whether CUDA is available on this system.
        memory_pool: CuPy memory pool if available.
        pinned_pool: CuPy pinned memory pool if available.
    """
    
    def __init__(self, device='cuda'):
        """Initialize the GPU memory manager.
        
        Args:
            device: Target device, 'cuda' or 'cpu'.
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.cuda_available = torch.cuda.is_available()
        
        if self.cuda_available and CUPY_AVAILABLE:
            self.memory_pool = cp.get_default_memory_pool()
            self.pinned_pool = cp.get_default_pinned_memory_pool()
        else:
            self.memory_pool = None
            self.pinned_pool = None
    
    @contextmanager
    def gpu_memory_context(self):
        """Context manager ensuring GPU memory cleanup.
        
        Yields control and clears GPU memory on exit, regardless of 
        whether an exception occurred.
        
        Yields:
            None
        """
        try:
            yield
        finally:
            self.clear_memory()
    
    def clear_memory(self):
        """Clear all GPU memory caches.
        
        Clears PyTorch CUDA cache, CuPy memory pools, and runs
        Python garbage collection.
        """
        if self.cuda_available:
            torch.cuda.empty_cache()
            if CUPY_AVAILABLE and self.memory_pool:
                self.memory_pool.free_all_blocks()
                self.pinned_pool.free_all_blocks()
        gc.collect()
    
    def get_memory_info(self) -> dict:
        """Return current GPU memory usage.
        
        Returns:
            dict: Memory usage information with keys:
                - torch_allocated_gb: PyTorch allocated memory in GB
                - torch_cached_gb: PyTorch cached memory in GB
                - cupy_used_gb: CuPy used memory in GB (if available)
                - cupy_total_gb: CuPy total memory in GB (if available)
                - error: Error message if CUDA not available
        """
        if not self.cuda_available:
            return {'torch_allocated_gb': 0, 'torch_cached_gb': 0}
        
        info = {
            'torch_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'torch_cached_gb': torch.cuda.memory_reserved() / 1024**3
        }
        
        if CUPY_AVAILABLE and self.memory_pool:
            info['cupy_used_gb'] = self.memory_pool.used_bytes() / 1024**3
            info['cupy_total_gb'] = self.memory_pool.total_bytes() / 1024**3
        
        return info
    
    def optimize_batch_size(self, base_batch_size: int = 1000, data_size: int = None) -> int:
        """Calculate optimal batch size for available GPU memory.
        
        Uses conservative estimate for RTX 4050 (5.5GB usable out of 6GB).
        
        Args:
            base_batch_size: Initial batch size to start from.
            data_size: Size of each data element in features (optional).
            
        Returns:
            int: Optimal batch size that fits within memory constraints.
        """
        # Conservative estimate for RTX 4050
        available_memory_gb = 5.5
        
        if data_size:
            # Estimate memory per sample (float32 = 4 bytes)
            memory_per_sample_gb = (data_size * 4) / (1024**3)
            # Account for forward and backward pass (factor of 2)
            max_batch = int(available_memory_gb / (memory_per_sample_gb * 2))
            return min(base_batch_size, max_batch)
        
        return base_batch_size
