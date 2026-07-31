"""Environment setup utilities for F1 tyre model."""
import fastf1
import os
import yaml
import logging
from pathlib import Path


def setup_environment(project_root=None):
    """Initialize project environment with logging and directories.
    
    Args:
        project_root: Path to project root. If None, uses CWD.
    """
    root = Path(project_root) if project_root else Path.cwd()
    
    # Create logs directory first to enable file handler
    log_dir = root / 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'f1_model.log'),
            logging.StreamHandler()
        ]
    )
    
    # Create required directories
    directories = [
        'data/raw', 'data/processed', 'data/external', 'data/cache',
        'results/models', 'results/plots', 'results/reports', 'logs'
    ]
    for directory in directories:
        os.makedirs(root / directory, exist_ok=True)
    
    # Configure FastF1 cache
    cache_dir = root / 'data' / 'cache' / 'fastf1_cache'
    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    
    logging.info(f"Environment initialized. Cache: {cache_dir}")
    return True


def load_config(config_path='config/config.yaml'):
    """Load YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        dict: Configuration dictionary or None if file not found.
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.error(f"Config not found: {config_path}")
        return None
