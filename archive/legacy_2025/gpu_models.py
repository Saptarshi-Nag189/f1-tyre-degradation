"""GPU-optimized models for tyre degradation prediction."""
import torch
import torch.nn as nn
import numpy as np
import logging
import os
from typing import Dict, Tuple, Optional
import joblib

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from cuml.ensemble import RandomForestRegressor as cuRF
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

GPU_ML_AVAILABLE = XGB_AVAILABLE and CUML_AVAILABLE


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism for sequence modeling.
    
    Architecture per Implementation Plan:
    - 2-layer LSTM with configurable hidden dimension
    - Multi-head self-attention (4 heads default)
    - Dense output layers
    
    Input shape: (batch, sequence_length, input_dim)
    Output shape: (batch, 1)
    
    Attributes:
        hidden_dim: LSTM hidden dimension.
        num_layers: Number of LSTM layers.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, attention_heads: int = 4):
        """Initialize the AttentionLSTM model.
        
        Args:
            input_dim: Number of input features per timestep.
            hidden_dim: LSTM hidden state dimension.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout rate for regularization.
            attention_heads: Number of multi-head attention heads.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout, batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim).
            
        Returns:
            Tensor: Output predictions of shape (batch, 1).
        """
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use last timestep
        final_hidden = attn_out[:, -1, :]
        
        # Prediction
        output = self.fc(final_hidden)
        return output


class GPUTyreModel:
    """Ensemble model combining LSTM, XGBoost GPU, and cuML.
    
    Ensemble weights per Implementation Plan:
    - LSTM: 0.4
    - XGBoost: 0.4
    - cuML RF: 0.2
    
    Attributes:
        config: Configuration dictionary.
        device: PyTorch device ('cuda' or 'cpu').
        lstm_model: AttentionLSTM model.
        xgb_model: XGBoost regressor.
        cuml_model: cuML Random Forest.
        history: Training history dictionary.
    """
    
    def __init__(self, config: Dict, models_dir: str = 'results/models'):
        """Initialize the GPUTyreModel ensemble.
        
        Args:
            config: GPU configuration dictionary with model parameters.
            models_dir: Directory for saving/loading model checkpoints.
        """
        self.config = config
        self.models_dir = str(models_dir)
        self.logger = logging.getLogger(__name__)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model components
        self.lstm_model = None
        self.xgb_model = None
        self.cuml_model = None
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': []}
        
    def build_models(self, input_dim: int, sequence_length: int):
        """Initialize all model components.
        
        Args:
            input_dim: Number of input features per timestep.
            sequence_length: Length of input sequences.
        """
        # LSTM model per Implementation Plan
        lstm_config = self.config['models']['lstm']
        self.lstm_model = AttentionLSTM(
            input_dim=input_dim,
            hidden_dim=lstm_config['hidden_dim'],
            num_layers=lstm_config['num_layers'],
            dropout=lstm_config['dropout'],
            attention_heads=lstm_config['attention_heads']
        ).to(self.device)
        
        self.logger.info(f"LSTM model built: {input_dim} inputs, {lstm_config['hidden_dim']} hidden")
        
        # XGBoost model — reads config for tree_method, lr, and device
        if XGB_AVAILABLE:
            xgb_config = self.config['models']['xgboost']
            self.xgb_model = xgb.XGBRegressor(
                tree_method=xgb_config.get('tree_method', 'hist'),
                device='cuda' if self.device == 'cuda' else 'cpu',
                learning_rate=xgb_config.get('learning_rate', 0.05),
                max_depth=xgb_config['max_depth'],
                n_estimators=xgb_config['n_estimators'],
                random_state=42
            )
            self.logger.info(f"XGBoost initialized (device={self.device}, lr={xgb_config.get('learning_rate', 0.05)})")
        else:
            self.logger.warning("XGBoost not available")
            
        # cuML Random Forest
        if CUML_AVAILABLE:
            self.cuml_model = cuRF(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.logger.info("cuML Random Forest initialized")
        else:
            self.logger.warning("cuML not available, using minimal ensemble")
    
    def train(self, X_train, y_train, X_val, y_val, epochs: int = 100):
        """Train all models in the ensemble.
        
        Args:
            X_train: Training features array (n_samples, seq_len, n_features).
            y_train: Training targets array.
            X_val: Validation features array.
            y_val: Validation targets array.
            epochs: Maximum training epochs.
        """
        self.logger.info("Training ensemble models...")
        
        # Train LSTM
        self._train_lstm(X_train, y_train, X_val, y_val, epochs)
        
        # Train XGBoost on flattened features - ensure float32 contiguous arrays
        X_train_flat = np.ascontiguousarray(X_train.reshape(X_train.shape[0], -1), dtype=np.float32)
        X_val_flat = np.ascontiguousarray(X_val.reshape(X_val.shape[0], -1), dtype=np.float32)
        y_train_np = np.ascontiguousarray(y_train, dtype=np.float32)
        y_val_np = np.ascontiguousarray(y_val, dtype=np.float32)
        
        if self.xgb_model:
            self.logger.info("Training XGBoost model...")
            self.xgb_model.fit(
                X_train_flat, y_train_np,
                eval_set=[(X_val_flat, y_val_np)],
                verbose=False
            )
        
        # Train cuML model
        if self.cuml_model:
            self.logger.info("Training cuML Random Forest...")
            try:
                import cudf
                X_train_cudf = cudf.DataFrame(X_train_flat)
                y_train_cudf = cudf.Series(y_train)
                self.cuml_model.fit(X_train_cudf, y_train_cudf)
            except Exception as e:
                self.logger.warning(f"cuML training failed: {e}")
                self.cuml_model = None
    
    def _train_lstm(self, X_train, y_train, X_val, y_val, epochs: int):
        """Train LSTM model with early stopping.
        
        Saves best model to results/models/best_lstm.pth.
        
        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features.
            y_val: Validation targets.
            epochs: Maximum epochs.
        """
        self.logger.info("Training LSTM model...")
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)
        
        # Setup training - use Huber loss for robustness to outliers
        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(
            self.lstm_model.parameters(),
            lr=self.config['models']['lstm'].get('learning_rate', 0.001)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            patience=self.config['optimization'].get('lr_schedule_patience', 10)
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['optimization']['early_stopping_patience']
        
        # Get batch size from config - handle nested structure
        batch_size = self.config.get('model', {}).get('batch_size', 32)
        if isinstance(batch_size, dict):
            batch_size = 32
        
        # AMP: Mixed precision for ~40% VRAM savings on RTX 40-series
        use_amp = (self.device == 'cuda')
        grad_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        if use_amp:
            self.logger.info("AMP mixed-precision enabled")
        
        for epoch in range(epochs):
            # Training phase
            self.lstm_model.train()
            train_loss = 0
            n_batches = 0
            
            for i in range(0, len(X_train_t), batch_size):
                batch_X = X_train_t[i:i+batch_size]
                batch_y = y_train_t[i:i+batch_size]
                
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = self.lstm_model(batch_X)
                    loss = criterion(outputs, batch_y)
                grad_scaler.scale(loss).backward()
                
                # Gradient clipping (unscale first for correct norm)
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.lstm_model.parameters(),
                    self.config['optimization']['gradient_clip_norm']
                )
                
                grad_scaler.step(optimizer)
                grad_scaler.update()
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= max(n_batches, 1)
            
            # Validation phase (batched + AMP)
            self.lstm_model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_batches = 0
                for j in range(0, len(X_val_t), batch_size):
                    xv = X_val_t[j:j+batch_size]
                    yv = y_val_t[j:j+batch_size]
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        val_out = self.lstm_model(xv)
                        val_loss += criterion(val_out, yv).item()
                    val_batches += 1
                val_loss /= max(val_batches, 1)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                best_path = os.path.join(self.models_dir, 'best_lstm.pth')
                os.makedirs(self.models_dir, exist_ok=True)
                torch.save(self.lstm_model.state_dict(), best_path)
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        best_path = os.path.join(self.models_dir, 'best_lstm.pth')
        if os.path.exists(best_path):
            self.lstm_model.load_state_dict(
                torch.load(best_path, weights_only=True)
            )
    
    def predict(self, X) -> np.ndarray:
        """Generate ensemble predictions.
        
        Uses weighted average per Implementation Plan:
        LSTM (0.4) + XGBoost (0.4) + cuML (0.2)
        
        Args:
            X: Input array (n_samples, seq_len, n_features).
            
        Returns:
            np.ndarray: Predictions of shape (n_samples,).
        """
        predictions = []
        weights = self.config['models']['ensemble']['weights']
        
        # LSTM predictions (batched + AMP)
        if self.lstm_model:
            self.lstm_model.eval()
            use_amp = (self.device == 'cuda')
            with torch.no_grad():
                lstm_parts = []
                for j in range(0, len(X), 1024):
                    X_t = torch.FloatTensor(X[j:j+1024]).to(self.device)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        lstm_parts.append(self.lstm_model(X_t).cpu().numpy().flatten())
                lstm_pred = np.concatenate(lstm_parts)
                predictions.append((lstm_pred, weights['lstm']))
        
        # XGBoost predictions
        if self.xgb_model:
            X_flat = X.reshape(X.shape[0], -1)
            xgb_pred = self.xgb_model.predict(X_flat)
            predictions.append((xgb_pred, weights['xgboost']))
        
        # cuML predictions
        if self.cuml_model:
            try:
                import cudf
                X_flat = X.reshape(X.shape[0], -1)
                X_cudf = cudf.DataFrame(X_flat)
                cuml_pred = self.cuml_model.predict(X_cudf).to_numpy()
                predictions.append((cuml_pred, weights['cuml']))
            except Exception:
                pass
        
        # Weighted ensemble
        if predictions:
            total_weight = sum(w for _, w in predictions)
            ensemble_pred = sum(pred * w for pred, w in predictions) / total_weight
            return ensemble_pred
        
        return np.zeros(X.shape[0])
    
    def save(self, path: str = 'results/models/gpu_ensemble'):
        """Save all models to disk.
        
        Args:
            path: Base path for saving models (without extension).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.lstm_model:
            torch.save(self.lstm_model.state_dict(), f'{path}_lstm.pth')
        if self.xgb_model:
            joblib.dump(self.xgb_model, f'{path}_xgb.pkl')  # Use joblib instead of save_model
        if self.cuml_model:
            joblib.dump(self.cuml_model, f'{path}_cuml.pkl')
        joblib.dump(self.history, f'{path}_history.pkl')
        self.logger.info(f"Models saved to {path}")
    
    def load(self, path: str = 'results/models/gpu_ensemble'):
        """Load all models from disk.
        
        Args:
            path: Base path for loading models (without extension).
        """
        try:
            if self.lstm_model and os.path.exists(f'{path}_lstm.pth'):
                self.lstm_model.load_state_dict(
                    torch.load(f'{path}_lstm.pth', weights_only=True)
                )
            if os.path.exists(f'{path}_xgb.pkl'):
                self.xgb_model = joblib.load(f'{path}_xgb.pkl')
            if os.path.exists(f'{path}_cuml.pkl'):
                self.cuml_model = joblib.load(f'{path}_cuml.pkl')
            if os.path.exists(f'{path}_history.pkl'):
                self.history = joblib.load(f'{path}_history.pkl')
            self.logger.info(f"Models loaded from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")