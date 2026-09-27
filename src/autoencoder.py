"""
Autoencoder Anomaly Detection Module for PaySim Fraud Detection.

Part of Research Project:
Hybrid Synthetic-Data and Anomaly-Aware Ensemble Learning for Financial Fraud Detection.

Principles:
- Semi-Supervised Anomaly Detection: Trained strictly on legitimate (isFraud == 0)
  transactions from the training split. Never sees fraud or test records during training.
- Deep Bottleneck Architecture: Compresses inputs through encoder layers into a low-dimensional
  latent bottleneck, then reconstructs the input through symmetric decoder layers.
- Reconstruction Error Metric: Legitimate patterns are reconstructed with low MSE loss;
  novel, rare, or fraudulent patterns suffer elevated reconstruction error.
- Feature Extraction Role: The resulting sample-wise error is scaled to produce the
  `anomaly_score` feature for the downstream ensemble, not used as a standalone classifier.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import joblib

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import DEFAULT_CONFIG, ProjectConfig, AutoencoderConfig, PathConfig


class PyTorchAutoencoder(nn.Module):
    """Deep fully connected Autoencoder with bottleneck compression and Dropout regularization."""

    def __init__(self, input_dim: int, latent_dim: int = 6, encoding_dims: Optional[List[int]] = None, dropout_rate: float = 0.2):
        super().__init__()
        dims = encoding_dims or [64, 32, 16]

        # Encoder
        encoder_layers = []
        in_d = input_dim
        for d in dims:
            encoder_layers.extend([
                nn.Linear(in_d, d),
                nn.BatchNorm1d(d),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
            ])
            in_d = d
        encoder_layers.append(nn.Linear(in_d, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (no dropout in decoder for clean reconstruction)
        decoder_layers = []
        in_d = latent_dim
        for d in reversed(dims):
            decoder_layers.extend([
                nn.Linear(in_d, d),
                nn.BatchNorm1d(d),
                nn.LeakyReLU(0.1),
            ])
            in_d = d
        decoder_layers.append(nn.Linear(in_d, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class AutoencoderAnomalyDetector:
    """
    Complete Pipeline Wrapper for Autoencoder Anomaly Detection.
    Handles data scaling, model training on legitimate data, early stopping,
    and reconstruction error / anomaly score generation.
    """

    def __init__(self, config: Optional[AutoencoderConfig] = None):
        self.config = config or DEFAULT_CONFIG.autoencoder
        self.model: Optional[PyTorchAutoencoder] = None
        self.scaler = RobustScaler()
        self.score_scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def _prepare_tensors(self, X: pd.DataFrame, is_train: bool = True) -> np.ndarray:
        """Scale continuous inputs using RobustScaler."""
        if is_train:
            self.feature_names = list(X.columns)
            scaled = self.scaler.fit_transform(X.values)
        else:
            scaled = self.scaler.transform(X[self.feature_names].values)
        return np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

    def fit(
        self,
        X_train_df: pd.DataFrame,
        y_train: Optional[pd.Series] = None,
        val_split: float = 0.1,
    ):
        """
        Fit Autoencoder strictly on legitimate (isFraud == 0) training samples.
        """
        # Filter strictly to legitimate transactions
        if y_train is not None:
            legit_mask = (y_train.values == 0)
            X_legit = X_train_df[legit_mask].copy()
            print(f"[*] Autoencoder training on {len(X_legit):,} legitimate records (filtered from {len(X_train_df):,} total).")
        else:
            X_legit = X_train_df.copy()
            print(f"[*] Autoencoder training on {len(X_legit):,} records.")

        X_scaled = self._prepare_tensors(X_legit, is_train=True)
        input_dim = X_scaled.shape[1]

        # Train / Validation split
        n_samples = len(X_scaled)
        n_val = int(n_samples * val_split)
        indices = np.random.RandomState(self.config.random_state).permutation(n_samples)
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        train_data = torch.tensor(X_scaled[train_idx], dtype=torch.float32)
        val_data = torch.tensor(X_scaled[val_idx], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(train_data),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=(len(train_data) > self.config.batch_size),
        )

        self.model = PyTorchAutoencoder(
            input_dim=input_dim,
            latent_dim=self.config.latent_dim,
            encoding_dims=self.config.encoding_dims,
            dropout_rate=getattr(self.config, 'dropout_rate', 0.2),
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5,
        )

        print(f"[*] Training Autoencoder on device: {self.device}")
        print(f"    - Input Dim   : {input_dim}")
        print(f"    - Latent Dim  : {self.config.latent_dim}")
        print(f"    - Architecture: {input_dim} -> {self.config.encoding_dims} -> {self.config.latent_dim}")
        print(f"    - Max Epochs  : {self.config.epochs}")
        print(f"    - Batch Size  : {self.config.batch_size}")

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            train_loss = 0.0
            for (batch_x,) in train_loader:
                batch_x = batch_x.to(self.device)
                optimizer.zero_grad()
                recon = self.model(batch_x)
                loss = criterion(recon, batch_x)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)
            train_loss /= len(train_data)

            # Validation loss
            self.model.eval()
            with torch.no_grad():
                val_x = val_data.to(self.device)
                val_recon = self.model(val_x)
                val_loss = criterion(val_recon, val_x).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 5 == 0 or epoch == self.config.epochs:
                print(f"    Epoch {epoch:2d}/{self.config.epochs:2d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Patience: {patience_counter}/{self.config.early_stopping_patience}")

            if patience_counter >= self.config.early_stopping_patience:
                print(f"[*] Early stopping triggered at epoch {epoch}. Restoring best weights (Val Loss: {best_val_loss:.6f}).")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Fit score scaler on legitimate reconstruction errors using robust percentile clipping
        raw_train_errors = self._compute_reconstruction_error(X_scaled)
        # Use 99th percentile as upper bound for MinMaxScaler to prevent outlier saturation
        p99 = np.percentile(raw_train_errors, 99)
        clipped_errors = np.clip(raw_train_errors, 0, p99)
        self.score_scaler.fit(clipped_errors.reshape(-1, 1))
        self._p99_clip = float(p99)
        self.is_fitted = True
        print("[+] Autoencoder training and calibration completed.")
        return self

    def _compute_reconstruction_error(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute sample-wise Mean Squared Error reconstruction loss."""
        self.model.eval()
        loader = DataLoader(TensorDataset(torch.tensor(X_scaled, dtype=torch.float32)), batch_size=1024, shuffle=False)
        errors = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                recon = self.model(batch_x)
                batch_err = torch.mean((batch_x - recon) ** 2, dim=1).cpu().numpy()
                errors.append(batch_err)

        return np.concatenate(errors)

    def compute_anomaly_score(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Compute normalized anomaly_score in [0, 1] for arbitrary transaction batches.
        """
        if not self.is_fitted:
            raise RuntimeError("Autoencoder must be fitted before predicting anomaly scores.")

        X_scaled = self._prepare_tensors(X_df, is_train=False)
        raw_errors = self._compute_reconstruction_error(X_scaled)
        # Clip to the 99th percentile learned on training legit data before scaling
        p99 = getattr(self, '_p99_clip', float(np.percentile(raw_errors, 99)))
        clipped = np.clip(raw_errors, 0, p99)
        normalized_scores = self.score_scaler.transform(clipped.reshape(-1, 1)).flatten()
        return np.clip(normalized_scores, 0.0, 1.0)

    def append_anomaly_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append 'anomaly_score' column directly to DataFrame."""
        out_df = df.copy()
        out_df['anomaly_score'] = self.compute_anomaly_score(df)
        return out_df

    def save(self, model_dir: Path):
        """Save model weights and scalers to directory."""
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_dir / "autoencoder.pt")
        joblib.dump(self.scaler, model_dir / "input_scaler.joblib")
        joblib.dump(self.score_scaler, model_dir / "score_scaler.joblib")
        joblib.dump(self.feature_names, model_dir / "feature_names.joblib")
        print(f"[+] Saved Autoencoder model artifacts -> {model_dir}")

    @classmethod
    def load(cls, model_dir: Path, config: Optional[AutoencoderConfig] = None) -> "AutoencoderAnomalyDetector":
        """Load trained Autoencoder detector from disk."""
        detector = cls(config=config)
        detector.scaler = joblib.load(model_dir / "input_scaler.joblib")
        detector.score_scaler = joblib.load(model_dir / "score_scaler.joblib")
        detector.feature_names = joblib.load(model_dir / "feature_names.joblib")

        detector.model = PyTorchAutoencoder(
            input_dim=len(detector.feature_names),
            latent_dim=detector.config.latent_dim,
            encoding_dims=detector.config.encoding_dims,
            dropout_rate=getattr(detector.config, 'dropout_rate', 0.2),
        ).to(detector.device)

        detector.model.load_state_dict(torch.load(model_dir / "autoencoder.pt", map_location=detector.device))
        detector.model.eval()
        detector.is_fitted = True
        return detector


def train_autoencoder(
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    config: Optional[AutoencoderConfig] = None,
    path_config: Optional[PathConfig] = None,
) -> AutoencoderAnomalyDetector:
    """Convenience function to train, evaluate, and save Autoencoder."""
    paths = path_config or DEFAULT_CONFIG.paths
    detector = AutoencoderAnomalyDetector(config=config)

    if feature_cols is None:
        exclude_cols = ['nameOrig', 'nameDest', 'isFraud', 'isFlaggedFraud', 'type']
        feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    y_train = train_df['isFraud'] if 'isFraud' in train_df.columns else None
    detector.fit(train_df[feature_cols], y_train=y_train)
    detector.save(paths.autoencoder_dir)
    return detector


if __name__ == "__main__":
    from src.data_loader import load_dataset, split_dataset
    from src.feature_engineering import PaySimFeatureEngineer
    from src.preprocessing import PaySimPreprocessor

    print("[*] Testing src/autoencoder.py...")
    raw = load_dataset(nrows=20000)
    train_part, test_part, _ = split_dataset(raw, method="stratified")

    # Feature engineering + preprocessing
    fe = PaySimFeatureEngineer()
    train_fe = fe.fit_transform(train_part)
    test_fe = fe.transform(test_part)

    pre = PaySimPreprocessor()
    train_prep = pre.fit_transform(train_fe)
    test_prep = pre.transform(test_fe)

    # Train fast autoencoder
    fast_ae_cfg = AutoencoderConfig(epochs=6, batch_size=256, early_stopping_patience=2)
    detector = train_autoencoder(train_prep, config=fast_ae_cfg)

    # Test scoring
    train_scores = detector.compute_anomaly_score(train_prep)
    test_scores = detector.compute_anomaly_score(test_prep)

    print(f"[+] Train Anomaly Score: Mean={train_scores.mean():.4f}, Max={train_scores.max():.4f}")
    print(f"[+] Test Anomaly Score : Mean={test_scores.mean():.4f}, Max={test_scores.max():.4f}")

    # Inspect fraud vs legitimate scores
    fraud_mask = (test_prep['isFraud'] == 1).values
    if fraud_mask.sum() > 0:
        fraud_scores = test_scores[fraud_mask]
        legit_scores = test_scores[~fraud_mask]
        print(f"[+] Test Fraud Avg Anomaly Score : {fraud_scores.mean():.4f}")
        print(f"[+] Test Legit Avg Anomaly Score : {legit_scores.mean():.4f}")

    print("[+] Autoencoder smoke test passed!")
