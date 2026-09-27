"""
Configuration Module for Fraud Detection Research Project.

Project: Hybrid Synthetic-Data and Anomaly-Aware Ensemble Learning for Financial Fraud Detection.
Contains all hyperparameters, file paths, model configurations, and pipeline settings.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class PathConfig:
    """Directory and file paths configuration."""
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = field(default=None)
    output_dir: Path = field(default=None)
    plots_dir: Path = field(default=None)
    models_dir: Path = field(default=None)
    
    # Model subdirectories
    ctgan_dir: Path = field(default=None)
    autoencoder_dir: Path = field(default=None)
    xgboost_dir: Path = field(default=None)
    lightgbm_dir: Path = field(default=None)
    catboost_dir: Path = field(default=None)
    rf_dir: Path = field(default=None)
    ensemble_dir: Path = field(default=None)
    
    # Data file paths
    raw_data_path: Path = field(default=None)
    synthetic_fraud_path: Path = field(default=None)
    augmented_train_path: Path = field(default=None)
    real_test_path: Path = field(default=None)
    experiment_results_path: Path = field(default=None)
    synthetic_quality_path: Path = field(default=None)

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.base_dir / "data"
        if self.output_dir is None:
            self.output_dir = self.base_dir / "output"
        if self.plots_dir is None:
            self.plots_dir = self.output_dir / "plots"
        if self.models_dir is None:
            self.models_dir = self.base_dir / "models"
            
        if self.ctgan_dir is None:
            self.ctgan_dir = self.models_dir / "ctgan"
        if self.autoencoder_dir is None:
            self.autoencoder_dir = self.models_dir / "autoencoder"
        if self.xgboost_dir is None:
            self.xgboost_dir = self.models_dir / "xgboost"
        if self.lightgbm_dir is None:
            self.lightgbm_dir = self.models_dir / "lightgbm"
        if self.catboost_dir is None:
            self.catboost_dir = self.models_dir / "catboost"
        if self.rf_dir is None:
            self.rf_dir = self.models_dir / "random_forest"
        if self.ensemble_dir is None:
            self.ensemble_dir = self.models_dir / "ensemble"
            
        if self.raw_data_path is None:
            self.raw_data_path = self.data_dir / "PS_20174392719_1491204439457_log.csv"
        if self.synthetic_fraud_path is None:
            self.synthetic_fraud_path = self.output_dir / "synthetic_fraud.csv"
        if self.augmented_train_path is None:
            self.augmented_train_path = self.output_dir / "augmented_training_data.csv"
        if self.real_test_path is None:
            self.real_test_path = self.output_dir / "real_test_data.csv"
        if self.experiment_results_path is None:
            self.experiment_results_path = self.output_dir / "experiment_results.csv"
        if self.synthetic_quality_path is None:
            self.synthetic_quality_path = self.output_dir / "synthetic_quality_report.csv"

    def create_dirs(self):
        """Ensure all required directories exist."""
        for d in [
            self.data_dir, self.output_dir, self.plots_dir, self.models_dir,
            self.ctgan_dir, self.autoencoder_dir, self.xgboost_dir, self.lightgbm_dir,
            self.catboost_dir, self.rf_dir, self.ensemble_dir
        ]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class SplitConfig:
    """Train/Test split configuration."""
    method: str = "temporal"  # "temporal" (primary) or "stratified"
    test_ratio: float = 0.20
    temporal_split_step: Optional[int] = None  # If None, computed as (1 - test_ratio) percentile of step
    random_state: int = 42


@dataclass
class CTGANConfig:
    """CTGAN synthetic data generation configuration."""
    n_synthetic_samples: int = 6000  # Number of synthetic fraud samples to generate
    epochs: int = 80                 # Training epochs for CTGAN
    batch_size: int = 500            # Batch size (must be divisible by pac, default pac=10)
    pac: int = 10
    random_state: int = 42
    generator_dim: tuple = (256, 256, 256)
    discriminator_dim: tuple = (256, 256, 256)
    verbose: bool = True


@dataclass
class AutoencoderConfig:
    """Autoencoder anomaly detector configuration."""
    encoding_dims: List[int] = field(default_factory=lambda: [64, 32, 16])
    latent_dim: int = 6
    learning_rate: float = 5e-4
    batch_size: int = 512
    epochs: int = 50
    early_stopping_patience: int = 8
    validation_split: float = 0.1
    random_state: int = 42
    dropout_rate: float = 0.2


@dataclass
class ModelHyperparameters:
    """Supervised models configuration."""
    random_state: int = 42
    
    # XGBoost hyperparameters
    xgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 400,
        "max_depth": 7,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.05,
        "reg_lambda": 1.5,
        "scale_pos_weight": 1.0,
        "n_jobs": -1,
        "eval_metric": "aucpr",
        "random_state": 42
    })
    
    # LightGBM hyperparameters
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 400,
        "max_depth": 8,
        "num_leaves": 63,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": 1.0,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1
    })
    
    # CatBoost hyperparameters
    cb_params: Dict[str, Any] = field(default_factory=lambda: {
        "iterations": 400,
        "depth": 7,
        "learning_rate": 0.03,
        "l2_leaf_reg": 5.0,
        "border_count": 128,
        "bagging_temperature": 0.5,
        "random_strength": 1.0,
        "random_seed": 42,
        "verbose": False,
        "thread_count": -1
    })
    
    # Random Forest hyperparameters
    rf_params: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": 42
    })


# Alias for backward-compatible import
ModelConfig = ModelHyperparameters


@dataclass
class ProjectConfig:
    """Master project configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    ctgan: CTGANConfig = field(default_factory=CTGANConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    models: ModelHyperparameters = field(default_factory=ModelHyperparameters)
    
    # General settings
    random_state: int = 42
    target_column: str = "isFraud"
    subsample_for_quick_test: Optional[int] = None  # None for full dataset


# Global default configuration instance
DEFAULT_CONFIG = ProjectConfig()
