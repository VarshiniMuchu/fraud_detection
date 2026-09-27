"""
Synthetic Data Generation Module using CTGAN for PaySim Fraud Detection.

Part of Research Project:
Hybrid Synthetic-Data and Anomaly-Aware Ensemble Learning for Financial Fraud Detection.

Principles:
- Strict Zero Data Leakage: CTGAN is trained exclusively on minority-class fraud
  samples extracted from the training partition. The test partition is never accessed.
- Conditional GAN Architecture: Employs mode-specific normalization and conditional
  generators to accurately capture multi-modal distributions and extreme transaction amounts.
- Financial Domain Constraints: Enforces validity of transaction steps, non-negative
  amounts/balances, and consistency with financial transaction types.
- Artifact Persistence: Generates synthetic fraud datasets, augmented training sets,
  and exports the trained synthesizer for reproducibility and TSTR evaluation.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from ctgan import CTGAN

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import DEFAULT_CONFIG, ProjectConfig, CTGANConfig, PathConfig
from src.preprocessing import prepare_for_ctgan


def train_ctgan_synthesizer(
    fraud_df: pd.DataFrame,
    ctgan_config: Optional[CTGANConfig] = None,
    discrete_columns: Optional[List[str]] = None,
) -> CTGAN:
    """
    Train a CTGAN synthesizer strictly on real training-split fraud records.

    Args:
        fraud_df: DataFrame containing only fraud transactions from training split.
        ctgan_config: Hyperparameters for CTGAN training (epochs, batch_size, dims, etc.).
        discrete_columns: List of categorical column names (e.g. ['type']).

    Returns:
        Fitted CTGAN synthesizer model.
    """
    cfg = ctgan_config or DEFAULT_CONFIG.ctgan
    
    if len(fraud_df) == 0:
        raise ValueError("Cannot train CTGAN on an empty fraud dataset.")

    if discrete_columns is None:
        discrete_columns = ['type'] if 'type' in fraud_df.columns else []

    # Filter discrete_columns to only those present in fraud_df
    discrete_columns = [col for col in discrete_columns if col in fraud_df.columns]

    print(f"[*] Initializing CTGAN synthesizer:")
    print(f"    - Fraud training samples : {len(fraud_df):,}")
    print(f"    - Epochs                 : {cfg.epochs}")
    print(f"    - Batch size             : {cfg.batch_size}")
    print(f"    - PAC                    : {cfg.pac}")
    print(f"    - Generator Dim          : {cfg.generator_dim}")
    print(f"    - Discriminator Dim      : {cfg.discriminator_dim}")
    print(f"    - Discrete columns       : {discrete_columns}")

    # Ensure pac divides batch_size evenly as required by CTGAN
    batch_size = cfg.batch_size
    if batch_size % cfg.pac != 0:
        batch_size = (batch_size // cfg.pac) * cfg.pac
        print(f"    [!] Adjusted batch_size to {batch_size} to be divisible by pac={cfg.pac}")

    # Ensure batch_size does not exceed sample count
    if batch_size > len(fraud_df):
        pac = min(cfg.pac, max(1, len(fraud_df) // 2))
        batch_size = len(fraud_df) - (len(fraud_df) % pac)
        print(f"    [!] Small dataset: adjusted batch_size={batch_size}, pac={pac}")
    else:
        pac = cfg.pac

    # Clean categorical types to strings to avoid CategoricalDtype issues in CTGAN
    df_to_fit = fraud_df.copy()
    
    # Exclude constant target column 'isFraud' from GAN training
    if 'isFraud' in df_to_fit.columns:
        df_to_fit = df_to_fit.drop(columns=['isFraud'])

    for col in discrete_columns:
        if col in df_to_fit.columns:
            df_to_fit[col] = df_to_fit[col].astype(str)

    # Cast continuous numeric columns to float64 to prevent Pandas 2.x lossless casting errors in RDT
    for col in df_to_fit.columns:
        if col not in discrete_columns:
            df_to_fit[col] = pd.to_numeric(df_to_fit[col], errors='coerce').astype(np.float64)

    model = CTGAN(
        embedding_dim=128,
        generator_dim=cfg.generator_dim,
        discriminator_dim=cfg.discriminator_dim,
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=batch_size,
        discriminator_steps=1,
        log_frequency=True,
        verbose=cfg.verbose,
        epochs=cfg.epochs,
        pac=pac,
        enable_gpu=True,
    )

    print("[*] Training CTGAN on fraud records...")
    model.fit(df_to_fit, discrete_columns=discrete_columns)
    print("[+] CTGAN training completed.")
    return model


def postprocess_synthetic_samples(
    synthetic_df: pd.DataFrame,
    reference_fraud_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Enforce financial domain validity constraints on synthetic transactions.

    Rules applied:
    - Non-negative balances and transaction amounts: clipped to >= 0.0.
    - Transaction step: integer >= 1.
    - Currency precision: rounded to 2 decimal places.
    - Class label: strictly 1 (synthetic fraud).
    - Transaction types: restricted to types observed in genuine fraud.
    """
    df = synthetic_df.copy()

    # Financial balance and amount clipping
    monetary_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    for col in monetary_cols:
        if col in df.columns:
            df[col] = df[col].astype(float).clip(lower=0.0)
            df[col] = df[col].round(2)

    # Step constraint
    if 'step' in df.columns:
        df['step'] = df['step'].astype(float).round().clip(lower=1.0).astype(int)

    # Ensure target column is strictly fraud
    df['isFraud'] = 1

    # Filter or re-map types if reference data provided
    if reference_fraud_df is not None and 'type' in reference_fraud_df.columns:
        valid_types = set(reference_fraud_df['type'].unique())
        if 'type' in df.columns:
            # Map any rare hallucinated types to the most prevalent fraud type
            primary_type = reference_fraud_df['type'].mode()[0]
            df['type'] = df['type'].apply(lambda t: t if t in valid_types else primary_type)

    return df


def generate_synthetic_fraud(
    model: CTGAN,
    n_samples: int,
    reference_fraud_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Sample synthetic fraud records from the trained CTGAN and enforce validity constraints.

    Args:
        model: Trained CTGAN instance.
        n_samples: Number of synthetic fraud samples to generate.
        reference_fraud_df: Genuine fraud records used for constraint mapping.

    Returns:
        Post-processed DataFrame of synthetic fraud records.
    """
    print(f"[*] Sampling {n_samples:,} synthetic fraud transactions from CTGAN...")
    raw_samples = model.sample(n_samples)
    synthetic_df = postprocess_synthetic_samples(raw_samples, reference_fraud_df=reference_fraud_df)
    print(f"[+] Sampled and post-processed {len(synthetic_df):,} synthetic fraud records.")
    return synthetic_df


def create_augmented_training_data(
    real_train_df: pd.DataFrame,
    synthetic_fraud_df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Combine genuine training data with generated synthetic fraud records.

    Ensures column alignment by adding synthetic identifier placeholders
    ('nameOrig', 'nameDest', 'isFlaggedFraud') if present in real training data.

    Args:
        real_train_df: Original training DataFrame.
        synthetic_fraud_df: Synthetic fraud records.
        random_state: Seed for reproducible record shuffling.

    Returns:
        Shuffled augmented training DataFrame.
    """
    synth = synthetic_fraud_df.copy()
    real = real_train_df.copy()

    # Align columns if real data contains identifier strings
    for col in real.columns:
        if col not in synth.columns:
            if col == 'nameOrig':
                synth['nameOrig'] = [f"SYNTH_ORIG_{i}" for i in range(len(synth))]
            elif col == 'nameDest':
                synth['nameDest'] = [f"SYNTH_DEST_{i}" for i in range(len(synth))]
            elif col == 'isFlaggedFraud':
                synth['isFlaggedFraud'] = 0
            else:
                synth[col] = 0

    # Ensure identical column order
    synth = synth[real.columns]

    # Concatenate and shuffle
    augmented_df = pd.concat([real, synth], axis=0, ignore_index=True)
    augmented_df = augmented_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    real_frauds = (real['isFraud'] == 1).sum()
    synth_frauds = len(synth)
    total_frauds = (augmented_df['isFraud'] == 1).sum()
    total_samples = len(augmented_df)

    print(f"[+] Augmented Training Data Summary:")
    print(f"    - Original train size      : {len(real):,}")
    print(f"    - Genuine fraud instances  : {real_frauds:,} ({real_frauds / len(real):.4%})")
    print(f"    - Synthetic fraud added    : {synth_frauds:,}")
    print(f"    - Total augmented size     : {total_samples:,}")
    print(f"    - Total fraud instances    : {total_frauds:,} ({total_frauds / total_samples:.4%})")

    return augmented_df


def save_synthetic_artifacts(
    synthetic_df: pd.DataFrame,
    augmented_df: pd.DataFrame,
    model: CTGAN,
    path_config: Optional[PathConfig] = None,
) -> Dict[str, Path]:
    """
    Persist synthetic fraud data, augmented training set, and trained CTGAN model.

    Returns:
        Dictionary mapping artifact names to their file paths.
    """
    paths = path_config or DEFAULT_CONFIG.paths
    paths.create_dirs()

    # 1. Save synthetic fraud CSV
    synth_path = paths.synthetic_fraud_path
    synthetic_df.to_csv(synth_path, index=False)
    print(f"[+] Saved synthetic fraud data -> {synth_path}")

    # 2. Save augmented training CSV
    aug_path = paths.augmented_train_path
    augmented_df.to_csv(aug_path, index=False)
    print(f"[+] Saved augmented training data -> {aug_path}")

    # 3. Save trained CTGAN model
    model_path = paths.ctgan_dir / "ctgan_synthesizer.pkl"
    model.save(str(model_path))
    print(f"[+] Saved trained CTGAN model -> {model_path}")

    return {
        "synthetic_fraud": synth_path,
        "augmented_train": aug_path,
        "ctgan_model": model_path,
    }


def run_synthetic_generation_pipeline(
    train_df: pd.DataFrame,
    config: Optional[ProjectConfig] = None,
    save_artifacts: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, CTGAN]:
    """
    End-to-end execution of the CTGAN Synthetic Fraud Generation pipeline.

    Steps:
    1. Extract genuine fraud cases strictly from training partition.
    2. Train CTGAN synthesizer on fraud distributions.
    3. Generate n_synthetic_samples of synthetic fraud.
    4. Postprocess with financial constraints.
    5. Construct augmented training dataset.
    6. Persist outputs to output/ and models/ directories.

    Args:
        train_df: The training split DataFrame (clean of test records).
        config: Project configuration instance.
        save_artifacts: Whether to write outputs to disk.

    Returns:
        Tuple of (synthetic_fraud_df, augmented_train_df, trained_model).
    """
    cfg = config or DEFAULT_CONFIG
    print("=" * 70)
    print("STAGE 4: CTGAN SYNTHETIC FRAUD GENERATION PIPELINE")
    print("=" * 70)

    # Step 1: Extract fraud rows strictly from train split
    fraud_df = prepare_for_ctgan(train_df, target_col="isFraud")
    n_real_fraud = len(fraud_df)
    print(f"[+] Isolated {n_real_fraud:,} real fraud transactions from training partition.")

    if n_real_fraud == 0:
        raise ValueError("No fraud records found in the provided training split.")

    # Step 2: Train CTGAN
    model = train_ctgan_synthesizer(
        fraud_df=fraud_df,
        ctgan_config=cfg.ctgan,
        discrete_columns=['type'],
    )

    # Step 3: Sample synthetic records
    synthetic_fraud_df = generate_synthetic_fraud(
        model=model,
        n_samples=cfg.ctgan.n_synthetic_samples,
        reference_fraud_df=fraud_df,
    )

    # Step 4: Construct augmented training set
    augmented_train_df = create_augmented_training_data(
        real_train_df=train_df,
        synthetic_fraud_df=synthetic_fraud_df,
        random_state=cfg.ctgan.random_state,
    )

    # Step 5: Save artifacts
    if save_artifacts:
        save_synthetic_artifacts(
            synthetic_df=synthetic_fraud_df,
            augmented_df=augmented_train_df,
            model=model,
            path_config=cfg.paths,
        )

    print("=" * 70)
    print("[+] CTGAN Generation Pipeline completed successfully!")
    print("=" * 70)

    return synthetic_fraud_df, augmented_train_df, model


if __name__ == "__main__":
    from src.data_loader import load_dataset, split_dataset
    print("[*] Testing src/synthetic_data.py module...")

    # Load subset to verify pipeline mechanics quickly
    test_df = load_dataset(nrows=100000)
    train_part, _, _ = split_dataset(test_df, method="stratified")

    # Fast test config
    fast_ctgan_cfg = CTGANConfig(
        n_synthetic_samples=50,
        epochs=2,
        batch_size=50,
        pac=5,
        verbose=False,
    )
    fast_project_cfg = ProjectConfig(ctgan=fast_ctgan_cfg)

    synth_df, aug_df, ctgan_model = run_synthetic_generation_pipeline(
        train_df=train_part,
        config=fast_project_cfg,
        save_artifacts=True,
    )

    print(f"[+] Smoke test passed! Generated {len(synth_df)} synthetic records.")
    print(f"[+] Sample synthetic fraud rows:\n{synth_df.head(3)}")
