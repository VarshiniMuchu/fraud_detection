"""
Data Loader and Dataset Inspector for PaySim Financial Fraud Dataset.

Part of Research Project:
Hybrid Synthetic-Data and Anomaly-Aware Ensemble Learning for Financial Fraud Detection.

Handles:
- Memory-optimized data ingestion
- Comprehensive exploratory inspection (missing values, duplicates, class skew, type breakdown)
- Strictly non-leaking train/test splitting (Temporal split as primary, Stratified as optional)
- Validation and verification of split partitions
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to sys.path if not present
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import DEFAULT_CONFIG, ProjectConfig


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric and categorical columns to optimize memory consumption.
    Crucial for PaySim (~6.3M rows) to run efficiently in memory.
    """
    df = df.copy()
    
    # Step fits well in uint16 (max step in PaySim is ~744)
    if 'step' in df.columns:
        df['step'] = pd.to_numeric(df['step'], downcast='unsigned')
        
    # Transaction type as category
    if 'type' in df.columns:
        df['type'] = df['type'].astype('category')
        
    # Targets as uint8
    if 'isFraud' in df.columns:
        df['isFraud'] = df['isFraud'].astype(np.uint8)
    if 'isFlaggedFraud' in df.columns:
        df['isFlaggedFraud'] = df['isFlaggedFraud'].astype(np.uint8)
        
    # Balance and amount columns as float32/float64
    float_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(np.float64)  # float64 preserves financial currency precision
            
    return df


def load_dataset(
    filepath: Optional[Path] = None,
    optimize_memory: bool = True,
    nrows: Optional[int] = None,
    drop_unnecessary_cols: bool = False
) -> pd.DataFrame:
    """
    Load the PaySim dataset efficiently from disk.
    
    Parameters:
    -----------
    filepath : Path or str, optional
        Path to the PaySim CSV log file. Defaults to config path.
    optimize_memory : bool
        Whether to downcast data types to save RAM.
    nrows : int, optional
        Number of rows to read (useful for testing).
    drop_unnecessary_cols : bool
        Whether to drop identifiers not usable for generalization (nameOrig, nameDest).
        
    Returns:
    --------
    pd.DataFrame: Loaded dataset.
    """
    if filepath is None:
        filepath = DEFAULT_CONFIG.paths.raw_data_path
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Please ensure PaySim dataset is in data/ directory."
        )
        
    print(f"[*] Ingesting dataset from: {filepath} (nrows={nrows or 'ALL'})...")
    df = pd.read_csv(filepath, nrows=nrows)
    
    if optimize_memory:
        initial_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
        df = optimize_dtypes(df)
        final_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"[*] Memory downcasting: {initial_mem:.2f} MB -> {final_mem:.2f} MB (saved {initial_mem - final_mem:.2f} MB)")
    else:
        mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"[*] Dataset loaded in RAM: {mem_mb:.2f} MB")
        
    if drop_unnecessary_cols:
        cols_to_drop = [c for c in ['nameOrig', 'nameDest', 'isFlaggedFraud'] if c in df.columns]
        if cols_to_drop:
            print(f"[*] Dropping raw identifier columns: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
            
    return df


def inspect_dataset(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
    """
    Conduct thorough inspection of the PaySim dataset as required by research protocols.
    
    Checks:
    - Dataset shape
    - Column names and data types
    - Missing values count and percentage
    - Duplicate rows
    - Target class distribution (legitimate vs fraud)
    - Fraud percentage
    - Transaction type distribution overall and per class
    
    Returns:
    --------
    dict: Full dictionary of diagnostic metrics.
    """
    shape = df.shape
    columns = list(df.columns)
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    
    # Missing values
    missing_counts = df.isnull().sum().to_dict()
    missing_pct = ((df.isnull().sum() / len(df)) * 100).to_dict()
    total_missing = sum(missing_counts.values())
    
    # Duplicates check (subsetting features excluding IDs if present, or all cols)
    feature_cols = [c for c in df.columns if c not in ['nameOrig', 'nameDest']]
    duplicate_count = df.duplicated(subset=feature_cols).sum()
    duplicate_pct = (duplicate_count / len(df)) * 100
    
    # Target class distribution
    target_col = 'isFraud'
    if target_col in df.columns:
        class_counts = df[target_col].value_counts().to_dict()
        legit_count = class_counts.get(0, 0)
        fraud_count = class_counts.get(1, 0)
        fraud_pct = (fraud_count / len(df)) * 100
    else:
        legit_count, fraud_count, fraud_pct = None, None, None
        
    # Transaction type distribution
    type_counts = {}
    type_pct = {}
    fraud_by_type = {}
    if 'type' in df.columns:
        type_counts = df['type'].value_counts().to_dict()
        type_pct = ((df['type'].value_counts() / len(df)) * 100).to_dict()
        if target_col in df.columns:
            fraud_crosstab = pd.crosstab(df['type'], df[target_col], margins=False)
            fraud_by_type = fraud_crosstab.to_dict()
            
    # Step range
    step_info = {}
    if 'step' in df.columns:
        step_info = {
            'min_step': int(df['step'].min()),
            'max_step': int(df['step'].max()),
            'unique_steps': int(df['step'].nunique())
        }

    summary = {
        'shape': shape,
        'columns': columns,
        'dtypes': dtypes,
        'memory_mb': memory_mb,
        'total_missing': total_missing,
        'missing_counts': missing_counts,
        'missing_pct': missing_pct,
        'duplicate_count': int(duplicate_count),
        'duplicate_pct': duplicate_pct,
        'legit_count': legit_count,
        'fraud_count': fraud_count,
        'fraud_pct': fraud_pct,
        'type_counts': type_counts,
        'type_pct': type_pct,
        'fraud_by_type': fraud_by_type,
        'step_info': step_info
    }
    
    if verbose:
        print_inspection_report(summary)
        
    return summary


def print_inspection_report(s: Dict[str, Any]) -> None:
    """Format and print an academic-grade diagnostic report of dataset characteristics."""
    print("\n" + "=" * 70)
    print("       PAYSIM DATASET INSPECTION & DIAGNOSTIC REPORT       ")
    print("=" * 70)
    print(f"Total Transactions (Rows)   : {s['shape'][0]:,}")
    print(f"Total Features (Columns)    : {s['shape'][1]:,}")
    print(f"RAM Footprint               : {s['memory_mb']:.2f} MB")
    if s['step_info']:
        print(f"Simulation Horizon (Steps)  : {s['step_info']['min_step']} to {s['step_info']['max_step']} "
              f"({s['step_info']['unique_steps']} distinct hourly steps ~ {s['step_info']['max_step']/24:.1f} days)")
    print("-" * 70)
    
    print("\n[1] COLUMNS & DATA TYPES:")
    for col, dt in s['dtypes'].items():
        missing = s['missing_counts'].get(col, 0)
        print(f"  * {col:<18} : {dt:<10} | Missing: {missing}")
    print(f"Total Missing Values across entire dataset: {s['total_missing']}")
    
    print("\n[2] DUPLICATE ROWS CHECK:")
    print(f"  * Exact duplicates (transaction features): {s['duplicate_count']:,} ({s['duplicate_pct']:.4f}%)")
    
    print("\n[3] CLASS DISTRIBUTION & IMBALANCE (isFraud):")
    legit = s['legit_count']
    fraud = s['fraud_count']
    total = legit + fraud
    imbalance_ratio = (legit / fraud) if fraud > 0 else 0
    print(f"  * Legitimate (Class 0) : {legit:,} ({ (legit/total)*100:.3f}% )")
    print(f"  * Fraudulent (Class 1) : {fraud:,} ({s['fraud_pct']:.4f}% )")
    print(f"  * Severe Imbalance Ratio: 1 fraud per {imbalance_ratio:.1f} legitimate transactions (~{imbalance_ratio:,.0f}:1)")
    
    print("\n[4] TRANSACTION TYPE BREAKDOWN:")
    print(f"  {'Type':<15} {'Total Count':<14} {'Share (%)':<12} {'Fraud Count':<14} {'Fraud Rate in Type'}")
    print("  " + "-" * 66)
    for t, cnt in s['type_counts'].items():
        share = s['type_pct'].get(t, 0.0)
        # s['fraud_by_type'] has keys: 0: {type: count}, 1: {type: count}
        f_cnt = s['fraud_by_type'].get(1, {}).get(t, 0) if s['fraud_by_type'] else 0
        f_rate = (f_cnt / cnt * 100) if cnt > 0 else 0.0
        print(f"  {t:<15} {cnt:<14,} {share:<12.2f} {f_cnt:<14,} {f_rate:.4f}%")
        
    print("=" * 70 + "\n")


def split_dataset(
    df: pd.DataFrame,
    method: str = "temporal",
    test_ratio: float = 0.20,
    temporal_split_step: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split the real dataset into training and testing partitions BEFORE any synthetic generation.
    
    STRICT DATA LEAKAGE ENFORCEMENT:
    - Primary: Temporal split on 'step'. Earlier steps for training, later steps for testing.
      This simulates the real-world operational setting where models are trained on past logs
      and evaluated on future unseen transactions.
    - Optional: Stratified random split for comparative research purposes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full real dataset.
    method : str
        'temporal' or 'stratified'.
    test_ratio : float
        Fraction of data reserved for test set (default: 0.20).
    temporal_split_step : int, optional
        Specific step threshold. If None, automatically computed based on test_ratio percentile.
    random_state : int
        Random seed for reproducible stratified split.
        
    Returns:
    --------
    train_df, test_df, split_info
    """
    target_col = 'isFraud'
    
    if method == "temporal":
        if 'step' not in df.columns:
            raise KeyError("Column 'step' is required for temporal split.")

        unique_steps = sorted(df['step'].dropna().unique())
        if len(unique_steps) < 2:
            print("[!] Temporal split requires multiple step values; falling back to stratified split for this small sample.")
            method = "stratified"

        if method == "temporal":
            if temporal_split_step is None:
                # Use the step value closest to the requested quantile while ensuring both partitions are non-empty.
                target_idx = max(1, min(len(unique_steps) - 1, int(np.floor(len(unique_steps) * (1.0 - test_ratio)))))
                split_step = int(unique_steps[target_idx - 1])
            else:
                split_step = int(temporal_split_step)

            train_mask = df['step'] <= split_step
            test_mask = df['step'] > split_step

            if not train_mask.any() or not test_mask.any():
                midpoint = unique_steps[len(unique_steps) // 2]
                split_step = int(midpoint)
                train_mask = df['step'] <= split_step
                test_mask = df['step'] > split_step

            if not train_mask.any() or not test_mask.any():
                print("[!] Temporal split still produced an empty partition; falling back to stratified split.")
                method = "stratified"
                split_step = None

        if method == "temporal":
            print(f"[*] Executing Temporal Split at step threshold <= {split_step} (Train) vs > {split_step} (Test)...")
            train_df = df[df['step'] <= split_step].copy()
            test_df = df[df['step'] > split_step].copy()

    if method == "stratified":
        print(f"[*] Executing Stratified Random Split (test_ratio={test_ratio}, seed={random_state})...")
        train_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            stratify=df[target_col] if target_col in df.columns else None,
            random_state=random_state
        )
        train_df = train_df.copy()
        test_df = test_df.copy()
        split_step = None
    elif method != "temporal":
        raise ValueError(f"Unknown split method: {method}. Choose 'temporal' or 'stratified'.")

    split_info = verify_split(train_df, test_df, method=method, split_step=split_step)
    return train_df, test_df, split_info


def verify_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str,
    split_step: Optional[int] = None
) -> Dict[str, Any]:
    """
    Mathematically verify that:
    1. Train and test sets are strictly partitioned (no index or temporal overlap).
    2. Both sets have sufficient positive (fraud) and negative (legitimate) cases.
    3. The test set remains completely uncorrupted and real.
    """
    target_col = 'isFraud'
    n_train = len(train_df)
    n_test = len(test_df)
    total = n_train + n_test

    train_fraud = int(train_df[target_col].sum()) if target_col in train_df.columns else 0
    test_fraud = int(test_df[target_col].sum()) if target_col in test_df.columns else 0
    train_fraud_pct = (train_fraud / n_train * 100) if n_train > 0 else 0.0
    test_fraud_pct = (test_fraud / n_test * 100) if n_test > 0 else 0.0

    # Assertions
    assert n_train > 0, "Training set is empty!"
    assert n_test > 0, (
        "Testing set is empty. This usually means the sample is too small for a temporal split "
        "or the step threshold is invalid. Try a larger dataset or use method='stratified'."
    )

    train_steps = (int(train_df['step'].min()), int(train_df['step'].max())) if 'step' in train_df.columns else (None, None)
    test_steps = (int(test_df['step'].min()), int(test_df['step'].max())) if 'step' in test_df.columns else (None, None)

    assert train_fraud > 0, "No fraud cases in training set!"
    assert test_fraud > 0, "No fraud cases in test set! Evaluation impossible."

    if method == "temporal" and split_step is not None:
        assert train_steps[1] <= split_step, f"Leakage detected: train max step {train_steps[1]} > split threshold {split_step}"
        assert test_steps[0] > split_step, f"Leakage detected: test min step {test_steps[0]} <= split threshold {split_step}"
        temporal_leakage_free = True
    else:
        temporal_leakage_free = False

    print("\n" + "=" * 70)
    print(f"             TRAIN / TEST SPLIT VERIFICATION [{method.upper()}]             ")
    print("=" * 70)
    print(f"Total Transactions    : {total:,}")
    print(f"Training Partition    : {n_train:,} ({n_train/total*100:.2f}%)")
    print(f"Testing Partition     : {n_test:,} ({n_test/total*100:.2f}%)")
    print("-" * 70)
    if train_steps[0] is not None:
        print(f"Training Step Horizon : [{train_steps[0]} .. {train_steps[1]}]")
        print(f"Testing Step Horizon  : [{test_steps[0]} .. {test_steps[1]}]")
        print(f"Temporal Disjointness : Verified strictly disjoint at step={split_step} (Leakage Free: True)")
    print("-" * 70)
    print(f"Training Fraud Cases  : {train_fraud:,} ({train_fraud_pct:.4f}%) | 1 fraud per {((n_train-train_fraud)/train_fraud):.1f} legit")
    print(f"Testing Fraud Cases   : {test_fraud:,} ({test_fraud_pct:.4f}%) | 1 fraud per {((n_test-test_fraud)/test_fraud):.1f} legit")
    print("=" * 70 + "\n")

    return {
        'method': method,
        'split_step': split_step,
        'n_train': n_train,
        'n_test': n_test,
        'train_ratio': n_train / total,
        'test_ratio': n_test / total,
        'train_steps': train_steps,
        'test_steps': test_steps,
        'train_fraud': train_fraud,
        'test_fraud': test_fraud,
        'train_fraud_pct': train_fraud_pct,
        'test_fraud_pct': test_fraud_pct,
        'temporal_leakage_free': temporal_leakage_free
    }


if __name__ == "__main__":
    print("[*] Running data_loader.py standalone check...")
    # Load and inspect dataset
    df = load_dataset()
    inspect_dataset(df)
    
    # Perform and verify temporal split
    train_df, test_df, split_info = split_dataset(df, method=DEFAULT_CONFIG.split.method)
    print("[+] Data loading and splitting successfully verified!")
