"""
Synthetic Data Quality Assessment Module for CTGAN Fraud Data.

Part of Research Project:
Hybrid Synthetic-Data and Anomaly-Aware Ensemble Learning for Financial Fraud Detection.

Evaluates:
A. Statistical Similarity (mean, std, quantiles, KS statistic, Wasserstein distance)
B. Correlation Preservation (matrix Frobenius norm and Mean Absolute Error)
C. Domain Validity (non-negativity, valid steps, valid transaction types)
D. Privacy & Memorization (Nearest Neighbor distance, duplicate detection)
E. TSTR (Train on Synthetic, Test on Real) and TRTS Utility Evaluations
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import DEFAULT_CONFIG, ProjectConfig, PathConfig


def compute_statistical_similarity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    continuous_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute statistical divergence metrics between genuine and synthetic fraud samples.
    Metrics: Mean, Std, Median, IQR, Kolmogorov-Smirnov statistic, Wasserstein Distance.
    """
    if continuous_cols is None:
        continuous_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    valid_cols = [c for c in continuous_cols if c in real_df.columns and c in synth_df.columns]
    results = []

    for col in valid_cols:
        real_vals = real_df[col].dropna().astype(float).values
        synth_vals = synth_df[col].dropna().astype(float).values

        if len(real_vals) == 0 or len(synth_vals) == 0:
            continue

        ks_stat, ks_pval = ks_2samp(real_vals, synth_vals)
        w_dist = wasserstein_distance(real_vals, synth_vals)

        results.append({
            'feature': col,
            'real_mean': float(np.mean(real_vals)),
            'synth_mean': float(np.mean(synth_vals)),
            'mean_diff_pct': float(abs(np.mean(synth_vals) - np.mean(real_vals)) / (abs(np.mean(real_vals)) + 1e-9) * 100),
            'real_std': float(np.std(real_vals)),
            'synth_std': float(np.std(synth_vals)),
            'real_median': float(np.median(real_vals)),
            'synth_median': float(np.median(synth_vals)),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pval),
            'wasserstein_distance': float(w_dist),
        })

    return pd.DataFrame(results)


def compute_correlation_similarity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    continuous_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Quantify correlation structure preservation between real and synthetic data.
    Returns: Real corr matrix, Synth corr matrix, Frobenius norm diff, Mean Absolute Error (MAE).
    """
    if continuous_cols is None:
        continuous_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    valid_cols = [c for c in continuous_cols if c in real_df.columns and c in synth_df.columns]
    
    real_corr = real_df[valid_cols].astype(float).corr().fillna(0.0)
    synth_corr = synth_df[valid_cols].astype(float).corr().fillna(0.0)

    corr_diff = (real_corr - synth_corr).values
    frobenius_norm = float(np.linalg.norm(corr_diff, 'fro'))
    mae = float(np.mean(np.abs(corr_diff)))

    return {
        'real_corr': real_corr,
        'synth_corr': synth_corr,
        'frobenius_norm': frobenius_norm,
        'correlation_mae': mae,
    }


def check_domain_validity(synth_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Verify financial domain rules on synthetic records.
    - No negative balances/amounts.
    - Transaction step >= 1.
    - Valid transaction types.
    """
    checks = {}
    monetary_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    total_records = len(synth_df)
    negative_counts = {}
    for col in monetary_cols:
        if col in synth_df.columns:
            neg_count = int((synth_df[col] < 0).sum())
            negative_counts[col] = neg_count

    checks['negative_values_found'] = sum(negative_counts.values())
    checks['negative_counts_per_col'] = negative_counts

    if 'step' in synth_df.columns:
        invalid_steps = int((synth_df['step'] < 1).sum())
        checks['invalid_steps'] = invalid_steps

    if 'type' in synth_df.columns:
        checks['type_distribution'] = synth_df['type'].value_counts().to_dict()

    checks['domain_validity_passed'] = (checks['negative_values_found'] == 0)
    return checks


def check_duplicate_memorization(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    continuous_cols: Optional[List[str]] = None,
    sample_size: int = 1000,
) -> Dict[str, Any]:
    """
    Analyze whether the CTGAN synthesizer memorized training instances
    by computing minimum Euclidean distances in normalized feature space.
    """
    if continuous_cols is None:
        continuous_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    valid_cols = [c for c in continuous_cols if c in real_df.columns and c in synth_df.columns]

    n_r = min(len(real_df), sample_size)
    n_s = min(len(synth_df), sample_size)

    r_sample = real_df[valid_cols].astype(float).sample(n=n_r, random_state=42).values
    s_sample = synth_df[valid_cols].astype(float).sample(n=n_s, random_state=42).values

    # Normalize with RobustScaler
    scaler = RobustScaler()
    r_norm = scaler.fit_transform(r_sample)
    s_norm = scaler.transform(s_sample)

    # Compute minimum distances
    min_distances = []
    exact_duplicates = 0
    for s_pt in s_norm:
        dists = np.linalg.norm(r_norm - s_pt, axis=1)
        min_d = np.min(dists)
        min_distances.append(min_d)
        if min_d < 1e-4:
            exact_duplicates += 1

    min_distances = np.array(min_distances)
    return {
        'exact_duplicate_count': exact_duplicates,
        'exact_duplicate_pct': (exact_duplicates / len(s_norm)) * 100,
        'min_distance_mean': float(np.mean(min_distances)),
        'min_distance_median': float(np.median(min_distances)),
        'min_distance_min': float(np.min(min_distances)),
        'is_memorizing': exact_duplicates > (0.01 * len(s_norm)),
    }


def evaluate_tstr_utility(
    real_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    synthetic_fraud_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Execute Train on Synthetic, Test on Real (TSTR) and Train on Real, Test on Real (TRTR) benchmarks.
    """
    if feature_cols is None:
        feature_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    valid_feats = [c for c in feature_cols if c in real_train_df.columns and c in synthetic_fraud_df.columns]

    # 1. TRTR: Train on Real, Test on Real
    X_real_train = real_train_df[valid_feats].fillna(0.0).values
    y_real_train = real_train_df['isFraud'].values

    X_real_test = real_test_df[valid_feats].fillna(0.0).values
    y_real_test = real_test_df['isFraud'].values

    clf_trtr = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=random_state, n_jobs=-1)
    clf_trtr.fit(X_real_train, y_real_train)
    probs_trtr = clf_trtr.predict_proba(X_real_test)[:, 1]
    preds_trtr = (probs_trtr >= 0.5).astype(int)

    trtr_metrics = {
        'precision': float(precision_score(y_real_test, preds_trtr, zero_division=0)),
        'recall': float(recall_score(y_real_test, preds_trtr, zero_division=0)),
        'f1': float(f1_score(y_real_test, preds_trtr, zero_division=0)),
        'pr_auc': float(average_precision_score(y_real_test, probs_trtr)),
        'roc_auc': float(roc_auc_score(y_real_test, probs_trtr)),
    }

    # 2. TSTR: Train on Synthetic Frauds + Real Legitimate, Test on Real
    real_legit = real_train_df[real_train_df['isFraud'] == 0]
    synth_train_df = pd.concat([real_legit, synthetic_fraud_df], axis=0).sample(frac=1.0, random_state=random_state)
    
    X_synth_train = synth_train_df[valid_feats].fillna(0.0).values
    y_synth_train = synth_train_df['isFraud'].values

    clf_tstr = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=random_state, n_jobs=-1)
    clf_tstr.fit(X_synth_train, y_synth_train)
    probs_tstr = clf_tstr.predict_proba(X_real_test)[:, 1]
    preds_tstr = (probs_tstr >= 0.5).astype(int)

    tstr_metrics = {
        'precision': float(precision_score(y_real_test, preds_tstr, zero_division=0)),
        'recall': float(recall_score(y_real_test, preds_tstr, zero_division=0)),
        'f1': float(f1_score(y_real_test, preds_tstr, zero_division=0)),
        'pr_auc': float(average_precision_score(y_real_test, probs_tstr)),
        'roc_auc': float(roc_auc_score(y_real_test, probs_tstr)),
    }

    return {
        'TRTR': trtr_metrics,
        'TSTR': tstr_metrics,
    }


def run_synthetic_quality_pipeline(
    real_fraud_df: pd.DataFrame,
    synthetic_fraud_df: pd.DataFrame,
    real_train_df: Optional[pd.DataFrame] = None,
    real_test_df: Optional[pd.DataFrame] = None,
    config: Optional[ProjectConfig] = None,
) -> pd.DataFrame:
    """
    Run complete synthetic quality assessment suite and export results.
    """
    cfg = config or DEFAULT_CONFIG
    cfg.paths.create_dirs()

    print("=" * 70)
    print("STAGE 5: SYNTHETIC DATA QUALITY & FIDELITY ASSESSMENT")
    print("=" * 70)

    # A. Statistical Similarity
    stat_df = compute_statistical_similarity(real_fraud_df, synthetic_fraud_df)
    print("\n[*] Statistical Similarity Metrics (Real Fraud vs Synthetic Fraud):")
    for _, row in stat_df.iterrows():
        print(f"    - {row['feature']:<16} | Mean Diff: {row['mean_diff_pct']:6.2f}% | KS: {row['ks_statistic']:.4f} (p={row['ks_pvalue']:.2e}) | Wasserstein: {row['wasserstein_distance']:.2f}")

    # B. Correlation Preservation
    corr_info = compute_correlation_similarity(real_fraud_df, synthetic_fraud_df)
    print(f"\n[*] Correlation Matrix Preservation:")
    print(f"    - Frobenius Distance : {corr_info['frobenius_norm']:.4f}")
    print(f"    - Correlation MAE    : {corr_info['correlation_mae']:.4f}")

    # C. Domain Validity
    validity_info = check_domain_validity(synthetic_fraud_df)
    print(f"\n[*] Domain Validity:")
    print(f"    - Negative values found: {validity_info['negative_values_found']}")
    print(f"    - Domain validity pass : {validity_info['domain_validity_passed']}")

    # D. Privacy & Memorization
    priv_info = check_duplicate_memorization(real_fraud_df, synthetic_fraud_df)
    print(f"\n[*] Duplicate & Memorization Analysis:")
    print(f"    - Exact duplicates    : {priv_info['exact_duplicate_count']} ({priv_info['exact_duplicate_pct']:.2f}%)")
    print(f"    - Mean Min Distance   : {priv_info['min_distance_mean']:.4f}")
    print(f"    - Is Memorizing       : {priv_info['is_memorizing']}")

    # E. TSTR Utility
    report_rows = []
    for _, row in stat_df.iterrows():
        report_rows.append({
            'category': 'statistical_similarity',
            'metric': f"KS_{row['feature']}",
            'value': row['ks_statistic'],
            'details': f"p_val={row['ks_pvalue']:.2e}, W_dist={row['wasserstein_distance']:.2f}",
        })

    report_rows.append({
        'category': 'correlation_preservation',
        'metric': 'frobenius_norm',
        'value': corr_info['frobenius_norm'],
        'details': f"MAE={corr_info['correlation_mae']:.4f}",
    })
    report_rows.append({
        'category': 'privacy',
        'metric': 'exact_duplicates',
        'value': priv_info['exact_duplicate_count'],
        'details': f"mean_min_dist={priv_info['min_distance_mean']:.4f}",
    })

    if real_train_df is not None and real_test_df is not None:
        print("\n[*] Evaluating TSTR vs TRTR Model Utility...")
        tstr_results = evaluate_tstr_utility(real_train_df, real_test_df, synthetic_fraud_df)
        print(f"    - TRTR (Real->Real)  : F1={tstr_results['TRTR']['f1']:.4f}, PR-AUC={tstr_results['TRTR']['pr_auc']:.4f}, ROC-AUC={tstr_results['TRTR']['roc_auc']:.4f}")
        print(f"    - TSTR (Synth->Real) : F1={tstr_results['TSTR']['f1']:.4f}, PR-AUC={tstr_results['TSTR']['pr_auc']:.4f}, ROC-AUC={tstr_results['TSTR']['roc_auc']:.4f}")

        for bench in ['TRTR', 'TSTR']:
            for k, v in tstr_results[bench].items():
                report_rows.append({
                    'category': 'tstr_utility',
                    'metric': f"{bench}_{k}",
                    'value': v,
                    'details': bench,
                })

    report_df = pd.DataFrame(report_rows)
    report_path = cfg.paths.synthetic_quality_path
    report_df.to_csv(report_path, index=False)
    print(f"\n[+] Saved synthetic data quality report -> {report_path}")
    print("=" * 70)

    return report_df


if __name__ == "__main__":
    from src.data_loader import load_dataset, split_dataset
    from src.preprocessing import prepare_for_ctgan

    print("[*] Testing src/synthetic_quality.py...")
    test_df = load_dataset(nrows=100000)
    train_df, test_df_part, _ = split_dataset(test_df, method="stratified")
    
    real_frauds = prepare_for_ctgan(train_df)
    
    # Load previously generated synthetic fraud or sample dummy
    if DEFAULT_CONFIG.paths.synthetic_fraud_path.exists():
        synth_frauds = pd.read_csv(DEFAULT_CONFIG.paths.synthetic_fraud_path)
    else:
        synth_frauds = real_frauds.sample(n=len(real_frauds), replace=True, random_state=42)

    rep = run_synthetic_quality_pipeline(
        real_fraud_df=real_frauds,
        synthetic_fraud_df=synth_frauds,
        real_train_df=train_df,
        real_test_df=test_df_part,
    )
    print("[+] Synthetic quality smoke test passed!")
