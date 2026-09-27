"""
Explainability Module using SHAP for PaySim Financial Fraud Detection.

Part of Research Project:
Hybrid Synthetic-Data and Anomaly-Aware Ensemble Learning for Financial Fraud Detection.

Capabilities:
1. Global Feature Importance (TreeExplainer across top model / ensemble).
2. Local Transaction Explanations (Waterfall / Force explanation of individual alerts).
3. Targeted Research Component Contribution Breakdown:
   - Autoencoder `anomaly_score` contribution
   - Transaction `amount` contribution
   - Behavioral engineered features (`orig_balance_error`, ratios)
   - Transaction `type` channel contributions
4. Automated Plot Generation saved directly to `output/plots/`.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt
import shap

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import DEFAULT_CONFIG, ProjectConfig, PathConfig


def compute_shap_explanations(
    model: Any,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    max_eval_samples: int = 500,
) -> Tuple[shap.Explanation, np.ndarray]:
    """
    Compute SHAP values using TreeExplainer on a sample of test transactions.

    Args:
        model: Trained tree-based model (e.g. XGBoost, LightGBM, or Random Forest).
        X_background: Background dataset for baseline expectations.
        X_explain: Data to explain.
        max_eval_samples: Maximum samples to evaluate for performance.

    Returns:
        Tuple of (shap_values_explanation_object, shap_values_array).
    """
    n_samples = min(len(X_explain), max_eval_samples)
    X_sub = X_explain.iloc[:n_samples].copy()

    print(f"[*] Computing SHAP values for {n_samples} transactions using TreeExplainer...")
    
    # Check if model has underlying booster or sklearn wrapper
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sub)
    except Exception as e:
        print(f"    [!] TreeExplainer fallback: {e}")
        bg = X_background.sample(n=min(100, len(X_background)), random_state=42)
        explainer = shap.Explainer(model.predict_proba, bg)
        shap_values = explainer(X_sub)

    # If binary classification returns 3D array (samples, features, classes), pick class 1 (fraud)
    if hasattr(shap_values, 'values') and len(shap_values.values.shape) == 3:
        raw_vals = shap_values.values[:, :, 1]
    elif hasattr(shap_values, 'values'):
        raw_vals = shap_values.values
    else:
        raw_vals = np.array(shap_values)

    return shap_values, raw_vals


def generate_shap_plots(
    shap_explanation: shap.Explanation,
    raw_shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    y_test: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Generate and save all required SHAP explainability visual figures.
    """
    out_dir = output_dir or DEFAULT_CONFIG.paths.plots_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_plots = {}

    feature_names = list(X_explain.columns)
    n_samples = len(raw_shap_values)
    X_sub = X_explain.iloc[:n_samples]

    # 1. Global Feature Importance Bar Plot
    mean_abs_shap = np.mean(np.abs(raw_shap_values), axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Abs_SHAP', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(importance_df['Feature'], importance_df['Mean_Abs_SHAP'], color='#2b5c8f', edgecolor='black', alpha=0.85)
    ax.set_title("Global Feature Importance (Mean |SHAP Value|)", fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Fraud Probability)", fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    bar_path = out_dir / "shap_feature_importance.png"
    plt.savefig(bar_path, dpi=300)
    plt.close(fig)
    saved_plots['feature_importance'] = bar_path
    print(f"[+] Saved SHAP Feature Importance -> {bar_path}")

    # 2. Global Summary Beeswarm Plot
    fig, ax = plt.subplots(figsize=(11, 8))
    try:
        if len(shap_explanation.shape) == 3:
            shap.summary_plot(shap_explanation.values[:, :, 1], X_sub, show=False)
        else:
            shap.summary_plot(raw_shap_values, X_sub, show=False)
        plt.title("SHAP Beeswarm Summary Distribution", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        beeswarm_path = out_dir / "shap_summary_plot.png"
        plt.savefig(beeswarm_path, dpi=300)
        saved_plots['summary_plot'] = beeswarm_path
        print(f"[+] Saved SHAP Summary Plot -> {beeswarm_path}")
    except Exception as e:
        print(f"    [!] Could not render beeswarm: {e}")
    finally:
        plt.close('all')

    # 3. Local High-Risk Fraud Transaction Explanation (Waterfall/Bar)
    # Find sample with highest predicted fraud impact or genuine fraud instance
    fraud_indices = np.where(y_test[:n_samples] == 1)[0] if y_test is not None else []
    target_idx = fraud_indices[0] if len(fraud_indices) > 0 else np.argmax(np.sum(raw_shap_values, axis=1))

    sample_shap = raw_shap_values[target_idx]
    sample_feat_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Value': sample_shap,
        'Feature_Value': X_sub.iloc[target_idx].values
    }).sort_values(by='SHAP_Value', key=abs, ascending=True)

    top_local = sample_feat_df.tail(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#d9534f' if v > 0 else '#5cb85c' for v in top_local['SHAP_Value']]
    bars = ax.barh(top_local['Feature'], top_local['SHAP_Value'], color=colors, edgecolor='black', alpha=0.85)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title(f"Local Transaction Explanation (Transaction Index: {target_idx})", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("SHAP Value (Positive = Pushes Toward Fraud)", fontsize=11)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    local_path = out_dir / "shap_local_fraud_explanation.png"
    plt.savefig(local_path, dpi=300)
    plt.close(fig)
    saved_plots['local_explanation'] = local_path
    print(f"[+] Saved Local SHAP Explanation -> {local_path}")

    # 4. Research Component Contribution Breakdown
    # Categorize features into the 4 research architectural pillars
    group_map = {
        'Autoencoder Anomaly Score': ['anomaly_score'],
        'Transaction Amount': ['amount'],
        'Behavioral / Accounting': [
            'orig_balance_error', 'dest_balance_error',
            'amount_to_oldbalanceOrg_ratio', 'amount_to_oldbalanceDest_ratio',
            'is_zero_orig_balance', 'is_zero_new_orig', 'is_zero_dest_balance',
            'is_orig_emptied', 'is_full_transfer', 'hour_of_day', 'day_of_week'
        ],
        'Transaction Type / Channel': ['type_TRANSFER', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_CASH_IN', 'is_high_risk_type']
    }

    group_contributions = {}
    for group_name, feats in group_map.items():
        matched_indices = [i for i, f in enumerate(feature_names) if f in feats]
        if matched_indices:
            group_contributions[group_name] = float(np.sum(mean_abs_shap[matched_indices]))
        else:
            group_contributions[group_name] = 0.0

    group_df = pd.DataFrame(list(group_contributions.items()), columns=['Component', 'Mean_Abs_SHAP'])
    group_df = group_df.sort_values(by='Mean_Abs_SHAP', ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(group_df['Component'], group_df['Mean_Abs_SHAP'], color=['#e6550d', '#3182bd', '#31a354', '#756bb1'], edgecolor='black', alpha=0.85)
    ax.set_title("Architectural Component Contribution Breakdown", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Aggregate Mean |SHAP Value|", fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.tight_layout()
    comp_path = out_dir / "shap_component_breakdown.png"
    plt.savefig(comp_path, dpi=300)
    plt.close(fig)
    saved_plots['component_breakdown'] = comp_path
    print(f"[+] Saved Architectural Component Breakdown -> {comp_path}")

    return saved_plots


def run_explainability_pipeline(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: Optional[np.ndarray] = None,
    config: Optional[ProjectConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Path]]:
    """
    Run full SHAP explainability analysis and export plots.
    """
    cfg = config or DEFAULT_CONFIG
    print("=" * 70)
    print("STAGE 10: SHAP MODEL EXPLAINABILITY & FEATURE IMPORTANCE")
    print("=" * 70)

    shap_exp, raw_vals = compute_shap_explanations(
        model=model,
        X_background=X_train,
        X_explain=X_test,
        max_eval_samples=400,
    )

    saved_plots = generate_shap_plots(
        shap_explanation=shap_exp,
        raw_shap_values=raw_vals,
        X_explain=X_test,
        y_test=y_test,
        output_dir=cfg.paths.plots_dir,
    )

    feature_names = list(X_test.columns)
    mean_abs_shap = np.mean(np.abs(raw_vals), axis=0)
    ranking_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

    print("\n[*] Top 10 Influential Features by Mean |SHAP| Value:")
    for i, row in ranking_df.head(10).iterrows():
        print(f"    {i+1:2d}. {row['Feature']:<28}: {row['Mean_Abs_SHAP']:.4f}")

    print("=" * 70)
    return ranking_df, saved_plots


if __name__ == "__main__":
    from src.data_loader import load_dataset, split_dataset
    from src.feature_engineering import PaySimFeatureEngineer
    from src.preprocessing import PaySimPreprocessor
    from sklearn.ensemble import RandomForestClassifier

    print("[*] Testing src/explainability.py...")
    raw = load_dataset(nrows=15000)
    train_part, test_part, _ = split_dataset(raw, method="stratified")

    fe = PaySimFeatureEngineer()
    train_fe = fe.fit_transform(train_part)
    test_fe = fe.transform(test_part)

    pre = PaySimPreprocessor()
    train_prep = pre.fit_transform(train_fe)
    test_prep = pre.transform(test_fe)

    feature_cols = [c for c in train_prep.columns if c not in ['isFraud']]
    X_tr = train_prep[feature_cols]
    y_tr = train_prep['isFraud'].values
    X_te = test_prep[feature_cols]
    y_te = test_prep['isFraud'].values

    rf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=42)
    rf.fit(X_tr, y_tr)

    ranking, plots = run_explainability_pipeline(
        model=rf,
        X_train=X_tr,
        X_test=X_te,
        y_test=y_te,
    )

    print(f"[+] Explainability test passed! Generated {len(plots)} visualization plots.")
