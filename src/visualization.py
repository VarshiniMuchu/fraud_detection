"""
Visualization Module for Financial Fraud Detection Research.

Part of Research Project:
Hybrid Synthetic-Data and Anomaly-Aware Ensemble Learning for Financial Fraud Detection.

Generates and saves publication-quality plots:
1. Class Imbalance Distribution
2. Real vs Synthetic Feature Distributions (KDE / Histograms)
3. Correlation Matrix Heatmaps (Real Fraud vs Synthetic Fraud)
4. Autoencoder Reconstruction Error & Anomaly Score Distribution (Fraud vs Legit)
5. Precision-Recall Curves (PR Curves) for all models
6. Receiver Operating Characteristic Curves (ROC Curves)
7. Confusion Matrix Heatmaps
8. Experimental Performance Comparison Chart (Experiments 1 to 5)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import DEFAULT_CONFIG, PathConfig


def plot_class_distribution(y: pd.Series, output_path: Path):
    """Plot and save transaction class imbalance distribution."""
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = y.value_counts().sort_index()
    labels = ['Legitimate (0)', 'Fraudulent (1)']
    colors = ['#2b5c8f', '#d9534f']

    bars = ax.bar(labels, counts.values, color=colors, edgecolor='black', alpha=0.85, width=0.5)
    ax.set_yscale('log')
    ax.set_title("Transaction Class Distribution (Log Scale)", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel("Transaction Count (Log)", fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(y)) * 100
        ax.annotate(f"{h:,}\n({pct:.3f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved Class Distribution Plot -> {output_path}")


def plot_real_vs_synthetic_distributions(
    real_fraud_df: pd.DataFrame,
    synth_fraud_df: pd.DataFrame,
    output_path: Path,
    features: Optional[List[str]] = None,
):
    """Plot overlaid KDE/histograms comparing genuine vs CTGAN synthetic fraud."""
    if features is None:
        features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    valid_feats = [f for f in features if f in real_fraud_df.columns and f in synth_fraud_df.columns]
    n_feats = len(valid_feats)
    if n_feats == 0:
        return

    n_cols = 3
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(valid_feats):
        ax = axes[i]
        r_vals = np.log1p(real_fraud_df[col].dropna().astype(float).clip(lower=0))
        s_vals = np.log1p(synth_fraud_df[col].dropna().astype(float).clip(lower=0))

        ax.hist(r_vals, bins=30, alpha=0.5, density=True, color='#2b5c8f', label='Real Fraud', edgecolor='black')
        ax.hist(s_vals, bins=30, alpha=0.5, density=True, color='#e6550d', label='CTGAN Synthetic', edgecolor='black')

        ax.set_title(f"{col} (log1p)", fontsize=11, fontweight='bold')
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=9)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Real Fraud vs CTGAN Synthetic Fraud Feature Distributions", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved Real vs Synthetic Distribution Plot -> {output_path}")


def plot_correlation_heatmaps(
    real_fraud_df: pd.DataFrame,
    synth_fraud_df: pd.DataFrame,
    output_path: Path,
    features: Optional[List[str]] = None,
):
    """Plot side-by-side correlation matrices for real and synthetic fraud."""
    if features is None:
        features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

    valid_feats = [f for f in features if f in real_fraud_df.columns and f in synth_fraud_df.columns]
    if len(valid_feats) < 2:
        return

    r_corr = real_fraud_df[valid_feats].astype(float).corr().fillna(0.0)
    s_corr = synth_fraud_df[valid_feats].astype(float).corr().fillna(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(r_corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=axes[0], cbar=False)
    axes[0].set_title("Real Fraud Correlations", fontsize=12, fontweight='bold')

    sns.heatmap(s_corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, ax=axes[1], cbar=True)
    axes[1].set_title("CTGAN Synthetic Fraud Correlations", fontsize=12, fontweight='bold')

    plt.suptitle("Correlation Matrix Preservation: Real vs Synthetic Fraud", fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved Correlation Heatmap Comparison -> {output_path}")


def plot_reconstruction_error_distribution(
    y_test: np.ndarray,
    anomaly_scores: np.ndarray,
    output_path: Path,
):
    """Plot Autoencoder anomaly score distribution comparing fraud vs legitimate transactions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    legit_scores = anomaly_scores[y_test == 0]
    fraud_scores = anomaly_scores[y_test == 1]

    ax.hist(legit_scores, bins=50, alpha=0.6, density=True, color='#2b5c8f', label=f'Legitimate (n={len(legit_scores):,})', edgecolor='black')
    ax.hist(fraud_scores, bins=50, alpha=0.6, density=True, color='#d9534f', label=f'Fraud (n={len(fraud_scores):,})', edgecolor='black')

    ax.set_title("Autoencoder Anomaly Score Distribution", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Normalized Anomaly Score (Reconstruction Error)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved Anomaly Score Distribution Plot -> {output_path}")


def plot_evaluation_curves(
    y_test: np.ndarray,
    model_probabilities: Dict[str, np.ndarray],
    pr_output_path: Path,
    roc_output_path: Path,
):
    """Plot multi-model Precision-Recall and ROC curves."""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # 1. PR Curves
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    for i, (name, probs) in enumerate(model_probabilities.items()):
        prec, rec, _ = precision_recall_curve(y_test, probs)
        from sklearn.metrics import average_precision_score
        score = average_precision_score(y_test, probs)
        ax_pr.plot(rec, prec, label=f"{name} (PR-AUC={score:.4f})", color=colors[i % len(colors)], linewidth=2)

    no_skill = np.sum(y_test == 1) / len(y_test)
    ax_pr.axhline(no_skill, linestyle='--', color='gray', label=f'No Skill Baseline ({no_skill:.4f})')
    ax_pr.set_title("Precision-Recall Curves (Fraud Detection)", fontsize=13, fontweight='bold', pad=12)
    ax_pr.set_xlabel("Recall (Sensitivity)", fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_ylim([0.0, 1.05])
    ax_pr.grid(True, linestyle='--', alpha=0.4)
    ax_pr.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    pr_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pr_output_path, dpi=300)
    plt.close(fig_pr)
    print(f"[+] Saved Precision-Recall Curves -> {pr_output_path}")

    # 2. ROC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    for i, (name, probs) in enumerate(model_probabilities.items()):
        fpr, tpr, _ = roc_curve(y_test, probs)
        from sklearn.metrics import roc_auc_score
        score = roc_auc_score(y_test, probs)
        ax_roc.plot(fpr, tpr, label=f"{name} (ROC-AUC={score:.4f})", color=colors[i % len(colors)], linewidth=2)

    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess (0.50)')
    ax_roc.set_title("ROC Curves (Fraud Detection)", fontsize=13, fontweight='bold', pad=12)
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.grid(True, linestyle='--', alpha=0.4)
    ax_roc.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    roc_output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(roc_output_path, dpi=300)
    plt.close(fig_roc)
    print(f"[+] Saved ROC Curves -> {roc_output_path}")


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
):
    """Plot annotated confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Pred Legit', 'Pred Fraud'],
                yticklabels=['Actual Legit', 'Actual Fraud'])
    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved Confusion Matrix -> {output_path}")


def plot_experiment_comparisons(
    experiments_df: pd.DataFrame,
    output_path: Path,
):
    """Plot multi-metric grouped bar chart comparing all 5 research experiments."""
    metrics_to_plot = ['PR-AUC', 'F1-Score', 'Precision', 'Recall', 'ROC-AUC']
    valid_metrics = [m for m in metrics_to_plot if m in experiments_df.columns]
    
    if len(valid_metrics) == 0:
        return

    exp_names = experiments_df['Experiment'].values
    n_exps = len(exp_names)
    n_metrics = len(valid_metrics)

    fig, ax = plt.subplots(figsize=(13, 6))
    bar_width = 0.15
    indices = np.arange(n_exps)

    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, metric in enumerate(valid_metrics):
        vals = experiments_df[metric].values
        ax.bar(indices + i * bar_width, vals, bar_width, label=metric, color=palette[i % len(palette)], edgecolor='black', alpha=0.85)

    ax.set_title("Experimental Framework Performance Comparison (Evaluated on Identical Real Test Set)", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_xticks(indices + bar_width * (n_metrics - 1) / 2)
    ax.set_xticklabels(exp_names, rotation=15, ha='right', fontsize=10)
    ax.set_ylim([0.0, 1.1])
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02), ncol=len(valid_metrics), fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[+] Saved Experiment Comparison Chart -> {output_path}")


if __name__ == "__main__":
    print("[*] Testing src/visualization.py...")
    plots_dir = DEFAULT_CONFIG.paths.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Test class distribution
    y_dummy = pd.Series([0] * 980 + [1] * 20)
    plot_class_distribution(y_dummy, plots_dir / "test_class_dist.png")

    # Test PR / ROC curves
    y_test_d = np.array([0] * 90 + [1] * 10)
    p_dummy = {'Model_A': np.random.uniform(0, 1, 100), 'Model_B': np.random.uniform(0, 1, 100)}
    plot_evaluation_curves(y_test_d, p_dummy, plots_dir / "test_pr.png", plots_dir / "test_roc.png")

    print("[+] Visualization module test passed!")
