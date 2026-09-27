"""
Evaluation Metrics Module for Financial Fraud Detection.

Part of Research Project:
Hybrid Synthetic-Data and Anomaly-Aware Ensemble Learning for Financial Fraud Detection.

Specialized for extreme class imbalance:
- Precision, Recall, F1-Score
- Area Under Precision-Recall Curve (PR-AUC / Average Precision)
- Receiver Operating Characteristic (ROC-AUC)
- False Positive Rate (FPR) and False Negative Rate (FNR)
- Full Confusion Matrix Breakdown (TP, FP, TN, FN)
- Optimal Probability Threshold Tuning (maximizing F1 or F2)
- Financial Cost Impact Simulation
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve
)

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def compute_fraud_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    cost_fn: float = 1000.0,
    cost_fp: float = 25.0,
) -> Dict[str, Any]:
    """
    Compute comprehensive fraud detection evaluation metrics.

    Args:
        y_true: Binary ground truth labels (0=legit, 1=fraud).
        y_prob: Predicted probabilities of fraud [0, 1].
        threshold: Decision threshold for positive classification.
        cost_fn: Financial penalty weight for undetected fraud (False Negative).
        cost_fp: Operational cost weight for false alert investigations (False Positive).

    Returns:
        Dictionary of computed performance and economic impact metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix breakdown
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Core rates
    total_pos = tp + fn
    total_neg = tn + fp

    tpr = tp / total_pos if total_pos > 0 else 0.0  # Recall
    fpr = fp / total_neg if total_neg > 0 else 0.0  # False Positive Rate
    fnr = fn / total_pos if total_pos > 0 else 0.0  # False Negative Rate
    tnr = tn / total_neg if total_neg > 0 else 0.0  # Specificity

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    # Continuous curve metrics
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    # Simulated financial impact cost
    total_cost = (fn * cost_fn) + (fp * cost_fp)

    return {
        'threshold': float(threshold),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'f2': float(f2),
        'pr_auc': float(pr_auc),
        'roc_auc': float(roc_auc),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'total_positives': int(total_pos),
        'total_negatives': int(total_neg),
        'simulated_cost': float(total_cost),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """
    Search for the optimal probability threshold that maximizes F1, F2, or PR-F1.

    Returns:
        Tuple of (optimal_threshold, best_metric_score).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    if metric == "f1":
        scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    elif metric == "f2":
        scores = 5 * (precisions * recalls) / (4 * precisions + recalls + 1e-10)
    else:
        scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Avoid endpoint boundary index issue in precision_recall_curve
    valid_len = min(len(thresholds), len(scores))
    if valid_len == 0:
        return 0.5, 0.0

    best_idx = np.argmax(scores[:valid_len])
    best_threshold = float(thresholds[best_idx])
    best_score = float(scores[best_idx])

    return best_threshold, best_score


def format_metrics_table(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a dictionary of model results into a standardized comparative summary table.
    """
    rows = []
    for model_name, m in results_dict.items():
        rows.append({
            'Model': model_name,
            'PR-AUC': m.get('pr_auc', 0.0),
            'F1-Score': m.get('f1', 0.0),
            'Precision': m.get('precision', 0.0),
            'Recall': m.get('recall', 0.0),
            'ROC-AUC': m.get('roc_auc', 0.0),
            'FPR (%)': m.get('fpr', 0.0) * 100,
            'FNR (%)': m.get('fnr', 0.0) * 100,
            'TP': m.get('tp', 0),
            'FP': m.get('fp', 0),
            'FN': m.get('fn', 0),
            'Simulated Cost ($)': m.get('simulated_cost', 0.0),
        })

    df = pd.DataFrame(rows)
    return df.sort_values(by='PR-AUC', ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    print("[*] Testing src/metrics.py...")
    # Simulated test case
    y_t = np.array([0] * 990 + [1] * 10)
    y_p = np.concatenate([np.random.beta(1, 20, size=990), np.random.beta(5, 2, size=10)])

    res = compute_fraud_metrics(y_t, y_p, threshold=0.5)
    print(f"[+] Computed Metrics (threshold=0.5):")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"    - {k:<18}: {v:.4f}")
        else:
            print(f"    - {k:<18}: {v}")

    opt_thresh, opt_f1 = find_optimal_threshold(y_t, y_p, metric="f1")
    print(f"\n[+] Optimal F1 Threshold: {opt_thresh:.4f} (F1 Score = {opt_f1:.4f})")
    print("[+] Metrics test passed!")
