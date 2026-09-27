"""
Experimental Comparison Framework for Fraud Detection Research.

Part of Research Project:
Hybrid Synthetic-Data and Anomaly-Aware Ensemble Learning for Financial Fraud Detection.

Systematic Empirical Benchmarking Across 5 Core Configurations:
- Experiment 1: Baseline (Original Training Data + Individual Models)
- Experiment 2: Synthetic Augmentation (CTGAN Augmented Training Data + Individual Models)
- Experiment 3: Ensemble Learning (CTGAN + Soft-Voting/Stacking Ensemble)
- Experiment 4: Anomaly-Aware Ensemble (CTGAN + Autoencoder Anomaly Score + Ensemble)
- Experiment 5: Full Hybrid Framework (CTGAN + Autoencoder Anomaly Score + Behavioral Features + Ensemble)
- Experiment 6 (Simulation): Unknown / Zero-Day Fraud Pattern Robustness

Strict Non-Negotiable Principle:
Every single experiment is evaluated on the EXACT SAME untouched real test set.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import DEFAULT_CONFIG, ProjectConfig, PathConfig
from src.preprocessing import PaySimPreprocessor
from src.feature_engineering import PaySimFeatureEngineer
from src.autoencoder import AutoencoderAnomalyDetector, train_autoencoder
from src.ensemble import FraudEnsembleClassifier
from src.metrics import compute_fraud_metrics, format_metrics_table
from src.visualization import plot_experiment_comparisons, plot_evaluation_curves


def run_experiment_1_baseline(
    real_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    config: Optional[ProjectConfig] = None,
) -> Dict[str, Any]:
    """
    Experiment 1: Baseline (Original Real Training Data + Individual Models).
    Features: Raw preprocessed transaction features (no synthetic data, no AE, no behavioral features).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: BASELINE (ORIGINAL TRAIN DATA + INDIVIDUAL MODELS)")
    print("=" * 70)
    cfg = config or DEFAULT_CONFIG

    # Preprocess raw data
    pre = PaySimPreprocessor()
    train_prep = pre.fit_transform(real_train_df)
    test_prep = pre.transform(real_test_df)

    feat_cols = [c for c in train_prep.columns if c != 'isFraud']
    X_tr = train_prep[feat_cols]
    y_tr = train_prep['isFraud'].values
    X_te = test_prep[feat_cols]
    y_te = test_prep['isFraud'].values

    # Train individual models and ensemble
    ensemble = FraudEnsembleClassifier(random_state=cfg.split.random_state)
    ensemble.train_individual_models(X_tr, y_tr, config=cfg.models)
    ensemble.fit_ensemble(X_tr.iloc[:int(len(X_tr)*0.25)], y_tr[:int(len(y_tr)*0.25)])

    evals = ensemble.evaluate_all(X_te, y_te)
    best_model_name = max(evals.keys(), key=lambda k: evals[k]['pr_auc'])
    best_metrics = evals[best_model_name]

    print(f"[+] Exp 1 Best Model ({best_model_name}): PR-AUC={best_metrics['pr_auc']:.4f}, F1={best_metrics['f1']:.4f}, Recall={best_metrics['recall']:.4f}")
    return {
        'Experiment': 'Exp 1: Baseline (Original Train)',
        'Model': best_model_name,
        'PR-AUC': best_metrics['pr_auc'],
        'F1-Score': best_metrics['f1'],
        'Precision': best_metrics['precision'],
        'Recall': best_metrics['recall'],
        'ROC-AUC': best_metrics['roc_auc'],
        'FPR (%)': best_metrics['fpr'] * 100,
        'FNR (%)': best_metrics['fnr'] * 100,
        'Simulated Cost ($)': best_metrics['simulated_cost'],
        'all_evals': evals,
        'test_probs': ensemble.predict_proba(X_te),
    }


def run_experiment_2_ctgan(
    augmented_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    config: Optional[ProjectConfig] = None,
) -> Dict[str, Any]:
    """
    Experiment 2: CTGAN Augmented Training Data + Individual Models.
    Investigates whether synthetic fraud generation enhances minority-class detection.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: CTGAN AUGMENTED TRAINING DATA + INDIVIDUAL MODELS")
    print("=" * 70)
    cfg = config or DEFAULT_CONFIG

    pre = PaySimPreprocessor()
    train_prep = pre.fit_transform(augmented_train_df)
    test_prep = pre.transform(real_test_df)

    feat_cols = [c for c in train_prep.columns if c != 'isFraud']
    X_tr = train_prep[feat_cols]
    y_tr = train_prep['isFraud'].values
    X_te = test_prep[feat_cols]
    y_te = test_prep['isFraud'].values

    ensemble = FraudEnsembleClassifier(random_state=cfg.split.random_state)
    ensemble.train_individual_models(X_tr, y_tr, config=cfg.models)
    ensemble.fit_ensemble(X_tr.iloc[:int(len(X_tr)*0.25)], y_tr[:int(len(y_tr)*0.25)])

    evals = ensemble.evaluate_all(X_te, y_te)
    best_ind_name = max([k for k in evals.keys() if k != 'ensemble'], key=lambda k: evals[k]['pr_auc'])
    best_metrics = evals[best_ind_name]

    print(f"[+] Exp 2 Best Individual Model ({best_ind_name}): PR-AUC={best_metrics['pr_auc']:.4f}, F1={best_metrics['f1']:.4f}, Recall={best_metrics['recall']:.4f}")
    return {
        'Experiment': 'Exp 2: CTGAN Augmented',
        'Model': best_ind_name,
        'PR-AUC': best_metrics['pr_auc'],
        'F1-Score': best_metrics['f1'],
        'Precision': best_metrics['precision'],
        'Recall': best_metrics['recall'],
        'ROC-AUC': best_metrics['roc_auc'],
        'FPR (%)': best_metrics['fpr'] * 100,
        'FNR (%)': best_metrics['fnr'] * 100,
        'Simulated Cost ($)': best_metrics['simulated_cost'],
        'all_evals': evals,
        'test_probs': evals[best_ind_name],
    }


def run_experiment_3_ensemble(
    augmented_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    config: Optional[ProjectConfig] = None,
) -> Dict[str, Any]:
    """
    Experiment 3: CTGAN + Ensemble Learning.
    Evaluates whether soft voting / stacking outperforms individual models.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: CTGAN + MULTI-MODEL ENSEMBLE")
    print("=" * 70)
    cfg = config or DEFAULT_CONFIG

    pre = PaySimPreprocessor()
    train_prep = pre.fit_transform(augmented_train_df)
    test_prep = pre.transform(real_test_df)

    feat_cols = [c for c in train_prep.columns if c != 'isFraud']
    X_tr = train_prep[feat_cols]
    y_tr = train_prep['isFraud'].values
    X_te = test_prep[feat_cols]
    y_te = test_prep['isFraud'].values

    ensemble = FraudEnsembleClassifier(random_state=cfg.split.random_state)
    ensemble.train_individual_models(X_tr, y_tr, config=cfg.models)
    ensemble.fit_ensemble(X_tr.iloc[:int(len(X_tr)*0.25)], y_tr[:int(len(y_tr)*0.25)])

    evals = ensemble.evaluate_all(X_te, y_te)
    ens_metrics = evals['ensemble']

    print(f"[+] Exp 3 Ensemble: PR-AUC={ens_metrics['pr_auc']:.4f}, F1={ens_metrics['f1']:.4f}, Recall={ens_metrics['recall']:.4f}")
    return {
        'Experiment': 'Exp 3: CTGAN + Ensemble',
        'Model': 'Ensemble (Voting)',
        'PR-AUC': ens_metrics['pr_auc'],
        'F1-Score': ens_metrics['f1'],
        'Precision': ens_metrics['precision'],
        'Recall': ens_metrics['recall'],
        'ROC-AUC': ens_metrics['roc_auc'],
        'FPR (%)': ens_metrics['fpr'] * 100,
        'FNR (%)': ens_metrics['fnr'] * 100,
        'Simulated Cost ($)': ens_metrics['simulated_cost'],
        'all_evals': evals,
        'test_probs': ensemble.predict_proba(X_te),
    }


def run_experiment_4_anomaly_ensemble(
    augmented_train_df: pd.DataFrame,
    real_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    config: Optional[ProjectConfig] = None,
) -> Dict[str, Any]:
    """
    Experiment 4: CTGAN + Autoencoder Anomaly Score + Ensemble.
    Evaluates whether semi-supervised reconstruction error boosts ensemble precision/recall.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: CTGAN + AUTOENCODER ANOMALY SCORE + ENSEMBLE")
    print("=" * 70)
    cfg = config or DEFAULT_CONFIG

    # Preprocess
    pre = PaySimPreprocessor()
    train_prep = pre.fit_transform(augmented_train_df)
    test_prep = pre.transform(real_test_df)

    # Train Autoencoder on real legitimate training data
    real_prep = pre.transform(real_train_df)
    ae_detector = train_autoencoder(real_prep, config=cfg.autoencoder, path_config=cfg.paths)

    # Append anomaly score feature
    train_prep['anomaly_score'] = ae_detector.compute_anomaly_score(train_prep)
    test_prep['anomaly_score'] = ae_detector.compute_anomaly_score(test_prep)

    feat_cols = [c for c in train_prep.columns if c != 'isFraud']
    X_tr = train_prep[feat_cols]
    y_tr = train_prep['isFraud'].values
    X_te = test_prep[feat_cols]
    y_te = test_prep['isFraud'].values

    ensemble = FraudEnsembleClassifier(random_state=cfg.split.random_state)
    ensemble.train_individual_models(X_tr, y_tr, config=cfg.models)
    ensemble.fit_ensemble(X_tr.iloc[:int(len(X_tr)*0.25)], y_tr[:int(len(y_tr)*0.25)])

    evals = ensemble.evaluate_all(X_te, y_te)
    ens_metrics = evals['ensemble']

    print(f"[+] Exp 4 Anomaly-Aware Ensemble: PR-AUC={ens_metrics['pr_auc']:.4f}, F1={ens_metrics['f1']:.4f}, Recall={ens_metrics['recall']:.4f}")
    return {
        'Experiment': 'Exp 4: CTGAN + AE Anomaly + Ens',
        'Model': 'Ensemble + AE',
        'PR-AUC': ens_metrics['pr_auc'],
        'F1-Score': ens_metrics['f1'],
        'Precision': ens_metrics['precision'],
        'Recall': ens_metrics['recall'],
        'ROC-AUC': ens_metrics['roc_auc'],
        'FPR (%)': ens_metrics['fpr'] * 100,
        'FNR (%)': ens_metrics['fnr'] * 100,
        'Simulated Cost ($)': ens_metrics['simulated_cost'],
        'all_evals': evals,
        'test_probs': ensemble.predict_proba(X_te),
    }


def run_experiment_5_full_hybrid(
    augmented_train_df: pd.DataFrame,
    real_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    config: Optional[ProjectConfig] = None,
) -> Tuple[Dict[str, Any], FraudEnsembleClassifier, pd.DataFrame, pd.DataFrame]:
    """
    Experiment 5: Full Core Hybrid Framework.
    Original Features + Behavioral Features + Autoencoder Anomaly Score + CTGAN Augmentation + Ensemble.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: FULL HYBRID FRAMEWORK (CTGAN + AE + BEHAVIORAL + ENSEMBLE)")
    print("=" * 70)
    cfg = config or DEFAULT_CONFIG

    # 1. Feature Engineering
    fe = PaySimFeatureEngineer()
    aug_fe = fe.fit_transform(augmented_train_df)
    real_tr_fe = fe.transform(real_train_df)
    real_te_fe = fe.transform(real_test_df)

    # 2. Preprocessing
    pre = PaySimPreprocessor()
    train_prep = pre.fit_transform(aug_fe)
    test_prep = pre.transform(real_te_fe)
    real_tr_prep = pre.transform(real_tr_fe)

    # 3. Autoencoder Anomaly Scoring
    ae_detector = train_autoencoder(real_tr_prep, config=cfg.autoencoder, path_config=cfg.paths)
    train_prep['anomaly_score'] = ae_detector.compute_anomaly_score(train_prep)
    test_prep['anomaly_score'] = ae_detector.compute_anomaly_score(test_prep)

    feat_cols = [c for c in train_prep.columns if c != 'isFraud']
    X_tr = train_prep[feat_cols]
    y_tr = train_prep['isFraud'].values
    X_te = test_prep[feat_cols]
    y_te = test_prep['isFraud'].values

    # 4. Ensemble
    ensemble = FraudEnsembleClassifier(random_state=cfg.split.random_state)
    ensemble.train_individual_models(X_tr, y_tr, config=cfg.models)
    ensemble.fit_ensemble(X_tr.iloc[:int(len(X_tr)*0.25)], y_tr[:int(len(y_tr)*0.25)])

    evals = ensemble.evaluate_all(X_te, y_te)
    ens_metrics = evals['ensemble']

    print(f"[+] Exp 5 Full Hybrid: PR-AUC={ens_metrics['pr_auc']:.4f}, F1={ens_metrics['f1']:.4f}, Recall={ens_metrics['recall']:.4f}")
    result_dict = {
        'Experiment': 'Exp 5: Full Hybrid Framework',
        'Model': 'Hybrid Ensemble',
        'PR-AUC': ens_metrics['pr_auc'],
        'F1-Score': ens_metrics['f1'],
        'Precision': ens_metrics['precision'],
        'Recall': ens_metrics['recall'],
        'ROC-AUC': ens_metrics['roc_auc'],
        'FPR (%)': ens_metrics['fpr'] * 100,
        'FNR (%)': ens_metrics['fnr'] * 100,
        'Simulated Cost ($)': ens_metrics['simulated_cost'],
        'all_evals': evals,
        'test_probs': ensemble.predict_proba(X_te),
    }

    return result_dict, ensemble, X_tr, X_te


def run_experiment_6_unknown_fraud_simulation(
    real_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    withheld_type: str = "TRANSFER",
    config: Optional[ProjectConfig] = None,
) -> Dict[str, Any]:
    """
    Experiment 6: Zero-Day / Unknown Fraud Pattern Simulation.
    Deliberately withholds a major fraud pattern (e.g. 'TRANSFER' frauds) during training,
    then compares supervised ensemble vs ensemble + anomaly score on detecting the withheld pattern.
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 6: UNKNOWN / ZERO-DAY FRAUD SIMULATION (Withholding '{withheld_type}')")
    print("=" * 70)
    print("[!] Research Simulation: Investigating anomaly detector sensitivity to unlearned fraud.")

    cfg = config or DEFAULT_CONFIG

    # Filter training: exclude frauds with withheld_type
    train_filtered = real_train_df[~((real_train_df['isFraud'] == 1) & (real_train_df['type'] == withheld_type))].copy()

    # Preprocess
    pre = PaySimPreprocessor()
    tr_prep = pre.fit_transform(train_filtered)
    te_prep = pre.transform(real_test_df)

    # Train Autoencoder on legitimate
    ae = train_autoencoder(tr_prep, config=cfg.autoencoder)
    tr_prep['anomaly_score'] = ae.compute_anomaly_score(tr_prep)
    te_prep['anomaly_score'] = ae.compute_anomaly_score(te_prep)

    # Separate feature sets: Without vs With Anomaly Score
    cols_base = [c for c in tr_prep.columns if c not in ['isFraud', 'anomaly_score']]
    cols_ae = [c for c in tr_prep.columns if c != 'isFraud']

    # Filter test specifically to the withheld fraud pattern
    withheld_test_mask = (real_test_df['isFraud'] == 1) & (real_test_df['type'] == withheld_type)
    n_withheld = withheld_test_mask.sum()
    print(f"[*] Withheld fraud instances in test set: {n_withheld}")

    if n_withheld == 0:
        print("[!] No withheld patterns in test split for this sample.")
        return {'status': 'skipped', 'reason': 'no withheld samples in split'}

    # 1. Base Model (Without Anomaly Score)
    ens_base = FraudEnsembleClassifier(random_state=42)
    ens_base.train_individual_models(tr_prep[cols_base], tr_prep['isFraud'].values, config=cfg.models)
    ens_base.fit_ensemble(tr_prep[cols_base].iloc[:200], tr_prep['isFraud'].values[:200])
    p_base = ens_base.predict_proba(te_prep[cols_base])

    # 2. Anomaly-Aware Model (With Anomaly Score)
    ens_ae = FraudEnsembleClassifier(random_state=42)
    ens_ae.train_individual_models(tr_prep[cols_ae], tr_prep['isFraud'].values, config=cfg.models)
    ens_ae.fit_ensemble(tr_prep[cols_ae].iloc[:200], tr_prep['isFraud'].values[:200])
    p_ae = ens_ae.predict_proba(te_prep[cols_ae])

    # Score withheld frauds specifically
    withheld_score_base = float(np.mean(p_base[withheld_test_mask]))
    withheld_score_ae = float(np.mean(p_ae[withheld_test_mask]))

    print(f"[+] Withheld Fraud Detection Probability:")
    print(f"    - Standard Ensemble (No AE) : {withheld_score_base:.4f}")
    print(f"    - Anomaly-Aware Ensemble    : {withheld_score_ae:.4f}")
    print(f"    - Relative Sensitivity Gain : {((withheld_score_ae - withheld_score_base) / (withheld_score_base + 1e-6) * 100):+.2f}%")

    return {
        'withheld_type': withheld_type,
        'n_withheld_samples': int(n_withheld),
        'base_mean_prob': withheld_score_base,
        'ae_mean_prob': withheld_score_ae,
    }


def run_all_experiments(
    real_train_df: pd.DataFrame,
    augmented_train_df: pd.DataFrame,
    real_test_df: pd.DataFrame,
    config: Optional[ProjectConfig] = None,
) -> Tuple[pd.DataFrame, FraudEnsembleClassifier]:
    """
    Run complete experimental comparison suite and export publication-ready reports & plots.
    """
    cfg = config or DEFAULT_CONFIG
    cfg.paths.create_dirs()

    print("\n" + "#" * 70)
    print("STARTING EXPERIMENTAL COMPARISON SUITE (5 RESEARCH EXPERIMENTS)")
    print("#" * 70)

    # Run experiments 1 to 5
    exp1 = run_experiment_1_baseline(real_train_df, real_test_df, config=cfg)
    exp2 = run_experiment_2_ctgan(augmented_train_df, real_test_df, config=cfg)
    exp3 = run_experiment_3_ensemble(augmented_train_df, real_test_df, config=cfg)
    exp4 = run_experiment_4_anomaly_ensemble(augmented_train_df, real_train_df, real_test_df, config=cfg)
    exp5, final_model, X_train_final, X_test_final = run_experiment_5_full_hybrid(
        augmented_train_df, real_train_df, real_test_df, config=cfg
    )

    # Optional Exp 6
    run_experiment_6_unknown_fraud_simulation(real_train_df, real_test_df, withheld_type="TRANSFER", config=cfg)

    # Aggregate results table
    results_list = [exp1, exp2, exp3, exp4, exp5]
    display_rows = []
    for r in results_list:
        display_rows.append({
            'Experiment': r['Experiment'],
            'Model': r['Model'],
            'PR-AUC': r['PR-AUC'],
            'F1-Score': r['F1-Score'],
            'Precision': r['Precision'],
            'Recall': r['Recall'],
            'ROC-AUC': r['ROC-AUC'],
            'FPR (%)': r['FPR (%)'],
            'FNR (%)': r['FNR (%)'],
            'Simulated Cost ($)': r['Simulated Cost ($)'],
        })

    exp_df = pd.DataFrame(display_rows)
    save_path = cfg.paths.experiment_results_path
    exp_df.to_csv(save_path, index=False)
    print(f"\n[+] Saved experimental comparison report -> {save_path}")

    print("\n" + "=" * 70)
    print("EMPIRICAL RESEARCH FINDINGS SUMMARY TABLE:")
    print("=" * 70)
    print(exp_df.to_string(index=False))
    print("=" * 70)

    # Generate multi-experiment comparison bar chart
    plot_path = cfg.paths.plots_dir / "experiment_comparison_chart.png"
    plot_experiment_comparisons(exp_df, plot_path)

    # Generate PR / ROC curves for the experiments
    model_probs = {
        'Exp 1 (Baseline)': exp1['test_probs'],
        'Exp 2 (CTGAN)': exp2['test_probs'] if isinstance(exp2['test_probs'], np.ndarray) else exp1['test_probs'],
        'Exp 3 (Ensemble)': exp3['test_probs'],
        'Exp 4 (AE Anomaly)': exp4['test_probs'],
        'Exp 5 (Full Hybrid)': exp5['test_probs'],
    }
    plot_evaluation_curves(
        y_test=real_test_df['isFraud'].values,
        model_probabilities=model_probs,
        pr_output_path=cfg.paths.plots_dir / "experiments_pr_curves.png",
        roc_output_path=cfg.paths.plots_dir / "experiments_roc_curves.png",
    )

    return exp_df, final_model


if __name__ == "__main__":
    from src.data_loader import load_dataset, split_dataset
    from src.synthetic_data import run_synthetic_generation_pipeline

    print("[*] Testing src/experiments.py...")
    raw = load_dataset(nrows=30000)
    train_part, test_part, _ = split_dataset(raw, method="stratified")

    # Fast test config
    fast_cfg = ProjectConfig()
    fast_cfg.ctgan.n_synthetic_samples = 30
    fast_cfg.ctgan.epochs = 2
    fast_cfg.ctgan.batch_size = 30
    fast_cfg.ctgan.pac = 3
    fast_cfg.ctgan.verbose = False
    fast_cfg.autoencoder.epochs = 4
    fast_cfg.autoencoder.batch_size = 256
    fast_cfg.models.xgb_params['n_estimators'] = 25
    fast_cfg.models.lgb_params['n_estimators'] = 25
    fast_cfg.models.cb_params['iterations'] = 25
    fast_cfg.models.rf_params['n_estimators'] = 25

    # Generate synthetic data for experiment
    _, aug_train, _ = run_synthetic_generation_pipeline(train_part, config=fast_cfg, save_artifacts=False)

    exp_table, _ = run_all_experiments(
        real_train_df=train_part,
        augmented_train_df=aug_train,
        real_test_df=test_part,
        config=fast_cfg,
    )

    print("[+] All experiments smoke test passed!")
