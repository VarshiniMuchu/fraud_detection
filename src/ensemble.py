"""
Ensemble Learning Module for Financial Fraud Detection.

Part of Research Project:
Hybrid Synthetic-Data and Anomaly-Aware Ensemble Learning for Financial Fraud Detection.

Models Implemented:
1. XGBoost (Extreme Gradient Boosting)
2. LightGBM (Light Gradient Boosting Machine)
3. CatBoost (Categorical Boosting)
4. Random Forest (Bagged Decision Tree Ensemble)

Ensemble Strategies:
- Soft Voting: Optimal weighted probability combination based on validation PR-AUC.
- Stacking: Logistic Regression meta-learner trained on model decision probabilities.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import DEFAULT_CONFIG, ProjectConfig, ModelConfig, PathConfig
from src.metrics import compute_fraud_metrics, find_optimal_threshold


class FraudEnsembleClassifier:
    """
    Ensemble container combining XGBoost, LightGBM, CatBoost, and Random Forest.
    """

    def __init__(
        self,
        ensemble_method: str = "soft_voting",  # 'soft_voting' or 'stacking'
        random_state: int = 42,
    ):
        self.ensemble_method = ensemble_method
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.meta_learner: Optional[LogisticRegression] = None
        self.feature_names: List[str] = []
        self.optimal_threshold: float = 0.5
        self.is_fitted: bool = False

    def _get_scale_pos_weight(self, y: np.ndarray) -> float:
        """Compute imbalance scaling factor (neg / pos)."""
        pos = int(np.sum(y == 1))
        neg = int(np.sum(y == 0))
        if pos == 0:
            return 1.0
        return float(neg / pos)

    def train_individual_models(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        config: Optional[ModelConfig] = None,
    ) -> Dict[str, Any]:
        """
        Train all four gradient boosting and tree classifiers.
        """
        self.feature_names = list(X_train.columns)
        cfg = config or DEFAULT_CONFIG.models
        spw = self._get_scale_pos_weight(y_train)

        print(f"[*] Training 4 Individual Classifiers (Class Imbalance Scale: {spw:.2f}):")

        # 1. XGBoost
        print("    [1/4] Training XGBoost Classifier...")
        xgb_params = dict(getattr(cfg, 'xgb_params', {
            'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'eval_metric': 'logloss'
        }))
        xgb_params['scale_pos_weight'] = spw
        xgb_params['random_state'] = self.random_state
        xgb_params['n_jobs'] = -1
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model

        # 2. LightGBM
        print("    [2/4] Training LightGBM Classifier...")
        lgb_params = dict(getattr(cfg, 'lgb_params', {
            'n_estimators': 100, 'max_depth': 6, 'num_leaves': 31,
            'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.8
        }))
        lgb_params['scale_pos_weight'] = spw
        lgb_params['random_state'] = self.random_state
        lgb_params['verbose'] = -1
        lgb_params['n_jobs'] = -1
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train)
        self.models['lightgbm'] = lgb_model

        # 3. CatBoost
        print("    [3/4] Training CatBoost Classifier...")
        cb_params = dict(getattr(cfg, 'cb_params', {
            'iterations': 100, 'depth': 6, 'learning_rate': 0.05
        }))
        cb_params['scale_pos_weight'] = spw
        cb_params['random_seed'] = self.random_state
        cb_params['verbose'] = 0
        cb_params['thread_count'] = -1
        cb_model = cb.CatBoostClassifier(**cb_params)
        cb_model.fit(X_train, y_train)
        self.models['catboost'] = cb_model

        # 4. Random Forest
        print("    [4/4] Training Random Forest Classifier...")
        rf_params = dict(getattr(cfg, 'rf_params', {
            'n_estimators': 100, 'max_depth': 12, 'class_weight': 'balanced'
        }))
        rf_params['random_state'] = self.random_state
        rf_params['n_jobs'] = -1
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)
        self.models['random_forest'] = rf_model

        print("[+] Individual model training completed.")
        return self.models

    def predict_individual_probabilities(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate fraud probabilities from each individual model."""
        probs = {}
        for name, model in self.models.items():
            probs[name] = model.predict_proba(X[self.feature_names])[:, 1]
        return probs

    def fit_ensemble(
        self,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        method: Optional[str] = None,
    ):
        """
        Calibrate ensemble weights or fit stacking meta-learner on validation predictions.
        """
        if method is not None:
            self.ensemble_method = method

        val_probs = self.predict_individual_probabilities(X_val)

        if self.ensemble_method == "soft_voting":
            # Weight models proportionally to their individual PR-AUC on validation split
            from sklearn.metrics import average_precision_score
            scores = {}
            for name, p in val_probs.items():
                pr_auc = average_precision_score(y_val, p)
                scores[name] = max(1e-4, pr_auc)

            total_score = sum(scores.values())
            self.weights = {k: v / total_score for k, v in scores.items()}
            print(f"[*] Soft Voting Ensemble Weights (calibrated on Val PR-AUC):")
            for k, w in self.weights.items():
                print(f"    - {k:<15}: {w:.4f} (PR-AUC: {scores[k]:.4f})")

        elif self.ensemble_method == "stacking":
            print("[*] Training Stacking Logistic Regression Meta-Learner...")
            meta_features = np.column_stack([val_probs[name] for name in sorted(self.models.keys())])
            self.meta_learner = LogisticRegression(
                class_weight='balanced',
                C=0.05,
                max_iter=500,
                random_state=self.random_state,
            )
            self.meta_learner.fit(meta_features, y_val)
            print("[+] Stacking Meta-Learner fitted.")

        # Tune optimal decision threshold on validation ensemble probabilities (F2 = recall-biased)
        ens_val_probs = self.predict_proba(X_val)
        self.optimal_threshold, best_f2 = find_optimal_threshold(y_val, ens_val_probs, metric="f2")
        print(f"[+] Optimal Decision Threshold (F2-biased): {self.optimal_threshold:.4f} (Validation F2 = {best_f2:.4f})")

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ensemble fraud probabilities."""
        ind_probs = self.predict_individual_probabilities(X)

        if self.ensemble_method == "stacking" and self.meta_learner is not None:
            meta_X = np.column_stack([ind_probs[name] for name in sorted(self.models.keys())])
            return self.meta_learner.predict_proba(meta_X)[:, 1]
        else:
            # Soft voting
            if not self.weights:
                # Default equal weights if not calibrated
                n = len(self.models)
                self.weights = {k: 1.0 / n for k in self.models.keys()}

            combined = np.zeros(len(X), dtype=float)
            for name, w in self.weights.items():
                combined += w * ind_probs[name]
            return combined

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary fraud labels using threshold."""
        thresh = threshold if threshold is not None else self.optimal_threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def evaluate_all(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate each individual model and the combined ensemble on test data.
        """
        thresh = threshold if threshold is not None else self.optimal_threshold
        results = {}

        # Individual models
        ind_probs = self.predict_individual_probabilities(X_test)
        for name, probs in ind_probs.items():
            results[name] = compute_fraud_metrics(y_test, probs, threshold=thresh)

        # Ensemble
        ens_probs = self.predict_proba(X_test)
        results['ensemble'] = compute_fraud_metrics(y_test, ens_probs, threshold=thresh)

        return results

    def save(self, paths: Optional[PathConfig] = None):
        """Save models to their respective directories."""
        cfg_paths = paths or DEFAULT_CONFIG.paths
        cfg_paths.create_dirs()

        # 1. XGBoost
        if 'xgboost' in self.models:
            self.models['xgboost'].save_model(str(cfg_paths.xgboost_dir / "xgboost_model.json"))
        # 2. LightGBM
        if 'lightgbm' in self.models:
            self.models['lightgbm'].booster_.save_model(str(cfg_paths.lightgbm_dir / "lightgbm_model.txt"))
        # 3. CatBoost
        if 'catboost' in self.models:
            self.models['catboost'].save_model(str(cfg_paths.catboost_dir / "catboost_model.cbm"))
        # 4. Random Forest
        if 'random_forest' in self.models:
            joblib.dump(self.models['random_forest'], cfg_paths.rf_dir / "rf_model.joblib")

        # 5. Ensemble metadata
        ensemble_meta = {
            'weights': self.weights,
            'meta_learner': self.meta_learner,
            'feature_names': self.feature_names,
            'optimal_threshold': self.optimal_threshold,
            'ensemble_method': self.ensemble_method,
        }
        joblib.dump(ensemble_meta, cfg_paths.ensemble_dir / "ensemble_meta.joblib")
        print(f"[+] All models successfully persisted to {cfg_paths.models_dir}")


def train_and_evaluate_ensemble(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    ensemble_method: str = "soft_voting",
    config: Optional[ProjectConfig] = None,
) -> Tuple[FraudEnsembleClassifier, Dict[str, Dict[str, Any]]]:
    """Convenience pipeline function to train and evaluate ensemble."""
    cfg = config or DEFAULT_CONFIG

    # Stratified train/val split within training set for ensemble calibration
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=cfg.split.random_state)
    idx_tr, idx_val = next(sss.split(X_train, y_train))

    X_tr = X_train.iloc[idx_tr].copy()
    y_tr = y_train[idx_tr]
    X_val = X_train.iloc[idx_val].copy()
    y_val = y_train[idx_val]

    ensemble = FraudEnsembleClassifier(ensemble_method=ensemble_method, random_state=cfg.split.random_state)
    ensemble.train_individual_models(X_tr, y_tr, config=cfg.models)
    ensemble.fit_ensemble(X_val, y_val, method=ensemble_method)

    results = ensemble.evaluate_all(X_test, y_test)
    ensemble.save(cfg.paths)

    return ensemble, results


if __name__ == "__main__":
    from src.data_loader import load_dataset, split_dataset
    from src.feature_engineering import PaySimFeatureEngineer
    from src.preprocessing import PaySimPreprocessor
    from src.metrics import format_metrics_table

    print("[*] Testing src/ensemble.py...")
    raw = load_dataset(nrows=30000)
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

    # Fast model config
    fast_proj_cfg = ProjectConfig()
    fast_proj_cfg.models.xgb_params['n_estimators'] = 30
    fast_proj_cfg.models.lgb_params['n_estimators'] = 30
    fast_proj_cfg.models.cb_params['iterations'] = 30
    fast_proj_cfg.models.rf_params['n_estimators'] = 30

    ensemble, metrics = train_and_evaluate_ensemble(
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_te,
        y_test=y_te,
        ensemble_method="soft_voting",
        config=fast_proj_cfg,
    )

    summary_table = format_metrics_table(metrics)
    print("\n[*] Smoke Test Performance Summary:")
    print(summary_table.to_string(index=False))
    print("[+] Ensemble smoke test passed!")
