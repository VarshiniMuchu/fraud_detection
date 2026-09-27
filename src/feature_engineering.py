"""
Feature Engineering Module for PaySim Financial Fraud Detection.

Part of Research Project:
Hybrid Synthetic-Data and Anomaly-Aware Ensemble Learning for Financial Fraud Detection.

Constructs fraud-sensitive domain and behavioral features:
1. Balance Discrepancy & Accounting Errors:
   - orig_balance_error = (newbalanceOrig + amount) - oldbalanceOrg
   - dest_balance_error = (oldbalanceDest + amount) - newbalanceDest
2. Liquidity & Capital Ratios:
   - amount_to_oldbalanceOrg_ratio = amount / (oldbalanceOrg + 1.0)
   - amount_to_oldbalanceDest_ratio = amount / (oldbalanceDest + 1.0)
3. Zero-Balance & Account Drainage Flags:
   - is_zero_orig_balance: Sender starts with zero balance
   - is_zero_new_orig: Sender's account completely emptied
   - is_zero_dest_balance: Beneficiary starts with zero balance
   - is_orig_emptied: Old balance > 0 and new balance == 0 (classic cashout pattern)
   - is_full_transfer: Transferred amount matches total available balance
4. Temporal Dynamics:
   - hour_of_day = step % 24
   - day_of_week = (step // 24) % 7
5. High-Risk Channel Flags:
   - is_high_risk_type: Boolean flag for TRANSFER and CASH_OUT types
"""

import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import joblib

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import DEFAULT_CONFIG, ProjectConfig


class PaySimFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Leakage-free Feature Engineering Transformer for PaySim Fraud Detection.
    Computes accounting discrepancies, liquidity ratios, account drainage indicators,
    and temporal cycles.
    """

    def __init__(self):
        self.engineered_feature_names: List[str] = [
            'orig_balance_error',
            'dest_balance_error',
            'amount_to_oldbalanceOrg_ratio',
            'amount_to_oldbalanceDest_ratio',
            'is_zero_orig_balance',
            'is_zero_new_orig',
            'is_zero_dest_balance',
            'is_orig_emptied',
            'is_full_transfer',
            'hour_of_day',
            'day_of_week',
            'is_high_risk_type',
            # Additional high-signal features
            'log_amount',
            'orig_net_change',
            'dest_net_change',
            'balance_change_ratio',
            'is_round_amount',
            'amount_gt_dest_balance',
            'net_orig_dest_flow',
        ]
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fit method (stateless as transformations are deterministic domain calculations)."""
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Generate engineered features and append them to the input DataFrame.
        """
        df = X.copy()

        # 1. Accounting and balance discrepancies
        # In a legitimate transfer: oldbalanceOrg - amount == newbalanceOrig
        # Hence: (newbalanceOrig + amount) - oldbalanceOrg should be 0.
        if 'newbalanceOrig' in df.columns and 'amount' in df.columns and 'oldbalanceOrg' in df.columns:
            df['orig_balance_error'] = (df['newbalanceOrig'] + df['amount']) - df['oldbalanceOrg']
        else:
            df['orig_balance_error'] = 0.0

        if 'oldbalanceDest' in df.columns and 'amount' in df.columns and 'newbalanceDest' in df.columns:
            df['dest_balance_error'] = (df['oldbalanceDest'] + df['amount']) - df['newbalanceDest']
        else:
            df['dest_balance_error'] = 0.0

        # 2. Financial ratios (with epsilon smoothing to avoid division by zero)
        if 'amount' in df.columns and 'oldbalanceOrg' in df.columns:
            df['amount_to_oldbalanceOrg_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1.0)
            df['amount_to_oldbalanceOrg_ratio'] = df['amount_to_oldbalanceOrg_ratio'].clip(upper=100.0)
        else:
            df['amount_to_oldbalanceOrg_ratio'] = 0.0

        if 'amount' in df.columns and 'oldbalanceDest' in df.columns:
            df['amount_to_oldbalanceDest_ratio'] = df['amount'] / (df['oldbalanceDest'] + 1.0)
            df['amount_to_oldbalanceDest_ratio'] = df['amount_to_oldbalanceDest_ratio'].clip(upper=100.0)
        else:
            df['amount_to_oldbalanceDest_ratio'] = 0.0

        # 3. Account drainage & zero-balance indicators
        if 'oldbalanceOrg' in df.columns:
            df['is_zero_orig_balance'] = (df['oldbalanceOrg'] == 0).astype(np.int32)
        else:
            df['is_zero_orig_balance'] = 0

        if 'newbalanceOrig' in df.columns:
            df['is_zero_new_orig'] = (df['newbalanceOrig'] == 0).astype(np.int32)
        else:
            df['is_zero_new_orig'] = 0

        if 'oldbalanceDest' in df.columns:
            df['is_zero_dest_balance'] = (df['oldbalanceDest'] == 0).astype(np.int32)
        else:
            df['is_zero_dest_balance'] = 0

        if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns:
            df['is_orig_emptied'] = ((df['oldbalanceOrg'] > 0) & (df['newbalanceOrig'] == 0)).astype(np.int32)
        else:
            df['is_orig_emptied'] = 0

        if 'amount' in df.columns and 'oldbalanceOrg' in df.columns:
            df['is_full_transfer'] = (np.abs(df['amount'] - df['oldbalanceOrg']) < 0.01).astype(np.int32)
        else:
            df['is_full_transfer'] = 0

        # 4. Temporal periodicity features
        if 'step' in df.columns:
            steps = df['step'].astype(int)
            df['hour_of_day'] = steps % 24
            df['day_of_week'] = (steps // 24) % 7
        else:
            df['hour_of_day'] = 0
            df['day_of_week'] = 0

        # 5. Channel risk flag
        if 'type' in df.columns:
            type_str = df['type'].astype(str)
            df['is_high_risk_type'] = type_str.isin(['TRANSFER', 'CASH_OUT']).astype(np.int32)
        else:
            # Check for one-hot encoded columns
            has_ohe = any(col in df.columns for col in ['type_TRANSFER', 'type_CASH_OUT'])
            if has_ohe:
                transfer_val = df.get('type_TRANSFER', 0)
                cashout_val = df.get('type_CASH_OUT', 0)
                df['is_high_risk_type'] = ((transfer_val == 1) | (cashout_val == 1)).astype(np.int32)
            else:
                df['is_high_risk_type'] = 0

        # 6. Log-transformed amount (reduce skewness, helps tree models)
        if 'amount' in df.columns:
            df['log_amount'] = np.log1p(df['amount'].clip(lower=0))
        else:
            df['log_amount'] = 0.0

        # 7. Net balance change for originator (negative means money left)
        if 'oldbalanceOrg' in df.columns and 'newbalanceOrig' in df.columns:
            df['orig_net_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        else:
            df['orig_net_change'] = 0.0

        # 8. Net balance change for destination (positive means money arrived)
        if 'oldbalanceDest' in df.columns and 'newbalanceDest' in df.columns:
            df['dest_net_change'] = df['newbalanceDest'] - df['oldbalanceDest']
        else:
            df['dest_net_change'] = 0.0

        # 9. Ratio of amount to total originator balance change (fraud: often ~1.0)
        if 'amount' in df.columns and 'orig_net_change' in df.columns:
            denom = df['amount'].abs() + 1.0
            df['balance_change_ratio'] = (df['orig_net_change'].abs() / denom).clip(upper=10.0)
        else:
            df['balance_change_ratio'] = 0.0

        # 10. Round-number amount flag (fraudsters often use round numbers)
        if 'amount' in df.columns:
            df['is_round_amount'] = ((df['amount'] % 1000 == 0) & (df['amount'] > 0)).astype(np.int32)
        else:
            df['is_round_amount'] = 0

        # 11. Amount greater than destination's old balance (suspicious inflow)
        if 'amount' in df.columns and 'oldbalanceDest' in df.columns:
            df['amount_gt_dest_balance'] = (df['amount'] > df['oldbalanceDest'] + 1).astype(np.int32)
        else:
            df['amount_gt_dest_balance'] = 0

        # 12. Net flow between orig and dest (combined signal)
        if 'orig_net_change' in df.columns and 'dest_net_change' in df.columns:
            df['net_orig_dest_flow'] = df['orig_net_change'] + df['dest_net_change']
        else:
            df['net_orig_dest_flow'] = 0.0

        return df

    def get_feature_names_out(self) -> List[str]:
        """Return the names of the newly constructed features."""
        return list(self.engineered_feature_names)

    def save(self, filepath: Path):
        """Save fitted transformer artifact."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "PaySimFeatureEngineer":
        """Load transformer artifact."""
        return joblib.load(filepath)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to engineer features on a DataFrame."""
    fe = PaySimFeatureEngineer()
    return fe.fit_transform(df)


if __name__ == "__main__":
    from src.data_loader import load_dataset
    print("[*] Testing src/feature_engineering.py...")
    sample_df = load_dataset(nrows=1000)
    fe = PaySimFeatureEngineer()
    engineered_df = fe.fit_transform(sample_df)
    
    print(f"[+] Original columns: {len(sample_df.columns)}")
    print(f"[+] Engineered columns: {len(engineered_df.columns)}")
    print(f"[+] Added features: {fe.get_feature_names_out()}")
    print(f"[+] Sample engineered row:\n{engineered_df[fe.get_feature_names_out()].head(3)}")
    print("[+] Feature engineering test passed!")
