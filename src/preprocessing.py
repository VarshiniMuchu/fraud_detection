"""
Data Preprocessing Pipeline for PaySim Fraud Detection.

Part of Research Project:
Hybrid Synthetic-Data and Anomaly-Aware Ensemble Learning for Financial Fraud Detection.

Principles:
- Strict zero-leakage enforcement: scalers and encodings are fit ONLY on the training split.
- Test transformations are applied purely out-of-sample.
- Proper handling of skewed financial distributions and categorical transaction types.
- Preservation of original data formats for CTGAN when needed.
"""

import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
import joblib

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import DEFAULT_CONFIG


class PaySimPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leakage-free Preprocessor for the PaySim Financial Transaction Dataset.
    
    Transforms:
    - Drops non-generalizable customer identifiers ('nameOrig', 'nameDest', 'isFlaggedFraud')
    - Encodes categorical transaction type via One-Hot Encoding (learned on train only)
    - Scales numerical amounts and balances via RobustScaler (robust to extreme financial outliers)
    - Tracks feature names for downstream model interpretability (SHAP)
    """

    def __init__(
        self,
        target_col: str = "isFraud",
        drop_identifiers: bool = True,
        scaler_type: str = "robust"
    ):
        self.target_col = target_col
        self.drop_identifiers = drop_identifiers
        self.scaler_type = scaler_type
        
        self.ohe: Optional[OneHotEncoder] = None
        self.scaler: Optional[RobustScaler | StandardScaler] = None
        self.categorical_cols: List[str] = ['type']
        self.numerical_cols: List[str] = [
            'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest'
        ]
        self.identifier_cols: List[str] = ['nameOrig', 'nameDest', 'isFlaggedFraud']
        self.feature_names_out: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit encodings and scaling solely on the training partition.
        """
        X_copy = X.copy()
        
        # Verify columns exist
        available_num_cols = [c for c in self.numerical_cols if c in X_copy.columns]
        self.numerical_cols = available_num_cols
        
        # Fit OneHotEncoder for 'type' with compatibility for older/newer sklearn versions.
        if 'type' in X_copy.columns:
            try:
                self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            except TypeError:
                self.ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.ohe.fit(X_copy[['type']])
            encoded_type_names = list(self.ohe.get_feature_names_out(['type']))
        else:
            self.ohe = None
            encoded_type_names = []

        # Fit Scaler for numerical features
        if self.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

        if self.numerical_cols:
            self.scaler.fit(X_copy[self.numerical_cols])
            
        # Compile feature names out
        self.feature_names_out = (
            (['step'] if 'step' in X_copy.columns else []) +
            encoded_type_names +
            self.numerical_cols
        )
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a dataset (train or unseen test) using parameters fit on train.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
            
        X_copy = X.copy()
        
        # 1. Preserve step and target if present
        step_series = X_copy['step'].values if 'step' in X_copy.columns else None
        target_series = X_copy[self.target_col].values if self.target_col in X_copy.columns else None
        
        # 2. Categorical transformation
        if self.ohe is not None and 'type' in X_copy.columns:
            ohe_array = self.ohe.transform(X_copy[['type']])
            if sparse.issparse(ohe_array):
                ohe_array = ohe_array.toarray()  # type: ignore[attr-defined]
            ohe_df = pd.DataFrame(
                np.asarray(ohe_array),
                columns=self.ohe.get_feature_names_out(['type']),
                index=X_copy.index
            )
        else:
            ohe_df = pd.DataFrame(index=X_copy.index)
            
        # 3. Numerical transformation
        if self.scaler is not None and self.numerical_cols:
            scaled_num = self.scaler.transform(X_copy[self.numerical_cols])
            num_df = pd.DataFrame(
                scaled_num,
                columns=self.numerical_cols,
                index=X_copy.index
            )
        else:
            num_df = pd.DataFrame(index=X_copy.index)
            
        # 4. Assemble transformed feature dataframe
        parts = []
        if step_series is not None:
            parts.append(pd.DataFrame({'step': step_series}, index=X_copy.index))
        if not ohe_df.empty:
            parts.append(ohe_df)
        if not num_df.empty:
            parts.append(num_df)
            
        transformed_df = pd.concat(parts, axis=1)
        
        # 5. Append target if present (unscaled)
        if target_series is not None:
            transformed_df[self.target_col] = target_series
            
        return transformed_df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params: Any) -> pd.DataFrame:  # type: ignore[override]
        """Fit on train and return transformed train DataFrame."""
        return self.fit(X, y).transform(X)

    def save(self, filepath: Path) -> None:
        """Save fitted preprocessor artifact to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[+] Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "PaySimPreprocessor":
        """Load a fitted preprocessor artifact from disk."""
        return joblib.load(filepath)


def prepare_for_ctgan(train_df: pd.DataFrame, target_col: str = "isFraud") -> pd.DataFrame:
    """
    Extract strictly fraud records from the training split for CTGAN generation.
    Removes raw non-generalizable strings (nameOrig, nameDest).
    Preserves original numeric values and categorical types as required by CTGAN.
    """
    fraud_df = train_df[train_df[target_col] == 1].copy()
    
    # Drop identifiers that are purely unique strings
    cols_to_drop = [c for c in ['nameOrig', 'nameDest', 'isFlaggedFraud'] if c in fraud_df.columns]
    if cols_to_drop:
        fraud_df = fraud_df.drop(columns=cols_to_drop)
        
    return fraud_df


if __name__ == "__main__":
    from src.data_loader import load_dataset, split_dataset
    print("[*] Testing preprocessing.py...")
    raw_df = load_dataset(nrows=50000)
    train_df, test_df, _ = split_dataset(raw_df, method="temporal")
    
    preprocessor = PaySimPreprocessor()
    transformed_train = preprocessor.fit_transform(train_df)
    transformed_test = preprocessor.transform(test_df)
    
    print(f"[+] Transformed Train Shape: {transformed_train.shape}")
    print(f"[+] Transformed Test Shape: {transformed_test.shape}")
    print(f"[+] Output Columns: {list(transformed_train.columns)}")
    
    # Check CTGAN extraction
    ctgan_input = prepare_for_ctgan(train_df)
    print(f"[+] Fraud rows prepared for CTGAN: {len(ctgan_input)}")
    print("[+] Preprocessing pipeline verification complete!")
