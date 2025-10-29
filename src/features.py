# src/features.py
"""
Feature engineering and preprocessing functions.
"""
import pandas as pd
import numpy as np

# scikit-learn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, StandardScaler

def prune_by_correlation(df: pd.DataFrame, target_col: str, thr: float = 0.95, random_state: int = 42) -> (pd.DataFrame, list):
    """
    Remove redundant features by absolute correlation threshold.
    """
    rng = np.random.RandomState(random_state)
    df_copy = df.copy()
    
    feat_cols = [c for c in df_copy.columns if c != target_col and pd.api.types.is_numeric_dtype(df_copy[c])]
    if len(feat_cols) < 2:
        return df_copy, []

    corr = df_copy[feat_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    kept = set(feat_cols)

    # Greedy elimination based on sum of correlations
    for col in upper.columns:
        high_pairs = upper[col][upper[col] > thr].dropna()
        for row, val in high_pairs.items():
            if row in kept and col in kept:
                s_row = corr[row].sum()
                s_col = corr[col].sum()
                drop = row if s_row >= s_col else col
                kept.discard(drop)
                to_drop.add(drop)

    pruned_df = df_copy.drop(columns=list(to_drop), errors="ignore")
    dropped_cols = sorted(list(to_drop))
    
    print(f"Correlation Pruning: Dropped {len(dropped_cols)} columns with threshold > {thr}")
    return pruned_df, dropped_cols

def create_preprocessor(available_cols: pd.Index, 
                        static_nominal_cols: list, 
                        static_ordinal_cols: list, 
                        static_numeric_cols: list) -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for preprocessing.
    
    Filters the static column lists to only include columns
    that are present in the provided available_cols (e.g., X.columns).
    """
    
    # --- Find columns that *actually exist* in the dataframe ---
    nominal_cols_to_use = [col for col in static_nominal_cols if col in available_cols]
    ordinal_cols_to_use = [col for col in static_ordinal_cols if col in available_cols]
    numeric_cols_to_use = [col for col in static_numeric_cols if col in available_cols]

    # --- Define transformers ---
    nominal_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    
    ordinal_transformer = StandardScaler()
    
    numeric_discrete_transformer = Pipeline(steps=[
        ("log1p", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", RobustScaler())
    ])
    
    # --- Build the ColumnTransformer ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("nom", nominal_transformer, nominal_cols_to_use),
            ("ord", ordinal_transformer, ordinal_cols_to_use),
            ("num", numeric_discrete_transformer, numeric_cols_to_use),
        ],
        remainder="drop" # Drop any columns not specified
    )

    print("Preprocessor created. Columns to be processed:")
    print(f"  Nominal: {len(nominal_cols_to_use)} (out of {len(static_nominal_cols)} defined)")
    print(f"  Ordinal: {len(ordinal_cols_to_use)} (out of {len(static_ordinal_cols)} defined)")
    print(f"  Numeric: {len(numeric_cols_to_use)} (out of {len(static_numeric_cols)} defined)")
    
    return preprocessor