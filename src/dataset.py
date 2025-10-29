# src/dataset.py
"""
Data loading and cleaning pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_raw_data(path: Path, columns: list) -> pd.DataFrame:
    """Loads the raw CSV file with no headers."""
    if not path.exists():
        logging.error(f"Raw data file not found at: {path}")
        raise FileNotFoundError(f"No se encontró '{path}'.")
    
    df = pd.read_csv(path, header=None, dtype=str)
    
    # Ensure correct number of columns
    if df.shape[1] > len(columns):
        df = df.iloc[:, :len(columns)]
    if df.shape[1] < len(columns):
        for c in range(df.shape[1], len(columns)):
            df[c] = np.nan
            
    df.columns = columns
    
    # Force numeric type (non-numeric -> NaN)
    df = df.apply(lambda s: pd.to_numeric(s.str.strip(), errors='coerce') if s.dtype == object else pd.to_numeric(s, errors='coerce'))
    return df

def _analyze_and_repair(df: pd.DataFrame, expected_ranges: dict, rare_threshold: float) -> pd.DataFrame:
    """Repairs data by clipping or setting NaN based on expected ranges."""
    out = df.copy()
    n = len(out)
    for col, (lo, hi) in expected_ranges.items():
        if col not in out.columns: continue
        
        s = out[col]
        mask_out = (~s.isna()) & ((s < lo) | (s > hi))
        cnt = int(mask_out.sum())
        if cnt == 0: continue
        
        pct = cnt / n if n > 0 else 0
        logging.info(f"{col}: {cnt} ({pct:.2%}) values out of range [{lo},{hi}]")
        
        if pct <= rare_threshold:
            out.loc[mask_out, col] = np.nan
            logging.info("  -> Rare: Set to NaN (will be imputed)")
        else:
            out[col] = s.clip(lo, hi)
            logging.info("  -> Widespread: Clipped to range")
            
    return out

def _finalize_and_impute(df_repaired: pd.DataFrame, expected_ranges: dict) -> pd.DataFrame:
    """Rounds, imputes missing values, and casts to integer."""
    df = df_repaired.copy()
    
    # Round numeric values to int while keeping NaN
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].round().where(~df[c].isna(), np.nan)

    # Define imputation groups
    policy = [c for c, v in expected_ranges.items() if v == (1, 12)]
    binary = [c for c, v in expected_ranges.items() if v == (0, 1)]
    catsm = [c for c in expected_ranges if c not in policy and c not in binary and expected_ranges[c][1] <= 9]

    # Impute categorical (mode)
    for c in catsm:
        if c in df.columns:
            mode_val = df[c].mode(dropna=True)
            fill = int(mode_val.iloc[0]) if len(mode_val) > 0 else int(expected_ranges[c][0])
            df[c] = df[c].fillna(fill).astype(int)
            
    # Impute policy counts (median)
    for c in policy:
        if c in df.columns:
            med = df[c].median(skipna=True)
            med = expected_ranges[c][0] if np.isnan(med) else med
            df[c] = df[c].fillna(int(round(med))).astype(int)
            
    # Impute binary (mode)
    for c in binary:
        if c in df.columns:
            mode_val = df[c].mode(dropna=True)
            fill = int(mode_val.iloc[0]) if len(mode_val) > 0 else expected_ranges[c][0]
            df[c] = df[c].fillna(fill).astype(int)

    # Impute remaining NaNs (median)
    for c in df.columns:
        if df[c].isna().any():
            med = df[c].median(skipna=True)
            fill_val = int(round(med)) if not np.isnan(med) else 0
            df[c] = df[c].fillna(fill_val)

    # Final cast to int
    df = df.astype(int)
    return df

def clean_raw_data(in_path: Path, out_path: Path, columns: list, expected_ranges: dict, rare_threshold: float):
    """
    Main data cleaning pipeline. Loads, repairs, imputes, and saves the data.
    """
    logging.info("--- Data Cleaning Pipeline Started ---")
    
    # 1. Load
    df = _load_raw_data(in_path, columns)
    logging.info(f"Loaded raw data: {df.shape}")
    
    # 2. Repair
    repaired = _analyze_and_repair(df, expected_ranges, rare_threshold)
    logging.info("Data analysis and repair complete.")
    
    # 3. Impute & Finalize
    cleaned = _finalize_and_impute(repaired, expected_ranges)
    logging.info("Imputation and finalization complete.")
    
    # 4. Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_path, index=False)
    logging.info(f"Cleaned data saved to: {out_path}")
    logging.info("--- Data Cleaning Pipeline Finished ---")

def load_processed_data(path: Path) -> pd.DataFrame:
    """Loads the final, cleaned data for modeling."""
    if not path.exists():
        logging.error(f"Processed data file not found at: {path}")
        logging.error("Please run the data cleaning pipeline first.")
        raise FileNotFoundError(f"No se encontró '{path}'.")
    
    df = pd.read_csv(path)
    logging.info(f"Loaded processed data from: {path}")
    return df