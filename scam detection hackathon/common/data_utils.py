"""
Shared helpers for loading the four datasets into a consistent shape,
regardless of which public dataset variant you downloaded.
"""

import pandas as pd
import numpy as np


def _find_column(df: pd.DataFrame, aliases):
    for alias in aliases:
        if alias in df.columns:
            return alias
    # case-insensitive fallback
    lower_map = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None


def load_text_dataset(cfg: dict):
    """For sms / url datasets: returns (texts: list[str], labels: np.ndarray[int])."""
    df = pd.read_csv(cfg["path"], encoding="latin-1", low_memory=False)
    df = df.dropna(axis=1, how="all")

    text_col = _find_column(df, cfg["text_col_aliases"])
    label_col = _find_column(df, cfg["label_col_aliases"])

    if text_col is None or label_col is None:
        raise ValueError(
            f"Could not find text/label columns in {cfg['path']}. "
            f"Found columns: {list(df.columns)}. "
            f"Add the exact column name to text_col_aliases/label_col_aliases "
            f"in backend/common/config.py."
        )

    df = df[[text_col, label_col]].dropna()
    texts = df[text_col].astype(str).tolist()

    positives = set(str(v).lower() for v in cfg["positive_values"])
    labels = df[label_col].apply(
        lambda v: 1 if str(v).strip().lower() in positives else 0
    ).values.astype(int)

    return texts, labels, text_col, label_col


def load_tabular_dataset(cfg: dict):
    """For bank / credit card datasets: returns (X: DataFrame, y: np.ndarray[int], feature_names)."""
    df = pd.read_csv(cfg["path"], low_memory=False)

    label_col = _find_column(df, cfg["label_col_aliases"])
    if label_col is None:
        raise ValueError(
            f"Could not find a label/target column in {cfg['path']}. "
            f"Found columns: {list(df.columns)}. "
            f"Add the exact column name to label_col_aliases in backend/common/config.py."
        )

    drop_cols = [c for c in cfg.get("drop_cols", []) if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    positives = set(str(v).lower() for v in cfg["positive_values"])
    y = df[label_col].apply(
        lambda v: 1 if str(v).strip().lower() in positives else 0
    ).values.astype(int)

    X = df.drop(columns=[label_col])

    # One-hot encode any remaining categorical/string columns (e.g. transaction "type")
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=False)

    # Coerce everything numeric, fill leftover NaNs
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    return X, y, list(X.columns)