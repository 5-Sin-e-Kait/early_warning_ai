import re
import numpy as np
import pandas as pd
from typing import Iterable
from config import EPS, VERY_NEGATIVE_SV, BIN2RANK


def parse_bin_string(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s_norm = (
        s.str.replace(r"\s*", "", regex=True)
        .str.replace("~", "-", regex=False)
        .str.replace("이하", "이하", regex=False)
        .str.replace("미만", "이하", regex=False)
        .str.replace("초과", "초과", regex=False)
    )

    m = s_norm.map(BIN2RANK)

    mask_range = m.isna() & s_norm.str.contains(r"^\d{1,3}\-?\d{1,3}%$")
    if mask_range.any():
        sub = s_norm[mask_range].str.replace("%", "", regex=False)
        lo = sub.str.extract(r"^(\d{1,3})\-?")[0].astype(float)
        hi = sub.str.extract(r"\-(\d{1,3})$")[0].astype(float)
        mid = ((lo + hi) / 200.0).clip(0, 1)
        m.loc[mask_range] = mid

    mask_hi = m.isna() & s_norm.str.contains(r"^\d{1,3}%초과$")
    if mask_hi.any():
        val = s_norm[mask_hi].str.replace("%초과", "", regex=False).astype(float) / 100.0
        m.loc[mask_hi] = np.minimum(0.95, np.maximum(0.90, val + 0.05))

    mask_lo = m.isna() & s_norm.str.contains(r"^\d{1,3}%이하$")
    if mask_lo.any():
        val = s_norm[mask_lo].str.replace("%이하", "", regex=False).astype(float) / 100.0
        m.loc[mask_lo] = np.maximum(0.05, np.minimum(0.10, val - 0.05))

    m = m.fillna(pd.to_numeric(s, errors="coerce"))

    return m.astype(float)


def map_bin_to_rank(s: pd.Series) -> pd.Series:
    return parse_bin_string(s)


def coerce_month_col(df: pd.DataFrame, ym_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[ym_col].astype(str), errors="coerce")
    dt = pd.to_datetime(dt.dt.to_period("M").astype(str))
    df[ym_col] = dt
    return df


def as_month_sorted(df: pd.DataFrame, ym_col: str):
    return coerce_month_col(df, ym_col).sort_values([ym_col])


def nz(x):
    if isinstance(x, pd.Series):
        return x.astype("float64").clip(lower=0.0, upper=1.0)
    arr = np.asarray(x, dtype="float64")
    return np.minimum(1.0, np.maximum(0.0, arr))


def relu_minus(x):
    if isinstance(x, pd.Series):
        return (-x).clip(lower=0.0)
    return np.maximum(0.0, -x)


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def to_percent(x):
    if isinstance(x, pd.Series):
        return (x.astype("float64") / 100.0).clip(lower=0.0, upper=1.0)
    arr = np.asarray(x, dtype="float64") / 100.0
    return np.minimum(1.0, np.maximum(0.0, arr))


def safe_nan(x):
    x = pd.to_numeric(x, errors="coerce")
    x[(x <= VERY_NEGATIVE_SV)] = np.nan
    return x


def map_bin_to_rank(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    return s.map(BIN2RANK).fillna(pd.to_numeric(s, errors="coerce")).astype(float)


def group_roll_median(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    return x.rolling(window=window, min_periods=min_periods).median()


def group_roll_mad(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    med = group_roll_median(x, window, min_periods)
    mad = (x - med).abs().rolling(window=window, min_periods=min_periods).median()
    return 1.4826 * mad


def robust_z(x: pd.Series, window: int, min_periods: int) -> pd.Series:
    med = group_roll_median(x, window, min_periods)
    mad = group_roll_mad(x, window, min_periods)
    return (x - med) / (mad + EPS)


def mom(x: pd.Series) -> pd.Series:
    return x - x.shift(1)


def hhi(weights: Iterable) -> float:
    w = np.asarray(weights, dtype="float64")
    s = w / (w.sum() + EPS)
    return float(np.sum(s * s))


def as_month_sorted(df: pd.DataFrame, ym_col: str):
    df = df.copy()
    df[ym_col] = pd.to_datetime(df[ym_col].astype(str), format="%Y%m")
    return df.sort_values([ym_col])
