import numpy as np
import pandas as pd
from config import THRESHOLDS


def rolling_mean(x: pd.Series, k: int) -> pd.Series:
    return x.rolling(window=k, min_periods=1).mean()


def assign_alert(prob: pd.Series) -> pd.Series:
    t_y, t_o, t_r, delta = THRESHOLDS["yellow"], THRESHOLDS["orange"], THRESHOLDS["red"], THRESHOLDS["delta"]
    k = THRESHOLDS["persistence_k"]
    pbar = rolling_mean(prob, k)
    lbl = np.where((prob >= t_r) & (pbar >= t_r - delta), "RED",
                   np.where(prob >= t_o, "ORANGE",
                            np.where(prob >= t_y, "YELLOW", "GREEN")))
    return pd.Series(lbl, index=prob.index)


def _label_series(s: pd.Series, q_y=0.80, q_o=0.90, q_r=0.97) -> pd.Series:
    y = s.quantile(q_y)
    o = s.quantile(q_o)
    r = s.quantile(q_r)

    def lab(v):
        if v >= r: return "RED"
        if v >= o: return "ORANGE"
        if v >= y: return "YELLOW"
        return "GREEN"

    return s.apply(lab)


def assign_alert_by_quantile(df: pd.DataFrame,
                             group_cols=None,
                             score_col="p_final",
                             q_y=0.80, q_o=0.90, q_r=0.97) -> pd.Series:
    if score_col not in df.columns:
        raise KeyError(f"score_col '{score_col}' not in DataFrame")

    cols = []
    if group_cols:
        cols = [c for c in group_cols if c in df.columns]

    if not cols:
        fallback_priority = [
            ["HPSN_MCT_ZCD_NM", "HPSN_MCT_BZN_CD_NM", "TA_YM"],
            ["HPSN_MCT_ZCD_NM", "TA_YM"],
            ["HPSN_MCT_BZN_CD_NM", "TA_YM"],
            ["TA_YM"],
            []
        ]
        for cand in fallback_priority:
            ok = [c for c in cand if c in df.columns]
            if len(ok) == len(cand):
                cols = ok
                break

    if cols:
        return df.groupby(cols, group_keys=False)[score_col].apply(lambda s: _label_series(s, q_y, q_o, q_r))
    else:
        return _label_series(df[score_col], q_y, q_o, q_r)
