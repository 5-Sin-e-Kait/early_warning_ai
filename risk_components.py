import numpy as np
import pandas as pd
from typing import Tuple
from config import ROLL_WINDOW, MIN_PERIODS, EPS
from utils import (
    nz, relu_minus, logistic, to_percent, robust_z, mom
)


def _zero(s: pd.Series) -> pd.Series:
    return s.fillna(0.0).clip(0.0, 1.0)


def compute_sales_risk(df: pd.DataFrame, key=["ENCODED_MCT", "TA_YM"]) -> pd.DataFrame:
    d = df.sort_values(key).copy()
    rS = d["RC_M1_SAA_RANK"]
    rC = d["RC_M1_TO_UE_CT_RANK"]
    rA = d["RC_M1_AV_NP_AT_RANK"]
    rX = d["APV_CE_RAT_RANK"]

    s_drop = _zero(nz((relu_minus(mom(rS)) + relu_minus(mom(rC))) / 2.0))
    s_aov = _zero(nz(relu_minus(robust_z(rA, ROLL_WINDOW, MIN_PERIODS))))
    s_cxl = _zero(nz(np.maximum(rX, logistic(robust_z(rX, ROLL_WINDOW, MIN_PERIODS)))))

    ind_rank = _zero(nz(1.0 - to_percent(d.get("M12_SME_RY_SAA_PCE_RT", pd.Series(50.0, index=d.index)))))
    bzn_rank = _zero(nz(1.0 - to_percent(d.get("M12_SME_BZN_SAA_PCE_RT", pd.Series(50.0, index=d.index)))))
    s_peer = _zero(nz((ind_rank + bzn_rank) / 2.0))

    dlv_raw = d["DLV_SAA_RAT"] if "DLV_SAA_RAT" in d.columns else pd.Series(0.0, index=d.index)
    dlv = _zero(nz(to_percent(dlv_raw)))
    s_dlv_jump = _zero(nz(relu_minus(-mom(dlv)) + relu_minus(-robust_z(dlv, ROLL_WINDOW, MIN_PERIODS))))

    sales_risk = _zero(0.35 * s_drop + 0.15 * s_aov + 0.20 * s_cxl + 0.20 * s_peer + 0.10 * (dlv * s_dlv_jump))
    out = d[["ENCODED_MCT", "TA_YM"]].copy()
    out["Sales_Risk"] = sales_risk
    return out


def compute_customer_risk(df: pd.DataFrame, key=["ENCODED_MCT", "TA_YM"]) -> pd.DataFrame:
    d = df.sort_values(key).copy()
    rU = d["RC_M1_UE_CUS_CN_RANK"]
    s_cus_drop = _zero(nz(relu_minus(mom(rU)) + relu_minus(robust_z(rU, ROLL_WINDOW, MIN_PERIODS))))

    q_reu = _zero(nz(to_percent(d.get("MCT_UE_CLN_REU_RAT", pd.Series(0.0, index=d.index)))))
    q_new = _zero(nz(to_percent(d.get("MCT_UE_CLN_NEW_RAT", pd.Series(0.0, index=d.index)))))
    s_loyal = _zero(nz(relu_minus(mom(q_reu))))
    s_acq = _zero(nz(relu_minus(mom(q_new))))

    age_cols = ["M12_MAL_1020_RAT", "M12_MAL_30_RAT", "M12_MAL_40_RAT", "M12_MAL_50_RAT", "M12_MAL_60_RAT",
                "M12_FME_1020_RAT", "M12_FME_30_RAT", "M12_FME_40_RAT", "M12_FME_50_RAT", "M12_FME_60_RAT"]
    for col in age_cols:
        if col not in d.columns: d[col] = 0.0
    w = _zero(pd.DataFrame({c: d[c] for c in age_cols}).fillna(0) / 100.0).to_numpy(dtype="float64")
    H_age = pd.Series((w * w).sum(axis=1), index=d.index)

    type_cols = ["RC_M1_SHC_RSD_UE_CLN_RAT", "RC_M1_SHC_WP_UE_CLN_RAT", "RC_M1_SHC_FLP_UE_CLN_RAT"]
    for col in type_cols:
        if col not in d.columns: d[col] = 0.0
    v = _zero(pd.DataFrame({c: d[c] for c in type_cols}).fillna(0) / 100.0).to_numpy(dtype="float64")
    H_type = pd.Series((v * v).sum(axis=1), index=d.index)

    def pos_norm(x: pd.Series) -> pd.Series:
        med = x.rolling(ROLL_WINDOW, MIN_PERIODS).median()
        mad = (x - med).abs().rolling(ROLL_WINDOW, MIN_PERIODS).median()
        z = (x - med) / (1.4826 * mad + EPS)
        return _zero(nz(np.maximum(0.0, z)))

    s_mix = pos_norm(H_age)
    s_type = pos_norm(H_type)

    customer_risk = _zero(0.40 * s_cus_drop + 0.25 * s_loyal + 0.20 * s_acq + 0.10 * s_mix + 0.05 * s_type)
    out = d[["ENCODED_MCT", "TA_YM"]].copy()
    out["Customer_Risk"] = customer_risk
    return out


def compute_market_risk(df: pd.DataFrame, key=["ENCODED_MCT", "TA_YM"]) -> pd.DataFrame:
    d = df.sort_values(key).copy()
    h_ind = _zero(nz(to_percent(d.get("M12_SME_RY_ME_MCT_RAT", pd.Series(0.0, index=d.index)))))
    h_bzn = _zero(nz(to_percent(d.get("M12_SME_BZN_ME_MCT_RAT", pd.Series(0.0, index=d.index)))))
    s_closure_env = _zero((h_ind + h_bzn) / 2.0)

    u_rev = _zero(nz(1.0 - to_percent(d.get("M1_SME_RY_SAA_RAT", pd.Series(100.0, index=d.index)))))
    u_cnt = _zero(nz(1.0 - to_percent(d.get("M1_SME_RY_CNT_RAT", pd.Series(100.0, index=d.index)))))
    s_underperf = _zero((u_rev + u_cnt) / 2.0)

    a = d["MCT_OPE_MS_CN_RANK"]
    s_age = _zero(nz(4.0 * np.minimum(a, 1.0 - a)))

    market_risk = _zero(0.50 * s_closure_env + 0.35 * s_underperf + 0.15 * s_age)
    out = d[["ENCODED_MCT", "TA_YM"]].copy()
    out["Market_Risk"] = market_risk
    return out
