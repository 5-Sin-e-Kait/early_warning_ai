import pandas as pd
from typing import Optional
from preprocessing import load_and_join, normalize_bins
from risk_aggregate import compute_all_risks
from ensemble import weighted_ensemble, Calibrator
from alerting import assign_alert_by_quantile
from config import LAMBDA_BLEND


def _coerce_month_col(s):
    dt = pd.to_datetime(s.astype(str), errors="coerce")
    return pd.to_datetime(dt.dt.to_period("M").astype(str))


def _coerce_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ENCODED_MCT" in out.columns:
        out["ENCODED_MCT"] = out["ENCODED_MCT"].astype(str)
    if "TA_YM" in out.columns:
        out["TA_YM"] = _coerce_month_col(out["TA_YM"])
    return out


def run_pipeline(ds1: pd.DataFrame, ds2: pd.DataFrame, ds3: pd.DataFrame,
                 preds: Optional[pd.DataFrame] = None,
                 calib_fit_y: Optional[pd.Series] = None) -> pd.DataFrame:
    df = load_and_join(ds1, ds2, ds3)
    df = normalize_bins(df)
    df = _coerce_keys(df)

    risks = compute_all_risks(df)
    risks = _coerce_keys(risks)

    if preds is None:
        risks["p_model"] = 0.0
        p_cal = risks["p_model"]
    else:
        p = _coerce_keys(preds.copy())
        risks = risks.merge(p, on=["ENCODED_MCT", "TA_YM"], how="left")
        risks["p_model"] = weighted_ensemble(risks)
        cal = Calibrator()
        if calib_fit_y is not None:
            cal.fit(risks["p_model"].values, calib_fit_y.values)
        p_cal = cal.transform(risks["p_model"].values)
        risks["p_model_cal"] = p_cal

    risks["p_final"] = (LAMBDA_BLEND * risks.get("p_model_cal", risks["p_model"]).fillna(0)
                        + (1.0 - LAMBDA_BLEND) * risks["RiskScore"].fillna(0))
    risks["Alert"] = assign_alert_by_quantile(
        risks,
        group_cols=["HPSN_MCT_ZCD_NM"],
        score_col="p_final",
        q_y=0.80, q_o=0.90, q_r=0.97
    )

    cols = ["ENCODED_MCT", "TA_YM", "Sales_Risk", "Customer_Risk", "Market_Risk", "RiskScore", "p_model", "p_final",
            "Alert"]
    return risks[cols]
