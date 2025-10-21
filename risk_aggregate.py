import pandas as pd
from config import ALPHA, BETA, GAMMA
from risk_components import compute_sales_risk, compute_customer_risk, compute_market_risk


def compute_all_risks(df: pd.DataFrame) -> pd.DataFrame:
    s = compute_sales_risk(df)
    c = compute_customer_risk(df)
    m = compute_market_risk(df)

    out = s.merge(c, on=["ENCODED_MCT", "TA_YM"], how="left")
    out = out.merge(m, on=["ENCODED_MCT", "TA_YM"], how="left")

    out["RiskScore"] = (
            ALPHA * out["Sales_Risk"].fillna(0) +
            BETA * out["Customer_Risk"].fillna(0) +
            GAMMA * out["Market_Risk"].fillna(0)
    )

    return out
