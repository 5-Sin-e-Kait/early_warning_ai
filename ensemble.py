import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from config import ENSEMBLE_WEIGHTS, CALIBRATION
from utils import logistic, nz


@dataclass
class PlattScaler:
    a: float = 1.0
    b: float = 0.0
    fitted: bool = False

    def fit(self, p: np.ndarray, y: np.ndarray):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        X = np.column_stack([np.ones_like(p), np.log(p / (1 - p))])
        w = np.linalg.lstsq(X, y, rcond=None)[0]
        self.b, self.a = float(w[0]), float(w[1])
        self.fitted = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        z = self.a * np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))) + self.b
        return nz(logistic(z))


def weighted_ensemble(df_pred: pd.DataFrame) -> pd.Series:
    cols = {
        "xgb": [c for c in df_pred.columns if "pred_xgb" in c.lower()],
        "lgbm": [c for c in df_pred.columns if "pred_lgbm" in c.lower()],
        "rf": [c for c in df_pred.columns if "pred_rf" in c.lower()],
        "gb": [c for c in df_pred.columns if "pred_gb" in c.lower() or "pred_gbdt" in c.lower()],
        "dl": [c for c in df_pred.columns if "pred_dl" in c.lower() or "pred_nn" in c.lower()],
    }
    s = 0.0
    for k, ws in ENSEMBLE_WEIGHTS.items():
        if cols[k]:
            s = s + ws * df_pred[cols[k][0]].astype(float)
    return nz(s)


class Calibrator:
    def __init__(self, method: Optional[str] = CALIBRATION):
        self.method = method
        self.scaler = PlattScaler() if method == "platt" else None

    def fit(self, p: np.ndarray, y: np.ndarray):
        if self.scaler is not None:
            self.scaler.fit(p, y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.scaler is not None and self.scaler.fitted:
            return self.scaler.transform(p)
        return nz(p)
