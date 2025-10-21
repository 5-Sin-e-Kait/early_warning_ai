BIN2RANK = {
    "10%이하": 0.05, "10-25%": 0.175, "25-50%": 0.375,
    "50-75%": 0.625, "75-90%": 0.875, "90%초과": 0.95,
    "10% 이하": 0.05, "10 ~ 25%": 0.175, "25 ~ 50%": 0.375,
    "50 ~ 75%": 0.625, "75 ~ 90%": 0.875, "90% 초과": 0.95,
}

ALPHA = 0.4
BETA = 0.3
GAMMA = 0.3

ENSEMBLE_WEIGHTS = {
    "xgb": 0.25,
    "lgbm": 0.25,
    "rf": 0.25,
    "gb": 0.15,
    "dl": 0.10,
}

CALIBRATION = "platt"
LAMBDA_BLEND = 0.6

THRESHOLDS = {
    "yellow": 0.20,
    "orange": 0.30,
    "red": 0.40,
    "delta": 0.05,
    "persistence_k": 3,
}

VERY_NEGATIVE_SV = -9e5
EPS = 1e-6
ROLL_WINDOW = 12
MIN_PERIODS = 6
