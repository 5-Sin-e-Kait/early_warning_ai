import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.dummy import DummyClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available, DL model will be skipped")

KEY_MCT, KEY_YM = "ENCODED_MCT", "TA_YM"


def to_month(s):
    dt = pd.to_datetime(s.astype(str), errors="coerce")
    return pd.to_datetime(dt.dt.to_period("M").astype(str))


def build_labels_robust(ds1, ds2, ds3, k_months=3):
    df = ds2.merge(ds3, on=[KEY_MCT, KEY_YM], how="outer")
    df[KEY_YM] = to_month(df[KEY_YM])
    df[KEY_MCT] = df[KEY_MCT].astype(str)
    df = df.sort_values([KEY_MCT, KEY_YM]).reset_index(drop=True)
    df["y"] = 0

    if "MCT_ME_D" in ds1.columns:
        tmp = ds1[[KEY_MCT, "MCT_ME_D"]].copy()
        tmp[KEY_MCT] = tmp[KEY_MCT].astype(str)
        tmp["MCT_ME_D"] = pd.to_datetime(
            tmp["MCT_ME_D"].dropna().astype(int).astype(str),
            format='%Y%m%d',
            errors="coerce"
        )

        valid_cnt = tmp["MCT_ME_D"].notna().sum()
        print(f"Valid closure dates: {valid_cnt}")

        if valid_cnt > 0:
            df = df.merge(tmp, on=KEY_MCT, how="left")
            t0 = df[KEY_YM]
            tK = t0 + pd.offsets.MonthEnd(0) + pd.DateOffset(months=k_months)
            cond = (df["MCT_ME_D"].notna()) & (df["MCT_ME_D"] > t0) & (df["MCT_ME_D"] <= tK)
            df.loc[cond, "y"] = 1
            print(f"Labels from actual closures: {df['y'].sum()}")

    def bin2num(s):
        s = s.astype(str).str.strip()
        m = s.str.extract(r"(\d+)", expand=False)
        return pd.to_numeric(m, errors="coerce")

    df["RC_SAA_num"] = bin2num(df.get("RC_M1_SAA", ""))
    df["RC_CUS_num"] = bin2num(df.get("RC_M1_UE_CUS_CN", ""))
    df["dSAA"] = df.groupby(KEY_MCT)["RC_SAA_num"].diff()
    df["dCUS"] = df.groupby(KEY_MCT)["RC_CUS_num"].diff()

    cxl = pd.to_numeric(df.get("APV_CE_RAT", 0), errors="coerce")
    indme = pd.to_numeric(df.get("M12_SME_RY_ME_MCT_RAT", 0), errors="coerce")
    bznme = pd.to_numeric(df.get("M12_SME_BZN_ME_MCT_RAT", 0), errors="coerce")

    sig_score = (
            (df["dSAA"] <= -5).astype(int) * 2 +
            (df["dCUS"] <= -5).astype(int) * 2 +
            (cxl >= 70).astype(int) +
            (indme >= 60).astype(int) +
            (bznme >= 60).astype(int)
    )

    proxy_cond = (sig_score >= 3) & (df["y"] == 0)
    df.loc[proxy_cond, "y"] = 1
    print(f"Additional labels from proxy: {proxy_cond.sum()}")

    print(f"Hybrid labels: {df['y'].value_counts().to_dict()}")

    if df["y"].sum() < 100:
        proxy_cond_relaxed = (sig_score >= 2) & (df["y"] == 0)
        df.loc[proxy_cond_relaxed, "y"] = 1
        print(f"Relaxed to: {df['y'].value_counts().to_dict()}")

    if df["y"].nunique() < 2:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from pipeline import run_pipeline
        out = run_pipeline(ds1, ds2, ds3, preds=None)
        outj = out.merge(df[[KEY_MCT, KEY_YM]], on=[KEY_MCT, KEY_YM], how="right")
        pf = pd.to_numeric(outj["p_final"], errors="coerce").fillna(0)
        thr = pf.quantile(0.85)
        df["y"] = (pf >= thr).astype(int)

    return df


def safe_proba(pipe, X, pos_label=1):
    n = len(X)
    if pipe is None:
        return np.zeros(n)

    clf = getattr(pipe, "named_steps", {}).get("clf", None)

    if hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)
            if proba.ndim == 1:
                return proba.astype(float)
            if proba.shape[1] > 1:
                classes_ = getattr(clf, "classes_", getattr(pipe, "classes_", None))
                if classes_ is not None and pos_label in list(classes_):
                    idx = int(np.where(classes_ == pos_label)[0][0])
                else:
                    idx = 1
                return proba[:, idx]
            else:
                classes_ = getattr(clf, "classes_", [0])
                return np.ones(n) if (len(classes_) == 1 and classes_[0] == pos_label) else np.zeros(n)
        except Exception:
            pass

    if hasattr(pipe, "decision_function"):
        try:
            s = pipe.decision_function(X)
            return 1.0 / (1.0 + np.exp(-s))
        except Exception:
            pass

    try:
        pred = pipe.predict(X)
        return pred.astype(float)
    except Exception:
        return np.zeros(n)


def train_and_predict(ds1, ds2, ds3, output_path="data/preds.csv"):
    print("=" * 60)
    print("STEP 1: Label Generation")
    print("=" * 60)
    robust_df = build_labels_robust(ds1, ds2, ds3, k_months=3)

    print(f"\nTotal: {len(robust_df):,}, Positive: {robust_df['y'].sum():,} ({robust_df['y'].mean():.2%})")

    if robust_df["y"].nunique() < 2:
        print("Single-class labels - using IsolationForest fallback")
        num_cols = ["M1_SME_RY_SAA_RAT", "M1_SME_RY_CNT_RAT"]
        X = robust_df[[c for c in num_cols if c in robust_df.columns]].fillna(0)

        from sklearn.preprocessing import StandardScaler
        pipe_if = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", IsolationForest(n_estimators=400, contamination=0.10, random_state=42))
        ])
        pipe_if.fit(X)
        s = pipe_if["clf"].score_samples(pipe_if["scaler"].transform(X))
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        pe = 1 - s

        preds_df = robust_df[[KEY_MCT, KEY_YM]].copy()
        preds_df[KEY_MCT] = preds_df[KEY_MCT].astype(str)
        preds_df[KEY_YM] = pd.to_datetime(preds_df[KEY_YM], errors="coerce").dt.to_period("M").dt.to_timestamp()
        preds_df["pred_xgb"] = pe
        preds_df["pred_lgbm"] = pe
        preds_df["pred_rf"] = pe
        preds_df["pred_gb"] = pe
        preds_df["pred_dl"] = pe
        preds_df = preds_df.dropna(subset=[KEY_MCT, KEY_YM])

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        preds_df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\n✓ Saved (IsolationForest): {output_path}")
        return preds_df

    print("\n" + "=" * 60)
    print("STEP 2: Feature Engineering")
    print("=" * 60)

    num_cols = [
        "M1_SME_RY_SAA_RAT", "M1_SME_RY_CNT_RAT",
        "M12_SME_RY_SAA_PCE_RT", "M12_SME_BZN_SAA_PCE_RT",
        "M12_SME_RY_ME_MCT_RAT", "M12_SME_BZN_ME_MCT_RAT",
        "DLV_SAA_RAT", "MCT_UE_CLN_REU_RAT", "MCT_UE_CLN_NEW_RAT"
    ]
    cat_cols = [c for c in ["HPSN_MCT_ZCD_NM", "HPSN_MCT_BZN_CD_NM"] if c in robust_df.columns]

    X = robust_df[num_cols + cat_cols].copy()
    y = robust_df["y"].astype(int)

    num_transform = Pipeline([("imp", SimpleImputer(strategy="median"))])
    ct = ColumnTransformer([
        ("num", num_transform, num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ], remainder="drop")

    print(f"Features: {len(num_cols)} numeric + {len(cat_cols)} categorical")

    print("\n" + "=" * 60)
    print("STEP 3: Train/Test Split")
    print("=" * 60)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Train: {len(ytr):,}, Test: {len(yte):,}")

    print("\n" + "=" * 60)
    print("STEP 4: Model Training")
    print("=" * 60)

    # RF
    print("\n[1/5] RandomForest...")
    rf = Pipeline([
        ("prep", ct),
        ("clf", RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1, class_weight="balanced"))
    ])
    rf.fit(Xtr, ytr)
    prf = safe_proba(rf, Xte)
    print(f"  Test AUC: {roc_auc_score(yte, prf):.4f}")

    # GB
    print("\n[2/5] GradientBoosting...")
    gb = Pipeline([
        ("prep", ct),
        ("clf", GradientBoostingClassifier(random_state=42))
    ])
    gb.fit(Xtr, ytr)
    pgb = safe_proba(gb, Xte)
    print(f"  Test AUC: {roc_auc_score(yte, pgb):.4f}")

    # XGB
    print("\n[3/5] XGBoost...")
    xgb_clf = Pipeline([
        ("prep", ct),
        ("clf", xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42, tree_method="hist"
        ))
    ])
    xgb_clf.fit(Xtr, ytr)
    pxgb = safe_proba(xgb_clf, Xte)
    print(f"  Test AUC: {roc_auc_score(yte, pxgb):.4f}")

    # LGB
    print("\n[4/5] LightGBM...")
    lgb_clf = Pipeline([
        ("prep", ct),
        ("clf", lgb.LGBMClassifier(
            n_estimators=500, max_depth=-1, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary", random_state=42, verbose=-1
        ))
    ])
    lgb_clf.fit(Xtr, ytr)
    plgb = safe_proba(lgb_clf, Xte)
    print(f"  Test AUC: {roc_auc_score(yte, plgb):.4f}")

    # DL
    print("\n[5/5] Deep Learning...")
    if TF_AVAILABLE:
        import copy
        ct_dl = copy.deepcopy(ct)
        Xd_tr = ct_dl.fit_transform(Xtr)
        Xd_te = ct_dl.transform(Xte)

        inp = keras.Input(shape=(Xd_tr.shape[1],))
        h = layers.Dense(128, activation="relu")(inp)
        h = layers.Dropout(0.2)(h)
        h = layers.Dense(64, activation="relu")(h)
        outp = layers.Dense(1, activation="sigmoid")(h)
        dl_model = keras.Model(inp, outp)
        dl_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
        dl_model.fit(Xd_tr, ytr, epochs=10, batch_size=256, verbose=0)
        pdl = dl_model.predict(Xd_te, verbose=0).ravel()
        print(f"  Test AUC: {roc_auc_score(yte, pdl):.4f}")
    else:
        pdl = np.zeros(len(Xte))
        print("  Skipped (TensorFlow not available)")

    print("\n" + "=" * 60)
    print("STEP 5: Full Data Prediction")
    print("=" * 60)

    prf_full = safe_proba(rf, X)
    pgb_full = safe_proba(gb, X)
    pxgb_full = safe_proba(xgb_clf, X)
    plgb_full = safe_proba(lgb_clf, X)

    if TF_AVAILABLE:
        Xd_full = ct_dl.transform(X)
        pdl_full = dl_model.predict(Xd_full, verbose=0).ravel()
    else:
        pdl_full = np.zeros(len(X))

    preds_df = robust_df[[KEY_MCT, KEY_YM]].copy()
    preds_df[KEY_MCT] = preds_df[KEY_MCT].astype(str)
    preds_df[KEY_YM] = pd.to_datetime(preds_df[KEY_YM], errors="coerce").dt.to_period("M").dt.to_timestamp()
    preds_df["pred_xgb"] = pxgb_full
    preds_df["pred_lgbm"] = plgb_full
    preds_df["pred_rf"] = prf_full
    preds_df["pred_gb"] = pgb_full
    preds_df["pred_dl"] = pdl_full
    preds_df = preds_df.dropna(subset=[KEY_MCT, KEY_YM])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    preds_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n✓ Saved: {output_path}")
    print(f"  Shape: {preds_df.shape}")
    print(f"  Columns: {preds_df.columns.tolist()}")

    return preds_df


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    def read_csv_smart(path):
        for enc in ["utf-8", "cp949", "euc-kr", "latin1"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                pass
        raise RuntimeError(f"Failed to read: {path}")

    ds1 = read_csv_smart(os.path.join(data_dir, "big_data_set1_f.csv"))
    ds2 = read_csv_smart(os.path.join(data_dir, "ds2_monthly_usage.csv"))
    ds3 = read_csv_smart(os.path.join(data_dir, "ds3_monthly_customers.csv"))

    output_path = os.path.join(data_dir, "preds.csv")
    train_and_predict(ds1, ds2, ds3, output_path)

    print("\n" + "=" * 60)
    print("✓ Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
