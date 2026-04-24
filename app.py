"""
Soybean Detection API
- 4-model ensemble (XGBoost + LightGBM + RandomForest + GradientBoosting)
- Trains on startup from data.csv
- Input: KML file + date  |  Output: Soybean / Non-Soybean
"""
import warnings, re, pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from xml.etree import ElementTree as ET

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, classification_report, confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb

# ── Paths ──────────────────────────────────────────────────────────────────
ARTIFACTS     = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH    = ARTIFACTS / "best_model.pkl"
FEATURES_PATH = ARTIFACTS / "selected_features.pkl"
SCALER_PATH   = ARTIFACTS / "scaler.pkl"
DATA_PATH     = "data.csv"


# ── Feature engineering (identical for train & inference) ──────────────────
def make_features(lat: float, lon: float, obs_date: datetime) -> dict:
    month = obs_date.month
    doy   = obs_date.timetuple().tm_yday
    year  = obs_date.year

    # Soybean cluster centroid from training data
    SOY_LAT_C, SOY_LON_C = 19.183, 77.061
    SOY_LAT_R, SOY_LON_R = 0.025,  0.025

    dist_soy_cluster = float(np.sqrt(
        ((lat - SOY_LAT_C) / SOY_LAT_R) ** 2 +
        ((lon - SOY_LON_C) / SOY_LON_R) ** 2
    ))
    in_soy_cluster = int(dist_soy_cluster <= 2.0)

    dist_vidarbha   = float(np.sqrt((lat - 20.5)**2 + (lon - 78.5)**2))
    dist_mp         = float(np.sqrt((lat - 23.0)**2 + (lon - 77.0)**2))
    dist_marathwada = float(np.sqrt((lat - 18.5)**2 + (lon - 76.5)**2))
    dist_nearest    = min(dist_vidarbha, dist_mp, dist_marathwada)

    is_kharif_harvest = int(month in [9, 10])
    is_kharif_season  = int(6 <= month <= 10)
    is_rabi_season    = int(month >= 11 or month <= 3)

    month_sin = float(np.sin(2 * np.pi * month / 12))
    month_cos = float(np.cos(2 * np.pi * month / 12))
    doy_sin   = float(np.sin(2 * np.pi * doy / 365))
    doy_cos   = float(np.cos(2 * np.pi * doy / 365))

    in_maha_belt = int(17.5 <= lat <= 21.5 and 73.5 <= lon <= 80.5)
    in_mp_belt   = int(21.5 <= lat <= 25.5 and 74.0 <= lon <= 82.0)
    in_soy_belt  = int(in_maha_belt or in_mp_belt)

    lat_norm = (lat - 19.183) / 2.0
    lon_norm = (lon - 77.061) / 2.0

    return {
        "lat": lat, "lon": lon,
        "lat_norm": lat_norm, "lon_norm": lon_norm,
        "lat_sq": lat**2, "lon_sq": lon**2, "lat_lon": lat * lon,
        "dist_soy_cluster": dist_soy_cluster,
        "in_soy_cluster": in_soy_cluster,
        "dist_vidarbha": dist_vidarbha,
        "dist_mp": dist_mp,
        "dist_marathwada": dist_marathwada,
        "dist_nearest_soy": dist_nearest,
        "in_maha_belt": in_maha_belt,
        "in_mp_belt": in_mp_belt,
        "in_soy_belt": in_soy_belt,
        "month": month, "doy": doy, "year": year,
        "month_sin": month_sin, "month_cos": month_cos,
        "doy_sin": doy_sin, "doy_cos": doy_cos,
        "is_kharif_harvest": is_kharif_harvest,
        "is_kharif_season": is_kharif_season,
        "is_rabi_season": is_rabi_season,
        "geo_x_kharif": in_soy_belt * is_kharif_harvest,
        "cluster_x_kharif": in_soy_cluster * is_kharif_harvest,
        "dist_cluster_x_month": dist_soy_cluster * month,
        "belt_x_doy": in_soy_belt * doy,
    }


# ── Training ───────────────────────────────────────────────────────────────
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["field_id"] = df["field_id"].astype(str)
    if "crop" in df.columns:
        df.drop(columns=["crop"], inplace=True)
    df["date"]  = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["lat"]   = pd.to_numeric(df["lat"],   errors="coerce")
    df["lon"]   = pd.to_numeric(df["lon"],   errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["date", "lat", "lon", "label"])
    df["label"] = df["label"].astype(int)
    return df


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        feat = make_features(row["lat"], row["lon"], row["date"])
        feat["field_id"] = row["field_id"]
        feat["label"]    = row["label"]
        rows.append(feat)
    return pd.DataFrame(rows)


def find_threshold(y_true, y_proba) -> float:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.96, 0.01):
        f = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t)


def train_and_save():
    print("\n" + "="*60)
    print("  TRAINING SOYBEAN DETECTION MODEL")
    print("="*60)

    df = load_raw()
    print(f"  Rows: {len(df)} | Soybean: {(df.label==1).sum()} | Non-soy: {(df.label==0).sum()}")

    feat_df = build_dataset(df)
    X = feat_df.drop(columns=["label", "field_id"])
    y = feat_df["label"].values
    groups = feat_df["field_id"].values
    feature_names = X.columns.tolist()

    medians = X.median()
    X = X.fillna(medians)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Held-out split for evaluation
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, te_idx = next(gss.split(X_sc, y, groups))
    X_tr, X_te = X_sc[tr_idx], X_sc[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    pos_w = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)

    # Train 4 models on split
    xgb_m = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=8,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.05, reg_lambda=0.5, scale_pos_weight=pos_w,
        eval_metric="logloss", early_stopping_rounds=40,
        random_state=42, verbosity=0,
    )
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    lgb_m = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=8,
        num_leaves=63, subsample=0.85, colsample_bytree=0.85,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    lgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])

    rf_m = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf_m.fit(X_tr, y_tr)

    gb_m = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.85, random_state=42,
    )
    gb_m.fit(X_tr, y_tr)

    # Evaluate
    print("\n  ── Held-out evaluation ──")
    all_probas = []
    for name, m in [("XGBoost", xgb_m), ("LightGBM", lgb_m),
                    ("RandomForest", rf_m), ("GradientBoosting", gb_m)]:
        p = m.predict_proba(X_te)[:, 1]
        all_probas.append(p)
        try:
            auc = roc_auc_score(y_te, p) if len(np.unique(y_te)) > 1 else 0.5
        except Exception:
            auc = 0.5
        acc = accuracy_score(y_te, (p >= 0.5).astype(int))
        print(f"    {name:22s}  AUC={auc:.4f}  Acc={acc:.4f}")

    ens_proba = np.mean(all_probas, axis=0)
    try:
        ens_auc = roc_auc_score(y_te, ens_proba) if len(np.unique(y_te)) > 1 else 0.5
    except Exception:
        ens_auc = 0.5
    threshold = find_threshold(y_te, ens_proba)
    ens_pred  = (ens_proba >= threshold).astype(int)

    print(f"    {'Ensemble':22s}  AUC={ens_auc:.4f}  Acc={accuracy_score(y_te, ens_pred):.4f}")
    print(f"\n  Optimal threshold: {threshold:.2f}")
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_te, ens_pred))
    print(classification_report(y_te, ens_pred, target_names=["Non-Soybean", "Soybean"]))

    # Retrain on FULL data
    print("  Retraining on full dataset ...")
    pos_w_full = float((y == 0).sum()) / max((y == 1).sum(), 1)

    final_models = {}

    m = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=8,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.05, reg_lambda=0.5, scale_pos_weight=pos_w_full,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    m.fit(X_sc, y)
    final_models["xgb"] = m

    m = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=8,
        num_leaves=63, subsample=0.85, colsample_bytree=0.85,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    m.fit(X_sc, y, callbacks=[lgb.log_evaluation(-1)])
    final_models["lgb"] = m

    m = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    m.fit(X_sc, y)
    final_models["rf"] = m

    m = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.85, random_state=42,
    )
    m.fit(X_sc, y)
    final_models["gb"] = m

    with open(MODEL_PATH,    "wb") as f:
        pickle.dump({"models": final_models, "threshold": threshold}, f)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_names, f)
    with open(SCALER_PATH,   "wb") as f:
        pickle.dump({"scaler": scaler, "medians": medians}, f)

    print("  Artifacts saved.\n" + "="*60)


# ── Inference helpers ──────────────────────────────────────────────────────
def load_artifacts():
    with open(MODEL_PATH,    "rb") as f: obj    = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f: feats  = pickle.load(f)
    with open(SCALER_PATH,   "rb") as f: sc_obj = pickle.load(f)
    return obj["models"], obj["threshold"], feats, sc_obj["scaler"], sc_obj["medians"]


def ensemble_predict(models: dict, X_sc: np.ndarray) -> np.ndarray:
    return np.mean([m.predict_proba(X_sc)[:, 1] for m in models.values()], axis=0)


# ── KML parser ─────────────────────────────────────────────────────────────
def parse_kml(content: bytes) -> list:
    coords_list = []

    def _extract(root):
        for elem in root.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag == "coordinates" and elem.text:
                for token in elem.text.strip().split():
                    parts = token.split(",")
                    if len(parts) >= 2:
                        try:
                            coords_list.append((float(parts[1]), float(parts[0])))
                        except ValueError:
                            pass

    try:
        _extract(ET.fromstring(content))
    except ET.ParseError:
        clean = re.sub(rb'\s+xmlns[^=]*="[^"]*"', b"", content)
        try:
            _extract(ET.fromstring(clean))
        except ET.ParseError as e:
            raise HTTPException(400, f"Cannot parse KML: {e}")

    if not coords_list:
        raise HTTPException(400, "No coordinates found in KML.")
    return coords_list


def parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    raise HTTPException(400, f"Bad date format: '{s}'. Use YYYY-MM-DD.")


# ── FastAPI ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Soybean Detection API",
    description="Upload KML + date → Soybean or Non-Soybean prediction.",
    version="2.0.0",
)


@app.on_event("startup")
def startup():
    if not MODEL_PATH.exists():
        print("No model found — training now ...")
        train_and_save()
    else:
        print("Model ready.")


@app.get("/", tags=["Health"])
def root():
    return {"status": "running", "docs": "/docs", "predict": "POST /predict"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/predict", tags=["Prediction"])
async def predict(
    file: UploadFile = File(..., description="KML file of the field"),
    date: str        = Form(..., description="Observation date YYYY-MM-DD"),
):
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model not ready yet.")

    obs_date = parse_date(date)
    content  = await file.read()
    coords   = parse_kml(content)

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    clat = float(np.mean(lats))
    clon = float(np.mean(lons))

    models, threshold, feat_names, scaler, medians = load_artifacts()

    feat_dict = make_features(clat, clon, obs_date)
    feat_df   = pd.DataFrame([feat_dict])
    for f in feat_names:
        if f not in feat_df.columns:
            feat_df[f] = float(medians.get(f, 0))
    feat_df = feat_df[feat_names].fillna(0)

    X_sc  = scaler.transform(feat_df)
    proba = float(ensemble_predict(models, X_sc)[0])
    pred  = int(proba >= threshold)

    conf = "Very High" if proba >= 0.85 else "High" if proba >= 0.65 else "Medium" if proba >= 0.45 else "Low"

    return JSONResponse({
        "prediction":           "Soybean" if pred else "Non-Soybean",
        "is_soybean":           bool(pred),
        "soybean_probability":  round(proba, 4),
        "confidence_level":     conf,
        "threshold_used":       round(threshold, 4),
        "field_centroid":       {"lat": round(clat, 6), "lon": round(clon, 6)},
        "observation_date":     obs_date.strftime("%Y-%m-%d"),
        "n_coordinates":        len(coords),
    })


@app.post("/retrain", tags=["Admin"])
def retrain():
    try:
        train_and_save()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(500, str(e))
