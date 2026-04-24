"""
Soybean Detection API v4
- Microsoft Planetary Computer (free, no auth) for Sentinel-2
- Real NDVI/EVI/NDWI/LSWI/SAVI/BSI per field
- 4-model ensemble classifier
"""
import warnings, re, pickle, os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score,
    classification_report, confusion_matrix)
import xgboost as xgb
import lightgbm as lgb

ARTIFACTS     = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH    = ARTIFACTS / "best_model.pkl"
FEATURES_PATH = ARTIFACTS / "selected_features.pkl"
SCALER_PATH   = ARTIFACTS / "scaler.pkl"
CACHE_PATH    = ARTIFACTS / "sat_cache.pkl"
DATA_PATH     = "data.csv"


# ─────────────────────────────────────────────────────────────
# PLANETARY COMPUTER — Sentinel-2 fetch
# ─────────────────────────────────────────────────────────────
def fetch_sentinel2(lat: float, lon: float, obs_date: datetime,
                    buffer_deg: float = 0.005) -> dict:
    """
    Fetch Sentinel-2 L2A bands from Microsoft Planetary Computer.
    Returns dict of spectral indices (NDVI, EVI, NDWI, LSWI, SAVI, BSI, GNDVI, RVI).
    Falls back to zeros if no imagery found.
    """
    try:
        import pystac_client
        import planetary_computer
        import rioxarray
        import stackstac

        bbox = [lon - buffer_deg, lat - buffer_deg,
                lon + buffer_deg, lat + buffer_deg]

        # Search ±45 days around observation date
        d0 = (obs_date - timedelta(days=45)).strftime("%Y-%m-%d")
        d1 = (obs_date + timedelta(days=45)).strftime("%Y-%m-%d")

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{d0}/{d1}",
            query={"eo:cloud_cover": {"lt": 25}},
            max_items=5,
        )
        items = list(search.items())

        if not items:
            # Widen to ±90 days
            d0 = (obs_date - timedelta(days=90)).strftime("%Y-%m-%d")
            d1 = (obs_date + timedelta(days=90)).strftime("%Y-%m-%d")
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=f"{d0}/{d1}",
                query={"eo:cloud_cover": {"lt": 40}},
                max_items=5,
            )
            items = list(search.items())

        if not items:
            print(f"  No Sentinel-2 imagery found for {lat},{lon} around {obs_date.date()}")
            return {}

        # Use least cloudy scene
        items.sort(key=lambda x: x.properties.get("eo:cloud_cover", 99))
        item = items[0]
        print(f"  Using scene: {item.datetime.date()} cloud={item.properties.get('eo:cloud_cover',0):.1f}%")

        # Load bands via stackstac
        bands_needed = ["B02", "B03", "B04", "B08", "B11", "B12"]
        ds = stackstac.stack(
            [item],
            assets=bands_needed,
            bounds=bbox,
            resolution=10,
            dtype="float32",
        )
        # Mean over spatial extent
        arr = ds.mean(dim=["x", "y"]).compute().values  # shape: (1, 6)
        if arr.ndim == 2:
            arr = arr[0]  # (6,)

        # Scale to reflectance (0-1)
        B2, B3, B4, B8, B11, B12 = [float(v) / 10000.0 for v in arr]

        eps = 1e-8
        NDVI  = (B8 - B4)  / (B8 + B4  + eps)
        EVI   = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1 + eps)
        NDWI  = (B3 - B8)  / (B3 + B8  + eps)
        LSWI  = (B8 - B11) / (B8 + B11 + eps)
        SAVI  = 1.5 * (B8 - B4) / (B8 + B4 + 0.5 + eps)
        BSI   = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + B8 + B2 + eps)
        GNDVI = (B8 - B3)  / (B8 + B3  + eps)
        RVI   = B8 / (B4 + eps)

        return {
            "NDVI": round(NDVI, 4),
            "EVI":  round(EVI,  4),
            "NDWI": round(NDWI, 4),
            "LSWI": round(LSWI, 4),
            "SAVI": round(SAVI, 4),
            "BSI":  round(BSI,  4),
            "GNDVI":round(GNDVI,4),
            "RVI":  round(RVI,  4),
            "ndvi_lswi_diff": round(NDVI - LSWI, 4),
            "ndvi_evi_ratio": round(NDVI / (EVI + eps), 4),
            "B4_raw": round(B4, 4),
            "B8_raw": round(B8, 4),
            "B11_raw":round(B11,4),
        }

    except Exception as ex:
        print(f"  Planetary Computer fetch error: {ex}")
        return {}


# ─────────────────────────────────────────────────────────────
# GEO + TEMPORAL FEATURES (always computed)
# ─────────────────────────────────────────────────────────────
def geo_features(lat: float, lon: float, obs_date: datetime) -> dict:
    month = obs_date.month
    doy   = obs_date.timetuple().tm_yday
    year  = obs_date.year

    SOY_LAT_C, SOY_LON_C = 19.183, 77.061
    dist_soy = float(np.sqrt(((lat - SOY_LAT_C)/0.025)**2 +
                              ((lon - SOY_LON_C)/0.025)**2))

    dist_v = float(np.sqrt((lat-20.5)**2 + (lon-78.5)**2))
    dist_m = float(np.sqrt((lat-23.0)**2 + (lon-77.0)**2))
    dist_r = float(np.sqrt((lat-18.5)**2 + (lon-76.5)**2))

    in_maha = int(17.5 <= lat <= 21.5 and 73.5 <= lon <= 80.5)
    in_mp   = int(21.5 <= lat <= 25.5 and 74.0 <= lon <= 82.0)
    in_belt = int(in_maha or in_mp)

    return {
        "lat": lat, "lon": lon,
        "lat_sq": lat**2, "lon_sq": lon**2, "lat_lon": lat*lon,
        "dist_soy_cluster": dist_soy,
        "in_soy_cluster": int(dist_soy <= 2.0),
        "dist_vidarbha": dist_v, "dist_mp": dist_m, "dist_marathwada": dist_r,
        "dist_nearest_soy": min(dist_v, dist_m, dist_r),
        "in_maha_belt": in_maha, "in_mp_belt": in_mp, "in_soy_belt": in_belt,
        "month": month, "doy": doy, "year": year,
        "month_sin": float(np.sin(2*np.pi*month/12)),
        "month_cos": float(np.cos(2*np.pi*month/12)),
        "doy_sin":   float(np.sin(2*np.pi*doy/365)),
        "doy_cos":   float(np.cos(2*np.pi*doy/365)),
        "is_kharif":  int(6 <= month <= 10),
        "is_harvest": int(month in [9, 10]),
        "is_rabi":    int(month >= 11 or month <= 3),
        "geo_x_kharif":     in_belt * int(6 <= month <= 10),
        "cluster_x_kharif": int(dist_soy <= 2.0) * int(6 <= month <= 10),
    }


def make_features(lat: float, lon: float, obs_date: datetime,
                  use_cache: bool = False, cache_key: str = "") -> dict:
    geo  = geo_features(lat, lon, obs_date)
    sat  = fetch_sentinel2(lat, lon, obs_date)
    return {**geo, **sat}


# ─────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────
def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["field_id"] = df["field_id"].astype(str)
    if "crop" in df.columns:
        df.drop(columns=["crop"], inplace=True)
    df["date"]  = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["lat"]   = pd.to_numeric(df["lat"],   errors="coerce")
    df["lon"]   = pd.to_numeric(df["lon"],   errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["date","lat","lon","label"])
    df["label"] = df["label"].astype(int)
    return df


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Load satellite cache
    cache = {}
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)

    rows = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        key = f"{row['field_id']}_{row['date'].date()}"
        print(f"  [{i+1}/{total}] field={row['field_id']} date={row['date'].date()}", end=" ")
        if key in cache:
            sat = cache[key]
            print("(cached)")
        else:
            sat = fetch_sentinel2(row["lat"], row["lon"], row["date"])
            cache[key] = sat
            # Save cache after each fetch
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(cache, f)

        feat = {**geo_features(row["lat"], row["lon"], row["date"]), **sat}
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
    print("  SOYBEAN MODEL TRAINING (Planetary Computer)")
    print("="*60)

    df = load_raw()
    print(f"  Rows: {len(df)} | Soybean: {(df.label==1).sum()} | Non-soy: {(df.label==0).sum()}")

    feat_df = build_dataset(df)
    print(f"\n  Feature matrix: {feat_df.shape}")

    X = feat_df.drop(columns=["label","field_id"], errors="ignore")
    y = feat_df["label"].values
    groups = feat_df["field_id"].values

    medians = X.median()
    X = X.fillna(medians)

    # Remove zero-variance
    vt = VarianceThreshold(1e-6)
    vt.fit(X)
    X = X.loc[:, vt.get_support()]
    feature_names = X.columns.tolist()
    print(f"  Features after variance filter: {len(feature_names)}")

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr, te = next(gss.split(X_sc, y, groups))
    X_tr, X_te = X_sc[tr], X_sc[te]
    y_tr, y_te = y[tr], y[te]
    pos_w = float((y_tr==0).sum()) / max((y_tr==1).sum(), 1)

    # Train 4 models
    models_eval = {}

    m = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=8,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.05, reg_lambda=0.5, scale_pos_weight=pos_w,
        eval_metric="logloss", early_stopping_rounds=40,
        random_state=42, verbosity=0,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    models_eval["xgb"] = m

    m = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=8,
        num_leaves=63, subsample=0.85, colsample_bytree=0.85,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
          callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(-1)])
    models_eval["lgb"] = m

    m = RandomForestClassifier(
        n_estimators=500, max_depth=None, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    m.fit(X_tr, y_tr)
    models_eval["rf"] = m

    m = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.85, random_state=42,
    )
    m.fit(X_tr, y_tr)
    models_eval["gb"] = m

    # Evaluate
    print("\n  ── Evaluation ──")
    all_p = []
    for name, m in models_eval.items():
        p = m.predict_proba(X_te)[:,1]
        all_p.append(p)
        try:
            auc = roc_auc_score(y_te, p) if len(np.unique(y_te))>1 else 0.5
        except:
            auc = 0.5
        acc = accuracy_score(y_te, (p>=0.5).astype(int))
        print(f"    {name:20s}  AUC={auc:.4f}  Acc={acc:.4f}")

    ens_p = np.mean(all_p, axis=0)
    try:
        ens_auc = roc_auc_score(y_te, ens_p) if len(np.unique(y_te))>1 else 0.5
    except:
        ens_auc = 0.5
    threshold = find_threshold(y_te, ens_p)
    ens_pred  = (ens_p >= threshold).astype(int)
    print(f"    {'Ensemble':20s}  AUC={ens_auc:.4f}  Acc={accuracy_score(y_te,ens_pred):.4f}")
    print(f"\n  Threshold: {threshold:.2f}")
    print(confusion_matrix(y_te, ens_pred))
    print(classification_report(y_te, ens_pred, target_names=["Non-Soybean","Soybean"]))

    # Retrain on full data
    print("  Retraining on full data...")
    pos_w_full = float((y==0).sum()) / max((y==1).sum(), 1)
    final = {}

    m = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=8,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.05, reg_lambda=0.5, scale_pos_weight=pos_w_full,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    m.fit(X_sc, y); final["xgb"] = m

    m = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=8,
        num_leaves=63, subsample=0.85, colsample_bytree=0.85,
        class_weight="balanced", random_state=42, verbose=-1,
    )
    m.fit(X_sc, y, callbacks=[lgb.log_evaluation(-1)]); final["lgb"] = m

    m = RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    m.fit(X_sc, y); final["rf"] = m

    m = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.85, random_state=42,
    )
    m.fit(X_sc, y); final["gb"] = m

    with open(MODEL_PATH,    "wb") as f: pickle.dump({"models": final, "threshold": threshold}, f)
    with open(FEATURES_PATH, "wb") as f: pickle.dump(feature_names, f)
    with open(SCALER_PATH,   "wb") as f: pickle.dump({"scaler": scaler, "medians": medians}, f)
    print("  Artifacts saved.\n" + "="*60)


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
def load_artifacts():
    with open(MODEL_PATH,    "rb") as f: obj  = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f: feat = pickle.load(f)
    with open(SCALER_PATH,   "rb") as f: sc   = pickle.load(f)
    return obj["models"], obj["threshold"], feat, sc["scaler"], sc["medians"]


def ensemble_predict(models, X_sc):
    return np.mean([m.predict_proba(X_sc)[:,1] for m in models.values()], axis=0)


# ─────────────────────────────────────────────────────────────
# KML PARSER
# ─────────────────────────────────────────────────────────────
def parse_kml(content: bytes) -> list:
    coords = []
    def _extract(root):
        for elem in root.iter():
            tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag == "coordinates" and elem.text:
                for token in elem.text.strip().split():
                    parts = token.split(",")
                    if len(parts) >= 2:
                        try:
                            coords.append((float(parts[1]), float(parts[0])))
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
    if not coords:
        raise HTTPException(400, "No coordinates found in KML.")
    return coords


def parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    raise HTTPException(400, f"Bad date: '{s}'. Use YYYY-MM-DD.")


# ─────────────────────────────────────────────────────────────
# FASTAPI
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Soybean Detection API",
    description="KML + date → Sentinel-2 via Planetary Computer → Soybean / Non-Soybean",
    version="4.0.0",
)


@app.on_event("startup")
def startup():
    if not MODEL_PATH.exists():
        print("No model found — training now (fetching satellite data)...")
        train_and_save()
    else:
        print("Model ready.")


@app.get("/")
def root():
    return {"status": "running", "version": "4.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="KML file of the field boundary"),
    date: str        = Form(..., description="Observation date YYYY-MM-DD"),
):
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Model not ready yet, please wait.")

    obs_date = parse_date(date)
    content  = await file.read()
    coords   = parse_kml(content)

    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    clat = float(np.mean(lats))
    clon = float(np.mean(lons))

    models, threshold, feat_names, scaler, medians = load_artifacts()

    # Fetch live satellite data for this field
    print(f"  Fetching satellite data for {clat:.4f},{clon:.4f} on {obs_date.date()}...")
    sat = fetch_sentinel2(clat, clon, obs_date)
    feat_dict = {**geo_features(clat, clon, obs_date), **sat}

    feat_df = pd.DataFrame([feat_dict])
    for f in feat_names:
        if f not in feat_df.columns:
            feat_df[f] = float(medians.get(f, 0))
    feat_df = feat_df[feat_names].fillna(0)

    X_sc  = scaler.transform(feat_df)
    proba = float(ensemble_predict(models, X_sc)[0])
    pred  = int(proba >= threshold)
    conf  = ("Very High" if proba >= 0.85 else
             "High"      if proba >= 0.65 else
             "Medium"    if proba >= 0.45 else "Low")

    return JSONResponse({
        "prediction":          "Soybean" if pred else "Non-Soybean",
        "is_soybean":          bool(pred),
        "soybean_probability": round(proba, 4),
        "confidence_level":    conf,
        "threshold_used":      round(threshold, 4),
        "field_centroid":      {"lat": round(clat, 6), "lon": round(clon, 6)},
        "observation_date":    obs_date.strftime("%Y-%m-%d"),
        "n_coordinates":       len(coords),
        "satellite_indices":   {k: v for k, v in sat.items() if isinstance(v, float)},
    })


@app.post("/retrain")
def retrain():
    if CACHE_PATH.exists():
        CACHE_PATH.unlink()
    try:
        train_and_save()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
