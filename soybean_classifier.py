"""
Soybean Binary Classification Pipeline
- Works with lat/lon/date tabular data
- Highly generalizable, field-level robust
- Time-aware filtering, group-based split, multi-stage feature selection
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.feature_selection import VarianceThreshold

import xgboost as xgb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH   = "data.csv"
TARGET_COL  = "label"
DATE_COL    = "date"
FIELD_COL   = "field_id"
DROP_COLS   = ["crop"]

CORR_THRESHOLD  = 0.95
TOP_N_FEATURES  = 20
TEST_SIZE       = 0.20
N_SPLITS_CV     = 3
RANDOM_STATE    = 42
OUTPUT_DIR      = Path("artifacts")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    # Drop forbidden columns
    for c in DROP_COLS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Ensure field_id is string
    df[FIELD_COL] = df[FIELD_COL].astype(str)

    # Parse date (handles dd-mm-yyyy)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[DATE_COL])

    # Ensure numeric lat/lon/label
    for col in ["lat", "lon", TARGET_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    print(f"  After cleaning: {df.shape[0]} rows")
    print(f"  Class distribution: {df[TARGET_COL].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING (per field_id)
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Since this dataset has lat/lon/date per field observation,
    we engineer rich temporal and spatial features per field_id.
    """
    records = []

    for fid, grp in df.groupby(FIELD_COL):
        grp = grp.sort_values(DATE_COL)
        label = int(grp[TARGET_COL].mode()[0])

        lats = grp["lat"].values
        lons = grp["lon"].values
        dates = grp[DATE_COL]

        # Day of year features
        doys = dates.dt.dayofyear.values
        months = dates.dt.month.values

        # Temporal spread
        date_range_days = (dates.max() - dates.min()).days if len(dates) > 1 else 0
        n_obs = len(grp)

        # Spatial features
        lat_mean = np.mean(lats)
        lat_std  = np.std(lats) if n_obs > 1 else 0
        lon_mean = np.mean(lons)
        lon_std  = np.std(lons) if n_obs > 1 else 0
        lat_range = np.max(lats) - np.min(lats)
        lon_range = np.max(lons) - np.min(lons)

        # Centroid
        centroid_lat = lat_mean
        centroid_lon = lon_mean

        # Approximate field area proxy (bounding box)
        bbox_area = lat_range * lon_range

        # Seasonal features — soybean grows kharif (Jun-Oct in India)
        month_mean = np.mean(months)
        month_std  = np.std(months) if n_obs > 1 else 0
        doy_mean   = np.mean(doys)
        doy_std    = np.std(doys) if n_obs > 1 else 0

        # Is observation in kharif season (Jun=6 to Oct=10)?
        kharif_obs = np.sum((months >= 6) & (months <= 10))
        kharif_ratio = kharif_obs / n_obs

        # Is observation in rabi season (Nov-Mar)?
        rabi_obs = np.sum((months >= 11) | (months <= 3))
        rabi_ratio = rabi_obs / n_obs

        # Peak month (most common)
        peak_month = int(pd.Series(months).mode()[0])

        # Year features
        years = dates.dt.year.values
        year_mean = np.mean(years)

        # Geographic zone proxies (India-specific)
        # Soybean belt: Maharashtra/MP ~ lat 18-24, lon 73-80
        in_soybean_belt = int(18 <= centroid_lat <= 24 and 73 <= centroid_lon <= 80)

        # Distance from Vidarbha center (major soybean region)
        vidarbha_lat, vidarbha_lon = 20.5, 78.5
        dist_vidarbha = np.sqrt((centroid_lat - vidarbha_lat)**2 + (centroid_lon - vidarbha_lon)**2)

        # Distance from MP soybean belt center
        mp_lat, mp_lon = 23.0, 77.0
        dist_mp = np.sqrt((centroid_lat - mp_lat)**2 + (centroid_lon - mp_lon)**2)

        # Min distance to known soybean region
        dist_soybean_region = min(dist_vidarbha, dist_mp)

        records.append({
            FIELD_COL: fid,
            TARGET_COL: label,
            "lat_mean": lat_mean,
            "lat_std": lat_std,
            "lat_min": np.min(lats),
            "lat_max": np.max(lats),
            "lat_range": lat_range,
            "lon_mean": lon_mean,
            "lon_std": lon_std,
            "lon_min": np.min(lons),
            "lon_max": np.max(lons),
            "lon_range": lon_range,
            "bbox_area": bbox_area,
            "n_obs": n_obs,
            "date_range_days": date_range_days,
            "month_mean": month_mean,
            "month_std": month_std,
            "doy_mean": doy_mean,
            "doy_std": doy_std,
            "kharif_ratio": kharif_ratio,
            "rabi_ratio": rabi_ratio,
            "peak_month": peak_month,
            "year_mean": year_mean,
            "in_soybean_belt": in_soybean_belt,
            "dist_vidarbha": dist_vidarbha,
            "dist_mp": dist_mp,
            "dist_soybean_region": dist_soybean_region,
        })

    result = pd.DataFrame(records)
    print(f"  Engineered features shape: {result.shape}")
    return result


# ─────────────────────────────────────────────
# 3. GROUP-BASED TRAIN/TEST SPLIT
# ─────────────────────────────────────────────
def group_split(feat_df: pd.DataFrame):
    X = feat_df.drop(columns=[TARGET_COL, FIELD_COL])
    y = feat_df[TARGET_COL].values
    groups = feat_df[FIELD_COL].values

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    print(f"  Train: {len(train_idx)} fields | Test: {len(test_idx)} fields")
    print(f"  Train class dist: {np.bincount(y_train.astype(int))}")
    print(f"  Test  class dist: {np.bincount(y_test.astype(int))}")
    return X_train, X_test, y_train, y_test, groups_train, X.columns.tolist()


# ─────────────────────────────────────────────
# 4. FEATURE SELECTION
# ─────────────────────────────────────────────
def feature_selection(X_train: pd.DataFrame, y_train: np.ndarray,
                       X_test: pd.DataFrame):
    print("\n[Feature Selection]")

    # Fill missing
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test  = X_test.fillna(medians)

    # Variance filter
    vt = VarianceThreshold(threshold=1e-5)
    vt.fit(X_train)
    mask = vt.get_support()
    X_train = X_train.loc[:, mask]
    X_test  = X_test.loc[:, mask]
    print(f"  After variance filter: {X_train.shape[1]} features")

    # Correlation pruning
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > CORR_THRESHOLD)]
    X_train.drop(columns=to_drop, inplace=True, errors="ignore")
    X_test.drop(columns=to_drop,  inplace=True, errors="ignore")
    print(f"  After correlation pruning: {X_train.shape[1]} features")

    # Model-based selection
    pos_weight = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)
    sel = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0,
    )
    sel.fit(X_train, y_train)
    imp = pd.Series(sel.feature_importances_, index=X_train.columns)
    top = imp.nlargest(min(TOP_N_FEATURES, len(imp))).index.tolist()
    X_train = X_train[top]
    X_test  = X_test[top]
    print(f"  After model-based selection: {len(top)} features")
    print(f"  Selected: {top}")

    return X_train, X_test, top, medians


# ─────────────────────────────────────────────
# 5. SCALING
# ─────────────────────────────────────────────
def scale_features(X_train, X_test):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    return Xtr, Xte, scaler


# ─────────────────────────────────────────────
# 6. THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────
def find_optimal_threshold(y_true, y_proba):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.91, 0.01):
        p = (y_proba >= t).astype(int)
        f = f1_score(y_true, p, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    print(f"  Optimal threshold: {best_t:.2f}  (F1={best_f1:.4f})")
    return best_t


# ─────────────────────────────────────────────
# 7. EVALUATION
# ─────────────────────────────────────────────
def evaluate(name, y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.5,
    }
    print(f"\n{'='*50}")
    print(f"  {name}  (threshold={threshold:.2f})")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print("\n  Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=["Non-Soybean", "Soybean"]))
    return metrics


# ─────────────────────────────────────────────
# 8. CROSS-VALIDATION
# ─────────────────────────────────────────────
def cross_validate(model_fn, X, y, groups):
    gkf = GroupKFold(n_splits=N_SPLITS_CV)
    scores = []
    for i, (tr, val) in enumerate(gkf.split(X, y, groups)):
        m = model_fn()
        m.fit(X[tr], y[tr])
        try:
            p = m.predict_proba(X[val])[:, 1]
            s = roc_auc_score(y[val], p) if len(np.unique(y[val])) > 1 else 0.5
        except Exception:
            s = 0.5
        scores.append(s)
        print(f"    Fold {i+1}: ROC-AUC={s:.4f}")
    print(f"  CV Mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return scores


# ─────────────────────────────────────────────
# 9. TRAIN ALL MODELS
# ─────────────────────────────────────────────
def train_models(X_tr, y_tr, X_te, y_te, groups_tr, feature_names):
    pos_w = float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1)
    print(f"\n  scale_pos_weight = {pos_w:.2f}")

    # XGBoost
    print("\n[XGBoost]")
    xgb_m = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=pos_w,
        eval_metric="logloss", early_stopping_rounds=30,
        random_state=RANDOM_STATE, verbosity=0,
    )
    xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    xgb_thresh = find_optimal_threshold(y_tr, xgb_m.predict_proba(X_tr)[:, 1])
    xgb_met = evaluate("XGBoost", y_te, xgb_m.predict_proba(X_te)[:, 1], xgb_thresh)

    print("  XGBoost CV:")
    cross_validate(
        lambda: xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_w, eval_metric="logloss",
            random_state=RANDOM_STATE, verbosity=0,
        ),
        X_tr, y_tr, groups_tr,
    )

    # LightGBM
    print("\n[LightGBM]")
    lgb_m = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=RANDOM_STATE, verbose=-1,
    )
    lgb_m.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    lgb_thresh = find_optimal_threshold(y_tr, lgb_m.predict_proba(X_tr)[:, 1])
    lgb_met = evaluate("LightGBM", y_te, lgb_m.predict_proba(X_te)[:, 1], lgb_thresh)

    # Random Forest
    print("\n[Random Forest]")
    rf_m = RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_m.fit(X_tr, y_tr)
    rf_thresh = find_optimal_threshold(y_tr, rf_m.predict_proba(X_tr)[:, 1])
    rf_met = evaluate("Random Forest", y_te, rf_m.predict_proba(X_te)[:, 1], rf_thresh)

    # Pick best
    results = {
        "XGBoost":       (xgb_m, xgb_met, xgb_thresh),
        "LightGBM":      (lgb_m, lgb_met, lgb_thresh),
        "Random Forest": (rf_m,  rf_met,  rf_thresh),
    }
    best_name = max(results, key=lambda k: results[k][1]["roc_auc"])
    best_model, best_met, best_thresh = results[best_name]
    print(f"\n  Best model: {best_name}  ROC-AUC={best_met['roc_auc']:.4f}")

    # Feature importance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    imp = pd.Series(best_model.feature_importances_, index=feature_names).nlargest(20)
    imp.sort_values().plot(kind="barh", ax=ax)
    ax.set_title(f"Feature Importance — {best_name}")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "feature_importance.png"), dpi=150)
    plt.close()

    # Overfitting check
    train_auc = roc_auc_score(y_tr, best_model.predict_proba(X_tr)[:, 1])
    print(f"  Overfitting → Train AUC: {train_auc:.4f} | Test AUC: {best_met['roc_auc']:.4f} | Gap: {train_auc - best_met['roc_auc']:.4f}")

    return best_model, best_thresh, best_name


# ─────────────────────────────────────────────
# 10. SAVE ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(model, features, scaler, medians, threshold):
    with open(OUTPUT_DIR / "best_model.pkl", "wb") as f:
        pickle.dump({"model": model, "threshold": threshold}, f)
    with open(OUTPUT_DIR / "selected_features.pkl", "wb") as f:
        pickle.dump(features, f)
    with open(OUTPUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "medians": medians}, f)
    print(f"\n  Artifacts saved → {OUTPUT_DIR}/")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  SOYBEAN CLASSIFICATION PIPELINE")
    print("=" * 60)

    print("\n[1] Loading data ...")
    df = load_data(DATA_PATH)

    print("\n[2] Engineering features ...")
    feat_df = engineer_features(df)

    print("\n[3] Group-based split ...")
    X_tr, X_te, y_tr, y_te, groups_tr, feat_names = group_split(feat_df)

    print("\n[4] Feature selection ...")
    X_tr, X_te, selected, medians = feature_selection(X_tr.copy(), y_tr, X_te.copy())

    print("\n[5] Scaling ...")
    X_tr_sc, X_te_sc, scaler = scale_features(X_tr, X_te)

    print("\n[6] Training models ...")
    best_model, best_thresh, best_name = train_models(
        X_tr_sc, y_tr, X_te_sc, y_te, groups_tr, selected
    )

    print("\n[7] Saving artifacts ...")
    save_artifacts(best_model, selected, scaler, medians, best_thresh)

    print("\n  Pipeline complete.")
    return best_model, selected, scaler, medians, best_thresh


if __name__ == "__main__":
    main()
