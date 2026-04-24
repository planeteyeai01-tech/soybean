"""
Inference script — run saved model on new unseen fields.
Usage: python predict.py --input new_data.csv --output predictions.csv
"""

import argparse
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path

from soybean_classifier import (
    load_data, time_aware_filter, aggregate_features,
    DATE_COL, FIELD_COL, TARGET_COL, OUTPUT_DIR
)

ARTIFACTS = Path("artifacts")


def load_artifacts():
    with open(ARTIFACTS / "best_model.pkl", "rb") as f:
        obj = pickle.load(f)
    with open(ARTIFACTS / "selected_features.pkl", "rb") as f:
        features = pickle.load(f)
    with open(ARTIFACTS / "scaler.pkl", "rb") as f:
        sc_obj = pickle.load(f)
    return obj["model"], obj["threshold"], features, sc_obj["scaler"], sc_obj["medians"]


def predict(input_path: str, output_path: str):
    model, threshold, features, scaler, medians = load_artifacts()

    df = load_data(input_path)
    df = time_aware_filter(df)

    # Aggregate (drop target if present)
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])
    # Temporarily add dummy target so aggregate_features works
    df[TARGET_COL] = 0
    agg_df = aggregate_features(df)

    field_ids = agg_df[FIELD_COL].values
    X = agg_df.drop(columns=[TARGET_COL, FIELD_COL])

    # Fill missing, select features, scale
    X = X.fillna(medians)
    # Keep only known features, fill any missing ones with median
    for f in features:
        if f not in X.columns:
            X[f] = medians.get(f, 0)
    X = X[features]

    X_sc = scaler.transform(X)
    proba = model.predict_proba(X_sc)[:, 1]
    preds = (proba >= threshold).astype(int)

    out = pd.DataFrame({
        FIELD_COL: field_ids,
        "soybean_probability": proba,
        "soybean_label": preds,
    })
    out.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(out.head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="new_data.csv")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    predict(args.input, args.output)
