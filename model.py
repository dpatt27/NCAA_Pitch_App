"""
model.py
--------
Train and save two model artifacts:
  - model.pkl   : XGBoost classifier predicting whiff probability on swings
  - kmeans.pkl  : KMeans clustering pitchers by arsenal profile

Run once before launching the Streamlit app:
    python model.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from utils import (
    DATA_PATH,
    FEATURES,
    clean,
    get_pitcher_summary,
    load_raw,
)

# ── Config ───────────────────────────────────────────────────────────────────

RANDOM_STATE   = 42
TEST_SIZE      = 0.20
N_CLUSTERS     = 4          # tune if you have more pitchers
MODEL_OUT      = "model.pkl"
SCALER_OUT     = "scaler.pkl"
KMEANS_OUT     = "kmeans.pkl"
KMEANS_SCALER  = "kmeans_scaler.pkl"
CLEAN_DATA_OUT = "clean_data.pkl"

KMEANS_FEATURES = [
    "avg_velo",
    "avg_spin",
    "avg_ivb",
    "avg_hb",
    "avg_extension",
    "n_pitch_types",
    "overall_whiff_rate",
]

# ── Main ─────────────────────────────────────────────────────────────────────

def train_whiff_model(df: pd.DataFrame):
    """Train XGBoost on swing-level data, print eval metrics, return model."""

    X = df[FEATURES]
    y = df["whiff"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Scale features (XGBoost doesn't require it but helps with regularisation)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Class imbalance: weight non-whiffs down
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos = neg / pos

    model = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = scale_pos,
        eval_metric       = "logloss",
        random_state      = RANDOM_STATE,
        n_jobs            = -1,
    )

    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )

    # ── Evaluation ───────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test_s)[:, 1]
    y_pred = model.predict(X_test_s)

    print("=" * 50)
    print("WHIFF MODEL — TEST SET RESULTS")
    print("=" * 50)
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Log-Loss : {log_loss(y_test, y_prob):.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["No Whiff", "Whiff"]))

    # 5-fold CV on full data
    X_all_s = scaler.transform(X)
    cv_scores = cross_val_score(
        model, X_all_s, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        n_jobs=-1,
    )
    print(f"5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("Feature Importance:")
    print(importance.round(4).to_string())
    print()

    return model, scaler


def train_kmeans(df_clean: pd.DataFrame):
    """Cluster pitchers by arsenal profile."""

    summary = get_pitcher_summary(df_clean)
    summary = summary.dropna(subset=KMEANS_FEATURES)

    X = summary[KMEANS_FEATURES].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    summary["cluster"] = kmeans.fit_predict(X_s)

    print("=" * 50)
    print("KMEANS CLUSTERS")
    print("=" * 50)
    for c in sorted(summary["cluster"].unique()):
        pitchers = summary[summary["cluster"] == c]["Pitcher"].tolist()
        avg_velo = summary[summary["cluster"] == c]["avg_velo"].mean()
        print(f"Cluster {c} (avg velo {avg_velo:.1f}): {pitchers}")
    print()

    return kmeans, scaler, summary


def main():
    print("Loading and cleaning data...")
    raw    = load_raw(DATA_PATH)
    df     = clean(raw)
    print(f"  Swings after cleaning: {len(df):,}")
    print(f"  Whiff rate: {df['whiff'].mean():.3f}")
    print()

    print("Training whiff model...")
    model, scaler = train_whiff_model(df)

    print("Training KMeans...")
    kmeans, kmeans_scaler, pitcher_summary = train_kmeans(df)

    print("Saving artifacts...")
    joblib.dump(model,          MODEL_OUT)
    joblib.dump(scaler,         SCALER_OUT)
    joblib.dump(kmeans,         KMEANS_OUT)
    joblib.dump(kmeans_scaler,  KMEANS_SCALER)
    joblib.dump(df,             CLEAN_DATA_OUT)
    joblib.dump(pitcher_summary, "pitcher_summary.pkl")
    print(f"  Saved: {MODEL_OUT}, {SCALER_OUT}, {KMEANS_OUT}, {KMEANS_SCALER}")
    print(f"  Saved: {CLEAN_DATA_OUT}, pitcher_summary.pkl")
    print()
    print("Done. Run `streamlit run app.py` to launch the app.")


if __name__ == "__main__":
    main()
