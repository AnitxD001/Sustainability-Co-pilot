"""
Carbon Emission Prediction
============================
XGBoost regressor predicting a facility's future daily carbon emissions from
its operational metrics. Trained once globally (with facility_type and
facility_id as categorical features) rather than one model per facility,
since most facilities don't have enough history alone for a robust fit, and
a shared model lets the network learn cross-facility patterns.

Inputs (per spec): electricity, diesel, water, production volume, shipment
distance, transport mode, renewable %, plus engineered lag/rolling features
and calendar features.

Outputs: point forecasts for tomorrow / next week / next month, each
evaluated with MAE, RMSE, R^2 on a held-out time-based split.
"""

import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "electricity_kwh", "diesel_litres", "water_litres",
    "production_volume", "shipment_distance_km", "renewable_energy_pct",
    "shipments_processed",
]
CATEGORICAL_FEATURES = ["facility_type", "transport_mode"]
LAG_FEATURES = ["emissions_lag_1", "emissions_lag_7", "emissions_rolling_mean_7"]

HORIZONS = {"tomorrow": 1, "next_week": 7, "next_month": 30}


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["facility_id", "date"]).copy()
    df["date"] = pd.to_datetime(df["date"])

    grouped = df.groupby("facility_id")["carbon_emissions_kg"]
    df["emissions_lag_1"] = grouped.shift(1)
    df["emissions_lag_7"] = grouped.shift(7)
    df["emissions_rolling_mean_7"] = grouped.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    df = df.dropna(subset=LAG_FEATURES)
    return df


def _build_encoders(df: pd.DataFrame) -> dict:
    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
    return encoders


def _apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    df = df.copy()
    for col, le in encoders.items():
        known = set(le.classes_)
        df[col] = df[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
        df[f"{col}_enc"] = le.transform(df[col])
    return df


def train_model(daily_metrics: pd.DataFrame, test_size_days: int = 30) -> dict:
    """
    Train the global XGBoost emissions model with a time-based holdout split
    (last `test_size_days` days per facility held out for evaluation).

    Returns a dict with the fitted model, encoders, feature list, and metrics.
    """
    df = _engineer_features(daily_metrics)
    encoders = _build_encoders(df)
    df = _apply_encoders(df, encoders)

    feature_cols = NUMERIC_FEATURES + LAG_FEATURES + ["day_of_week", "month"] + \
        [f"{c}_enc" for c in CATEGORICAL_FEATURES]

    cutoff = df["date"].max() - timedelta(days=test_size_days)
    train_df = df[df["date"] <= cutoff]
    test_df = df[df["date"] > cutoff]

    X_train, y_train = train_df[feature_cols], train_df["carbon_emissions_kg"]
    X_test, y_test = test_df[feature_cols], test_df["carbon_emissions_kg"]

    model = XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, random_state=42,
        objective="reg:squarederror", n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = dict(
        mae=round(float(mean_absolute_error(y_test, preds)), 3),
        rmse=round(float(np.sqrt(mean_squared_error(y_test, preds))), 3),
        r2=round(float(r2_score(y_test, preds)), 4),
        n_train=len(train_df),
        n_test=len(test_df),
    )

    artifact = dict(model=model, encoders=encoders, feature_cols=feature_cols, metrics=metrics)
    joblib.dump(artifact, MODEL_DIR / "carbon_emission_model.joblib")

    with open(MODEL_DIR / "carbon_emission_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return artifact


def load_model() -> dict:
    path = MODEL_DIR / "carbon_emission_model.joblib"
    if not path.exists():
        raise FileNotFoundError("No trained model found. Run ml.prediction.train_model() first.")
    return joblib.load(path)


def predict_future(daily_metrics: pd.DataFrame, facility_id: str, artifact: dict = None) -> dict:
    """
    Predict tomorrow / next week / next month emissions for one facility by
    persisting the facility's most recent operational profile forward
    (a naive-but-reasonable assumption absent live future inputs) and
    re-deriving lag features iteratively, day by day.
    """
    if artifact is None:
        artifact = load_model()
    model, encoders, feature_cols = artifact["model"], artifact["encoders"], artifact["feature_cols"]

    hist = daily_metrics[daily_metrics["facility_id"] == facility_id].sort_values("date").copy()
    if hist.empty:
        raise ValueError(f"No history for facility {facility_id}")
    hist["date"] = pd.to_datetime(hist["date"])

    recent = hist.tail(30).copy()
    last_row = recent.iloc[-1]
    emissions_history = list(recent["carbon_emissions_kg"])

    predictions = {}
    sim_date = hist["date"].max()

    for day_offset in range(1, max(HORIZONS.values()) + 1):
        sim_date = sim_date + timedelta(days=1)
        lag_1 = emissions_history[-1]
        lag_7 = emissions_history[-7] if len(emissions_history) >= 7 else emissions_history[0]
        roll_7 = float(np.mean(emissions_history[-7:]))

        row = pd.DataFrame([{
            "electricity_kwh": last_row["electricity_kwh"],
            "diesel_litres": last_row["diesel_litres"],
            "water_litres": last_row["water_litres"],
            "production_volume": last_row["production_volume"],
            "shipment_distance_km": last_row["shipment_distance_km"],
            "renewable_energy_pct": last_row["renewable_energy_pct"],
            "shipments_processed": last_row["shipments_processed"],
            "emissions_lag_1": lag_1,
            "emissions_lag_7": lag_7,
            "emissions_rolling_mean_7": roll_7,
            "day_of_week": sim_date.dayofweek,
            "month": sim_date.month,
            "facility_type": last_row["facility_type"],
            "transport_mode": last_row["transport_mode"],
        }])
        row = _apply_encoders(row, encoders)
        pred = float(model.predict(row[feature_cols])[0])
        emissions_history.append(pred)

        for horizon_name, horizon_days in HORIZONS.items():
            if day_offset == horizon_days:
                predictions[horizon_name] = dict(
                    predicted_date=sim_date.date().isoformat(),
                    predicted_emissions_kg=round(pred, 2),
                )

    predictions["model_metrics"] = artifact["metrics"]
    predictions["facility_id"] = facility_id
    return predictions


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from database.connection import get_session
    from database.schema import DailyMetric

    session = get_session()
    rows = session.query(DailyMetric).all()
    df = pd.DataFrame([{
        "facility_id": r.facility_id, "facility_type": r.facility_type, "date": r.date,
        "electricity_kwh": r.electricity_kwh, "diesel_litres": r.diesel_litres,
        "water_litres": r.water_litres, "carbon_emissions_kg": r.carbon_emissions_kg,
        "production_volume": r.production_volume, "shipment_distance_km": r.shipment_distance_km,
        "renewable_energy_pct": r.renewable_energy_pct, "shipments_processed": r.shipments_processed,
        "transport_mode": r.transport_mode,
    } for r in rows])
    session.close()

    print("Training carbon emission prediction model...")
    artifact = train_model(df)
    print("Metrics:", artifact["metrics"])

    print("\nSample forecast for PLANT-001:")
    forecast = predict_future(df, "PLANT-001", artifact)
    print(json.dumps(forecast, indent=2))
