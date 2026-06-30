"""
Anomaly Detection
==================
Isolation Forest trained per facility_type (a Plant's "normal" electricity
usage is wildly different from a Distribution Center's, so pooling them
would make the model nearly useless). For each flagged anomaly we also run a
lightweight heuristic to explain *which* metric drove the flag and bucket a
severity, since "an anomaly was detected" alone isn't actionable.
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = [
    "electricity_kwh", "diesel_litres", "water_litres",
    "carbon_emissions_kg", "waste_generated_kg",
]

CONTAMINATION = 0.025  # expected anomaly rate, matches the synthetic injection rate


def _severity_from_score(score: float, score_min: float, score_max: float) -> str:
    """Lower (more negative) isolation-forest scores = more anomalous."""
    if score_max - score_min < 1e-9:
        return "Low"
    normalized = (score - score_min) / (score_max - score_min)  # 0 = most anomalous, 1 = least
    if normalized < 0.33:
        return "High"
    if normalized < 0.66:
        return "Medium"
    return "Low"


def _likely_cause(row: pd.Series, facility_baseline: pd.Series) -> str:
    """Identify which feature deviates most (in std-devs) from the facility's own baseline."""
    deviations = {}
    for feat in FEATURES:
        mean = facility_baseline[f"{feat}_mean"]
        std = facility_baseline[f"{feat}_std"] or 1.0
        deviations[feat] = abs(row[feat] - mean) / std
    top_feature = max(deviations, key=deviations.get)
    direction = "spike in" if row[top_feature] > facility_baseline[f"{top_feature}_mean"] else "drop in"
    readable = {
        "electricity_kwh": "electricity consumption",
        "diesel_litres": "diesel consumption",
        "water_litres": "water usage (possible leakage)",
        "carbon_emissions_kg": "carbon emissions",
        "waste_generated_kg": "waste generation",
    }
    return f"Unusual {direction} {readable[top_feature]}"


def detect_anomalies(daily_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Run Isolation Forest per facility_type and return flagged anomalies.

    Parameters
    ----------
    daily_metrics : DataFrame with facility_id, facility_type, date, and FEATURES columns.

    Returns
    -------
    DataFrame: facility_id, date, severity, anomaly_score, likely_cause, metric_snapshot (JSON)
    """
    df = daily_metrics.copy()
    df["date"] = pd.to_datetime(df["date"])

    all_anomalies = []

    for ftype, group in df.groupby("facility_type"):
        if len(group) < 30:
            continue  # not enough data to fit a meaningful model

        X = group[FEATURES].values
        model = IsolationForest(
            n_estimators=200, contamination=CONTAMINATION, random_state=42
        )
        preds = model.fit_predict(X)
        scores = model.decision_function(X)

        group = group.copy()
        group["is_anomaly"] = preds == -1
        group["anomaly_score"] = scores

        anomalies = group[group["is_anomaly"]].copy()
        if anomalies.empty:
            continue

        score_min, score_max = scores.min(), scores.max()

        # Per-facility baseline (mean/std) for explaining the cause
        baseline = group.groupby("facility_id")[FEATURES].agg(["mean", "std"])
        baseline.columns = [f"{a}_{b}" for a, b in baseline.columns]

        for _, row in anomalies.iterrows():
            fac_baseline = baseline.loc[row["facility_id"]]
            severity = _severity_from_score(row["anomaly_score"], score_min, score_max)
            cause = _likely_cause(row, fac_baseline)
            snapshot = {f: float(row[f]) for f in FEATURES}

            all_anomalies.append(dict(
                facility_id=row["facility_id"],
                facility_type=ftype,
                date=row["date"].date().isoformat(),
                severity=severity,
                anomaly_score=round(float(row["anomaly_score"]), 4),
                likely_cause=cause,
                metric_snapshot=json.dumps(snapshot),
            ))

    result = pd.DataFrame(all_anomalies)
    if not result.empty:
        severity_order = {"High": 0, "Medium": 1, "Low": 2}
        result = result.sort_values(
            by="severity", key=lambda s: s.map(severity_order)
        ).reset_index(drop=True)
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from database.connection import get_session
    from database.schema import DailyMetric

    session = get_session()
    rows = session.query(DailyMetric).all()
    df = pd.DataFrame([{
        "facility_id": r.facility_id, "facility_type": r.facility_type, "date": r.date,
        "electricity_kwh": r.electricity_kwh, "diesel_litres": r.diesel_litres,
        "water_litres": r.water_litres, "carbon_emissions_kg": r.carbon_emissions_kg,
        "waste_generated_kg": r.waste_generated_kg,
    } for r in rows])
    session.close()

    anomalies = detect_anomalies(df)
    print(f"Detected {len(anomalies)} anomalies out of {len(df):,} records "
          f"({len(anomalies)/len(df)*100:.2f}%)")
    print(anomalies.head(10).to_string(index=False))
    print("\nSeverity breakdown:")
    print(anomalies["severity"].value_counts())
