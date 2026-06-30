"""
Synthetic Data Generator
=========================
Generates a realistic, internally-consistent daily sustainability dataset for
an apparel manufacturing supply chain:

    50 Suppliers -> 8 Manufacturing Plants -> 15 Warehouses -> 30 Distribution Centers

Over 365 days this produces ~37,600 records (103 facilities x 365 days).

Design notes
------------
- Each facility gets a fixed "profile" (base consumption rates, a renewable
  energy trajectory, and a long-term trend) so that the resulting time series
  is coherent rather than pure noise. This is what lets the ML modules
  (regression / anomaly detection / scoring) learn something meaningful.
- Carbon emissions are *derived* from electricity, diesel and production
  volume using standard-ish emission factors, not generated independently.
  This keeps the dataset physically plausible and gives the prediction model
  real signal to learn from.
- ~2.5% of facility-days are injected as anomalies (sudden spikes), which the
  anomaly detection module is expected to surface.
- Each facility is assigned a slow drift (improving / worsening / stable) so
  that month-over-month trend insights ("Plant 5 emissions up 18%") are real.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

RNG_SEED = 42
N_SUPPLIERS = 50
N_PLANTS = 8
N_WAREHOUSES = 15
N_DISTRIBUTION_CENTERS = 30
N_DAYS = 365
START_DATE = datetime(2025, 1, 1)

TRANSPORT_MODES = ["Road", "Rail", "Air"]
TRANSPORT_MODE_WEIGHTS = [0.70, 0.22, 0.08]

REGIONS = ["North", "South", "East", "West", "Central"]

# Emission factors (kept simple/plausible, not authoritative)
GRID_EMISSION_FACTOR = 0.71      # kg CO2 per kWh of non-renewable electricity
DIESEL_EMISSION_FACTOR = 2.68    # kg CO2 per litre of diesel
PRODUCTION_EMISSION_FACTOR = 0.04  # kg CO2 per unit produced (process emissions)
SHIPMENT_EMISSION_FACTOR = {"Road": 0.12, "Rail": 0.04, "Air": 0.55}  # kg CO2 per km per shipment-equivalent


def _facility_id(prefix: str, idx: int) -> str:
    return f"{prefix}-{idx:03d}"


def build_facility_master():
    """Create the static facility master table with per-facility profiles."""
    rng = np.random.default_rng(RNG_SEED)
    facilities = []

    def base_profile(ftype):
        if ftype == "Supplier":
            return dict(electricity=(800, 200), diesel=(150, 60), water=(3000, 900),
                        waste=(300, 90), production=(2000, 500))
        if ftype == "Plant":
            return dict(electricity=(18000, 4000), diesel=(900, 250), water=(40000, 9000),
                        waste=(3500, 800), production=(15000, 3500))
        if ftype == "Warehouse":
            return dict(electricity=(4500, 1200), diesel=(600, 180), water=(1200, 350),
                        waste=(700, 200), production=(0, 0))
        if ftype == "DC":
            return dict(electricity=(2500, 700), diesel=(950, 260), water=(800, 250),
                        waste=(500, 150), production=(0, 0))
        raise ValueError(ftype)

    counts = [
        ("Supplier", "SUP", N_SUPPLIERS),
        ("Plant", "PLANT", N_PLANTS),
        ("Warehouse", "WH", N_WAREHOUSES),
        ("DC", "DC", N_DISTRIBUTION_CENTERS),
    ]

    for ftype, prefix, n in counts:
        prof = base_profile(ftype)
        for i in range(1, n + 1):
            fid = _facility_id(prefix, i)
            # Long-term drift: -1 = improving (emissions trending down),
            # 0 = stable, +1 = worsening (emissions trending up)
            trend = rng.choice([-1, 0, 1], p=[0.35, 0.35, 0.30])
            trend_magnitude = rng.uniform(0.05, 0.25)  # up to 25% drift over the year
            renewable_start = rng.uniform(5, 35)
            renewable_end = np.clip(renewable_start + rng.uniform(-5, 25), 2, 85)

            facilities.append(dict(
                facility_id=fid,
                facility_type=ftype,
                facility_name=f"{ftype} {i}",
                region=rng.choice(REGIONS),
                elec_base=max(50, rng.normal(*prof["electricity"])),
                diesel_base=max(20, rng.normal(*prof["diesel"])),
                water_base=max(100, rng.normal(*prof["water"])),
                waste_base=max(20, rng.normal(*prof["waste"])),
                production_base=max(0, rng.normal(*prof["production"])),
                trend_direction=trend,
                trend_magnitude=trend_magnitude,
                renewable_start=renewable_start,
                renewable_end=renewable_end,
                recycling_base=np.clip(rng.normal(45, 15), 5, 95),
                baseline_incident_rate=rng.uniform(0.002, 0.02),
            ))

    return pd.DataFrame(facilities)


def assign_supplier_links(facility_master: pd.DataFrame, rng):
    """Give every non-supplier facility 1-3 candidate upstream suppliers."""
    suppliers = facility_master.loc[facility_master.facility_type == "Supplier", "facility_id"].tolist()
    links = {}
    for fid, ftype in zip(facility_master.facility_id, facility_master.facility_type):
        if ftype == "Supplier":
            links[fid] = [fid]
        else:
            k = rng.integers(1, 4)
            links[fid] = list(rng.choice(suppliers, size=k, replace=False))
    return links


def generate_dataset(n_days: int = N_DAYS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    facility_master = build_facility_master()
    supplier_links = assign_supplier_links(facility_master, rng)

    dates = [START_DATE + timedelta(days=d) for d in range(n_days)]
    rows = []

    for _, fac in facility_master.iterrows():
        fid = fac.facility_id
        ftype = fac.facility_type
        candidate_suppliers = supplier_links[fid]

        for d_idx, date in enumerate(dates):
            progress = d_idx / max(1, n_days - 1)  # 0 -> 1 across the year

            # Weekly seasonality: weekends quieter for plants/warehouses/DCs
            weekday = date.weekday()
            weekend_factor = 0.55 if weekday >= 5 and ftype != "Supplier" else 1.0

            # Slow drift applied multiplicatively to consumption (worse=more, improving=less)
            drift_factor = 1 + (fac.trend_direction * fac.trend_magnitude * progress)

            # Small day-to-day noise
            noise = rng.normal(1.0, 0.06)

            electricity = max(10, fac.elec_base * weekend_factor * drift_factor * noise)
            diesel = max(0, fac.diesel_base * weekend_factor * drift_factor * rng.normal(1.0, 0.10))
            water = max(10, fac.water_base * weekend_factor * drift_factor * rng.normal(1.0, 0.07))
            production = max(0, fac.production_base * weekend_factor * (2 - drift_factor) * rng.normal(1.0, 0.08))
            waste = max(0, fac.waste_base * weekend_factor * drift_factor * rng.normal(1.0, 0.09))

            renewable_pct = np.clip(
                fac.renewable_start + (fac.renewable_end - fac.renewable_start) * progress
                + rng.normal(0, 2), 1, 90
            )
            recycling_pct = np.clip(fac.recycling_base + rng.normal(0, 5)
                                     - (fac.trend_direction * 5 * progress), 2, 98)

            shipments = max(0, int(rng.poisson(40 if ftype in ("Warehouse", "DC") else
                                                 15 if ftype == "Plant" else 8) * weekend_factor))
            transport_mode = rng.choice(TRANSPORT_MODES, p=TRANSPORT_MODE_WEIGHTS)
            shipment_distance = max(1, rng.normal(350 if transport_mode == "Air" else
                                                    180 if transport_mode == "Rail" else 90, 60))

            supplier_id = rng.choice(candidate_suppliers)

            # --- Anomaly injection (~2.5% of rows) ---
            is_anomaly = rng.random() < 0.025
            anomaly_type = None
            if is_anomaly:
                anomaly_type = rng.choice(
                    ["electricity_spike", "diesel_spike", "water_spike", "waste_spike"],
                    p=[0.35, 0.30, 0.20, 0.15]
                )
                spike_mult = rng.uniform(2.2, 4.0)
                if anomaly_type == "electricity_spike":
                    electricity *= spike_mult
                elif anomaly_type == "diesel_spike":
                    diesel *= spike_mult
                elif anomaly_type == "water_spike":
                    water *= spike_mult
                elif anomaly_type == "waste_spike":
                    waste *= spike_mult

            # --- Derived carbon emissions (physically grounded, not random) ---
            non_renewable_share = (100 - renewable_pct) / 100
            emissions = (
                electricity * non_renewable_share * GRID_EMISSION_FACTOR
                + diesel * DIESEL_EMISSION_FACTOR
                + production * PRODUCTION_EMISSION_FACTOR
                + shipments * shipment_distance * SHIPMENT_EMISSION_FACTOR[transport_mode] / 100
            )
            emissions = max(1, emissions * rng.normal(1.0, 0.03))

            incidents = rng.poisson(fac.baseline_incident_rate * (3 if is_anomaly else 1))

            rows.append(dict(
                date=date.strftime("%Y-%m-%d"),
                facility_id=fid,
                facility_type=ftype,
                facility_name=fac.facility_name,
                region=fac.region,
                supplier_id=supplier_id,
                electricity_kwh=round(electricity, 2),
                diesel_litres=round(diesel, 2),
                water_litres=round(water, 2),
                carbon_emissions_kg=round(emissions, 2),
                waste_generated_kg=round(waste, 2),
                recycling_pct=round(recycling_pct, 2),
                renewable_energy_pct=round(renewable_pct, 2),
                shipments_processed=shipments,
                shipment_distance_km=round(shipment_distance, 2),
                transport_mode=transport_mode,
                production_volume=round(production, 2),
                sustainability_incidents=int(incidents),
                is_injected_anomaly=is_anomaly,
                anomaly_type=anomaly_type,
            ))

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df, facility_master


def main():
    out_dir = Path(__file__).resolve().parent.parent / "data"
    out_dir.mkdir(exist_ok=True)

    print("Generating synthetic dataset...")
    df, facility_master = generate_dataset()

    data_path = out_dir / "sustainability_data.csv"
    facility_path = out_dir / "facility_master.csv"

    df.to_csv(data_path, index=False)
    facility_master.to_csv(facility_path, index=False)

    print(f"Wrote {len(df):,} records for {facility_master.shape[0]} facilities -> {data_path}")
    print(f"Wrote facility master -> {facility_path}")
    print(f"Injected anomalies: {df['is_injected_anomaly'].sum():,} ({df['is_injected_anomaly'].mean()*100:.2f}%)")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
