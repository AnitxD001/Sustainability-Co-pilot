"""
Database Seeder
================
Loads the synthetic CSV data (synthetic_data/generator.py output) into the
database defined by database/connection.py.

Usage:
    python -m synthetic_data.generator      # writes data/*.csv
    python -m database.seed                 # loads CSVs into the DB
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.connection import init_db, get_session, engine
from database.schema import Facility, DailyMetric

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def seed():
    facility_csv = DATA_DIR / "facility_master.csv"
    data_csv = DATA_DIR / "sustainability_data.csv"

    if not facility_csv.exists() or not data_csv.exists():
        raise FileNotFoundError(
            "Synthetic data not found. Run `python -m synthetic_data.generator` first."
        )

    print(f"Initializing database schema at: {engine.url}")
    init_db()

    facility_df = pd.read_csv(facility_csv)
    data_df = pd.read_csv(data_csv, parse_dates=["date"])

    session = get_session()
    try:
        print("Clearing existing rows...")
        session.query(DailyMetric).delete()
        session.query(Facility).delete()
        session.commit()

        print(f"Inserting {len(facility_df)} facilities...")
        facilities = [
            Facility(
                facility_id=row.facility_id,
                facility_type=row.facility_type,
                facility_name=row.facility_name,
                region=row.region,
            )
            for row in facility_df.itertuples()
        ]
        session.bulk_save_objects(facilities)
        session.commit()

        print(f"Inserting {len(data_df):,} daily metric records (this may take a moment)...")
        batch = []
        BATCH_SIZE = 2000
        for i, row in enumerate(data_df.itertuples(), start=1):
            batch.append(DailyMetric(
                date=row.date.date(),
                facility_id=row.facility_id,
                facility_type=row.facility_type,
                supplier_id=row.supplier_id,
                electricity_kwh=row.electricity_kwh,
                diesel_litres=row.diesel_litres,
                water_litres=row.water_litres,
                carbon_emissions_kg=row.carbon_emissions_kg,
                waste_generated_kg=row.waste_generated_kg,
                recycling_pct=row.recycling_pct,
                renewable_energy_pct=row.renewable_energy_pct,
                shipments_processed=row.shipments_processed,
                shipment_distance_km=row.shipment_distance_km,
                transport_mode=row.transport_mode,
                production_volume=row.production_volume,
                sustainability_incidents=row.sustainability_incidents,
            ))
            if len(batch) >= BATCH_SIZE:
                session.bulk_save_objects(batch)
                session.commit()
                batch = []
                print(f"  ...{i:,} records inserted")
        if batch:
            session.bulk_save_objects(batch)
            session.commit()

        print("Seed complete.")
    finally:
        session.close()


if __name__ == "__main__":
    seed()
