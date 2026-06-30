"""
Database Schema
================
SQLAlchemy ORM models for the sustainability tracking system.

Tables
------
- facilities          : static facility master data
- daily_metrics       : the core fact table (one row per facility per day)
- sustainability_scores : computed scoring engine output (ml/scoring.py)
- anomalies           : computed anomaly detection output (ml/anomaly.py)
- recommendations     : computed recommendation engine output (backend/recommendation.py)
- emission_predictions: computed forecasting output (ml/prediction.py)

Works against PostgreSQL in production (DATABASE_URL env var) and falls back
to a local SQLite file for development/testing — see database/connection.py.
"""

from sqlalchemy import (
    Column, String, Float, Integer, Boolean, Date, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import relationship
from datetime import datetime

from database.connection import Base


class Facility(Base):
    __tablename__ = "facilities"

    facility_id = Column(String(20), primary_key=True)
    facility_type = Column(String(20), nullable=False, index=True)  # Supplier/Plant/Warehouse/DC
    facility_name = Column(String(100), nullable=False)
    region = Column(String(50))

    metrics = relationship("DailyMetric", back_populates="facility", cascade="all, delete-orphan")


class DailyMetric(Base):
    __tablename__ = "daily_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True)
    facility_id = Column(String(20), ForeignKey("facilities.facility_id"), nullable=False, index=True)
    facility_type = Column(String(20), index=True)
    supplier_id = Column(String(20), index=True)

    electricity_kwh = Column(Float, nullable=False)
    diesel_litres = Column(Float, nullable=False)
    water_litres = Column(Float, nullable=False)
    carbon_emissions_kg = Column(Float, nullable=False)
    waste_generated_kg = Column(Float, nullable=False)
    recycling_pct = Column(Float, nullable=False)
    renewable_energy_pct = Column(Float, nullable=False)
    shipments_processed = Column(Integer, nullable=False)
    shipment_distance_km = Column(Float, nullable=False)
    transport_mode = Column(String(10))
    production_volume = Column(Float, nullable=False)
    sustainability_incidents = Column(Integer, default=0)

    facility = relationship("Facility", back_populates="metrics")


class SustainabilityScore(Base):
    __tablename__ = "sustainability_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    facility_id = Column(String(20), ForeignKey("facilities.facility_id"), nullable=False, index=True)
    period_start = Column(Date, nullable=False)
    period_end = Column(Date, nullable=False)
    carbon_subscore = Column(Float)
    energy_subscore = Column(Float)
    water_subscore = Column(Float)
    waste_subscore = Column(Float)
    renewable_subscore = Column(Float)
    recycling_subscore = Column(Float)
    overall_score = Column(Float, nullable=False)
    classification = Column(String(20), nullable=False)
    computed_at = Column(DateTime, default=datetime.utcnow)


class Anomaly(Base):
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    facility_id = Column(String(20), ForeignKey("facilities.facility_id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    severity = Column(String(10), nullable=False)  # Low / Medium / High
    anomaly_score = Column(Float, nullable=False)
    likely_cause = Column(String(200))
    metric_snapshot = Column(Text)  # JSON-encoded snapshot of the offending metrics
    detected_at = Column(DateTime, default=datetime.utcnow)


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    facility_id = Column(String(20), ForeignKey("facilities.facility_id"), nullable=False, index=True)
    category = Column(String(50), nullable=False)
    recommendation_text = Column(Text, nullable=False)
    estimated_impact = Column(String(200))
    priority = Column(String(10))  # Low / Medium / High
    generated_at = Column(DateTime, default=datetime.utcnow)


class EmissionPrediction(Base):
    __tablename__ = "emission_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    facility_id = Column(String(20), ForeignKey("facilities.facility_id"), nullable=False, index=True)
    horizon = Column(String(20), nullable=False)  # tomorrow / next_week / next_month
    predicted_date = Column(Date, nullable=False)
    predicted_emissions_kg = Column(Float, nullable=False)
    model_mae = Column(Float)
    model_rmse = Column(Float)
    model_r2 = Column(Float)
    generated_at = Column(DateTime, default=datetime.utcnow)
