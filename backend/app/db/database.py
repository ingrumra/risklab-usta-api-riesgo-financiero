# =============================================================================
# backend/app/db/database.py – Persistencia SQLAlchemy ORM
# Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez
# =============================================================================
from __future__ import annotations

import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # necesario para SQLite con FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────
# MODELOS ORM
# ─────────────────────────────────────────────────────────────────

class PrecioCache(Base):
    """Caché persistente de precios descargados de Yahoo Finance."""
    __tablename__ = "precio_cache"

    id         = Column(Integer, primary_key=True, index=True)
    ticker     = Column(String(20), nullable=False, index=True)
    period     = Column(String(10), nullable=False)
    fecha      = Column(String(12), nullable=False)
    precio     = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class MacroCache(Base):
    """Caché persistente de indicadores macro con TTL de 24 h."""
    __tablename__ = "macro_cache"

    id         = Column(Integer, primary_key=True, index=True)
    clave      = Column(String(50), unique=True, nullable=False, index=True)
    valor      = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)


class PredictionLog(Base):
    """
    Registro de todas las predicciones del modelo ML.
    Cumple el requerimiento de versionar y auditar el modelo.
    """
    __tablename__ = "prediction_log"

    id              = Column(Integer, primary_key=True, index=True)
    model_version   = Column(String(50), nullable=False, default="v1")
    ticker          = Column(String(20), nullable=False)
    horizon_days    = Column(Integer, nullable=False, default=5)
    features_json   = Column(Text, nullable=True)
    prediction      = Column(Float, nullable=False)
    direction       = Column(String(10), nullable=False)  # "UP" | "DOWN" | "NEUTRAL"
    confidence      = Column(Float, nullable=True)
    created_at      = Column(DateTime, default=datetime.datetime.utcnow)


class StressLog(Base):
    """Registro de escenarios de stress testing aplicados."""
    __tablename__ = "stress_log"

    id            = Column(Integer, primary_key=True, index=True)
    scenario_name = Column(String(100), nullable=False)
    portfolio_var = Column(Float, nullable=True)
    max_loss_pct  = Column(Float, nullable=True)
    created_at    = Column(DateTime, default=datetime.datetime.utcnow)


def create_tables() -> None:
    """Crea todas las tablas si no existen."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependencia FastAPI: inyecta sesión de BD en cada request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
