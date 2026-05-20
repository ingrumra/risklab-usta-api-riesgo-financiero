# =============================================================================
# backend/app/services/fred.py – Integración con FRED (Federal Reserve)
# Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez
# =============================================================================
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

FRED_SERIES = {
    "DGS3MO": 0.25,
    "DGS1":   1.0,
    "DGS2":   2.0,
    "DGS5":   5.0,
    "DGS10":  10.0,
    "DGS30":  30.0,
}


def get_fred_client():
    """Obtiene cliente FRED si la API key está configurada."""
    from app.config import get_settings
    settings = get_settings()
    if not settings.fred_api_key:
        raise RuntimeError("FRED_API_KEY no configurada en .env")
    from fredapi import Fred
    return Fred(api_key=settings.fred_api_key)


def get_yield_curve_fred() -> dict:
    """Obtiene puntos de la curva de rendimiento desde FRED."""
    fred = get_fred_client()
    maturities, yields_pct = [], []
    for series_id, maturity in FRED_SERIES.items():
        try:
            data = fred.get_series(series_id).dropna()
            if not data.empty:
                maturities.append(maturity)
                yields_pct.append(float(data.iloc[-1]))
        except Exception as e:
            logger.warning(f"Error FRED {series_id}: {e}")
    return {"maturities": maturities, "yields_pct": yields_pct}


def get_inflation_us() -> float | None:
    """Obtiene inflación US interanual (CPIAUCSL) desde FRED."""
    try:
        fred = get_fred_client()
        cpi = fred.get_series("CPIAUCSL").dropna()
        if len(cpi) >= 13:
            inf = (float(cpi.iloc[-1]) / float(cpi.iloc[-13]) - 1) * 100
            return round(inf, 4)
    except Exception as e:
        logger.warning(f"Error FRED inflación: {e}")
    return None


def get_rf_fred() -> float | None:
    """Obtiene tasa libre de riesgo (DGS3MO) desde FRED."""
    try:
        fred = get_fred_client()
        data = fred.get_series("DGS3MO").dropna()
        if not data.empty:
            return round(float(data.iloc[-1]) / 100, 6)
    except Exception as e:
        logger.warning(f"Error FRED Rf: {e}")
    return None
