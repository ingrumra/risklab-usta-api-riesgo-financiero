# =============================================================================
# backend/app/services/data.py – Servicio de datos financieros
# Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez
# =============================================================================
from __future__ import annotations

import datetime
import functools
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import yfinance as yf

# ✅ FIX Error 1: importación condicional para evitar error de Pylance
# si sqlalchemy no está instalado en el entorno del linter.
# En tiempo de ejecución, el import real ocurre dentro de DataService.
if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from app.db.database import MacroCache, PrecioCache

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# DECORADORES PERSONALIZADOS (Semana 1 del curso)
# ─────────────────────────────────────────────────────────────────

def log_execution_time(func: Any) -> Any:
    """Decorador: mide y loguea el tiempo de ejecución de cada función."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"[{func.__name__}] ejecutado en {elapsed:.3f}s")
        return result
    return wrapper


def cache_result(ttl_seconds: int = 3600) -> Any:
    """
    Decorador factory con TTL en memoria.
    Para caché persistente usamos SQLAlchemy (ver DataService).
    """
    def decorator(func: Any) -> Any:
        _cache: dict[str, tuple[float, Any]] = {}

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = str(args) + str(sorted(kwargs.items()))
            now = time.time()
            if key in _cache:
                ts, val = _cache[key]
                if now - ts < ttl_seconds:
                    return val
            result = func(*args, **kwargs)
            _cache[key] = (now, result)
            return result
        return wrapper
    return decorator


# ─────────────────────────────────────────────────────────────────
# SERVICIO DE DATOS
# ─────────────────────────────────────────────────────────────────

class DataService:
    """
    Servicio de acceso a datos financieros.
    Usa caché en memoria + persistencia SQLite vía SQLAlchemy.
    """

    @log_execution_time
    @cache_result(ttl_seconds=3600)
    def get_prices(self, ticker: str, period: str = "2y") -> pd.Series:
        """Descarga precios de cierre ajustados desde Yahoo Finance."""
        try:
            raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if raw is None or raw.empty:
                raise ValueError(f"No se encontraron datos para el ticker '{ticker}'.")
            close = raw["Close"]
            if not isinstance(close, pd.Series):
                close = pd.Series(close.iloc[:, 0])
            return close.dropna()
        except Exception as e:
            raise RuntimeError(f"Error al obtener datos de '{ticker}': {e}") from e

    @log_execution_time
    def get_multiple_prices(self, tickers: list[str], period: str = "2y") -> pd.DataFrame:
        """Descarga precios individualmente, omitiendo tickers que fallen."""
        frames = {}
        for ticker in tickers:
            try:
                raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
                if raw is None or raw.empty:
                    logger.warning(f"{ticker}: sin datos, omitiendo.")
                    continue
                close = raw["Close"]
                if not isinstance(close, pd.Series):
                    close = pd.Series(close.iloc[:, 0])
                close = close.dropna()
                close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
                if len(close) > 10:
                    frames[ticker] = close
                else:
                    logger.warning(f"{ticker}: muy pocos datos, omitiendo.")
            except Exception as e:
                logger.warning(f"{ticker}: error al descargar: {e}")
        if not frames:
            raise RuntimeError("No se pudo descargar ningún ticker.")
        df = pd.DataFrame(frames)
        df = df.ffill().bfill().dropna(how="all")
        logger.info(f"Descargados: {list(df.columns)} | shape: {df.shape}")
        return df

    @log_execution_time
    @cache_result(ttl_seconds=3600)
    def get_rf_rate(self) -> float:
        """Obtiene la tasa libre de riesgo (T-Bill 13W) desde Yahoo Finance."""
        try:
            h = yf.Ticker("^IRX").history(period="5d")
            if h is not None and not h.empty:
                close_val = h["Close"].dropna()
                if len(close_val) > 0:
                    return float(close_val.iloc[-1]) / 100
        except Exception:
            pass
        return 0.045

    @log_execution_time
    @cache_result(ttl_seconds=3600)
    def get_macro(self) -> dict[str, float | None]:
        """Obtiene indicadores macroeconómicos actualizados vía Yahoo Finance."""
        result: dict[str, float | None] = {"rf_annual": self.get_rf_rate()}
        symbols = {
            "vix":    "^VIX",
            "gold":   "GC=F",
            "oil":    "BZ=F",
            "usdcop": "COP=X",
            "dxy":    "DX-Y.NYB",
        }
        for key, sym in symbols.items():
            try:
                h = yf.Ticker(sym).history(period="5d")
                if h is not None and not h.empty:
                    close_val = h["Close"].dropna()
                    result[key] = float(close_val.iloc[-1]) if len(close_val) > 0 else None
                else:
                    result[key] = None
            except Exception:
                result[key] = None
        try:
            from app.services.fred import get_inflation_us
            result["inflation_us"] = get_inflation_us()
        except Exception:
            result["inflation_us"] = None
        return result

    def get_yield_curve(self) -> dict:
        """Obtiene puntos de la curva de rendimiento US desde FRED."""
        try:
            from app.services.fred import get_yield_curve_fred
            return get_yield_curve_fred()
        except Exception as e:
            logger.warning(f"FRED no disponible: {e}. Usando fallback Yahoo Finance.")
            return self._yield_curve_fallback()

    def _yield_curve_fallback(self) -> dict:
        """Fallback: obtiene rendimientos de bonos US vía Yahoo Finance."""
        tickers_map = {0.25: "^IRX", 5.0: "^FVX", 10.0: "^TNX", 30.0: "^TYX"}
        maturities, yields_pct = [], []
        for mat, sym in tickers_map.items():
            try:
                h = yf.Ticker(sym).history(period="5d")
                if h is not None and not h.empty:
                    val = float(h["Close"].dropna().iloc[-1])
                    if val > 0:
                        maturities.append(mat)
                        yields_pct.append(val)
                        logger.info(f"Yield {sym} ({mat}y): {val:.4f}%")
            except Exception as e:
                logger.warning(f"Error yield {sym}: {e}")
        if len(maturities) < 3:
            logger.warning("Usando curva de referencia hardcodeada como fallback.")
            maturities = [0.25, 1.0, 2.0, 5.0, 10.0, 30.0]
            yields_pct = [4.30, 4.10, 3.95, 3.85, 4.20, 4.60]
        return {"maturities": maturities, "yields_pct": yields_pct}