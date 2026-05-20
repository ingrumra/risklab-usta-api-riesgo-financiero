# =============================================================================
# backend/app/ml/model.py – Modelo ML con patrón Singleton
# Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez
#
# El modelo predice la dirección del rendimiento a N días vista.
# Pipeline: StandardScaler + RandomForestClassifier
# Singleton garantiza que el modelo se carga UNA sola vez en memoria.
# =============================================================================
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "model_v1.joblib"
MODEL_VERSION = "v1"


class MLModelSingleton:
    """
    Patrón Singleton: el modelo se carga UNA SOLA VEZ en memoria.
    Todas las llamadas a /predict usan la misma instancia.
    Verificable en los logs: el mensaje 'Modelo ML cargado' aparece una sola vez.
    """
    _instance: Optional["MLModelSingleton"] = None
    _pipeline: Optional[Pipeline] = None

    def __new__(cls) -> "MLModelSingleton":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self) -> None:
        if MODEL_PATH.exists():
            logger.info(f"[MLModelSingleton] Modelo ML cargado desde {MODEL_PATH}")
            self._pipeline = joblib.load(MODEL_PATH)
        else:
            logger.warning("[MLModelSingleton] Modelo no encontrado – entrenando modelo base...")
            self._pipeline = self._train_base_model()

    def _train_base_model(self) -> Pipeline:
        """
        Entrena un modelo base con datos sintéticos si no existe model_v1.joblib.
        En producción, usar train_model.py con datos reales.
        """
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 6)  # 6 features técnicas
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # regla simple para bootstrap

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=100, random_state=42)),
        ])
        pipeline.fit(X, y)
        joblib.dump(pipeline, MODEL_PATH)
        logger.info(f"[MLModelSingleton] Modelo base guardado en {MODEL_PATH}")
        return pipeline

    def predict(self, features: dict[str, float]) -> dict:
        """
        Realiza una predicción con el modelo cargado.
        Retorna dirección (UP/DOWN/NEUTRAL) y probabilidad.
        """
        if self._pipeline is None:
            raise RuntimeError("Modelo no disponible.")

        feature_names = ["rsi", "macd", "bb_position", "sma_ratio", "ret_5d", "vol_20d"]
        X = np.array([[features.get(f, 0.0) for f in feature_names]])

        pred_class = int(self._pipeline.predict(X)[0])
        proba = self._pipeline.predict_proba(X)[0]
        confidence = float(max(proba))

        direction = "UP" if pred_class == 1 else "DOWN"
        return {
            "direction":   direction,
            "confidence":  round(confidence, 4),
            "pred_class":  pred_class,
        }


def extract_features(prices: pd.Series, ticker: str) -> dict[str, float]:
    """
    Extrae features técnicas para el modelo ML.
    Evita data leakage usando ventanas sobre datos pasados.
    """
    from app.services.portfolio import TechnicalIndicators

    ind   = TechnicalIndicators.all_indicators(prices)
    px    = float(prices.iloc[-1])
    log_r = np.log(prices / prices.shift(1)).dropna()

    bb_range = ind["bb_upper"] - ind["bb_lower"]
    bb_pos   = ((px - ind["bb_lower"]) / bb_range) if bb_range > 0 else 0.5
    sma_ratio = (ind["sma_20"] / ind["sma_50"]) - 1 if ind["sma_50"] > 0 else 0.0
    ret_5d    = float((prices.iloc[-1] / prices.iloc[-6] - 1)) if len(prices) > 5 else 0.0
    vol_20d   = float(log_r.rolling(20).std().iloc[-1]) if len(log_r) >= 20 else 0.01

    return {
        "rsi":         ind["rsi"] / 100,
        "macd":        ind["macd"] / px if px != 0 else 0.0,
        "bb_position": bb_pos,
        "sma_ratio":   sma_ratio,
        "ret_5d":      ret_5d,
        "vol_20d":     vol_20d,
    }
