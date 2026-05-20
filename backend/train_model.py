"""
train_model.py – Script standalone para reentrenar el modelo ML
=====================================================================
Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez
Proyecto Integrador – Teoría del Riesgo · USTA

Uso:
    cd backend
    python train_model.py --tickers MSI XOM JNJ PG UL TSM --period 3y

El modelo entrenado se serializa en backend/app/ml/model_v1.joblib.
El backend (MLModelSingleton) cargará automáticamente la versión actualizada
en el próximo reinicio o si se llama al endpoint /predict.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_OUT = Path(__file__).parent / "app" / "ml" / "model_v1.joblib"
FEATURE_NAMES = ["rsi", "macd", "bb_position", "sma_ratio", "ret_5d", "vol_20d"]


def compute_features(prices: pd.Series) -> pd.DataFrame:
    """
    Calcula todas las features técnicas para un activo.
    Evita data leakage: todas las features son backward-looking.
    """
    df = pd.DataFrame({"price": prices})

    # RSI(14)
    delta = df["price"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = (100 - (100 / (1 + gain / loss))) / 100  # normalizado 0-1

    # MACD / precio
    ema12 = df["price"].ewm(span=12).mean()
    ema26 = df["price"].ewm(span=26).mean()
    df["macd"] = (ema12 - ema26) / df["price"]

    # Posición en Bandas de Bollinger
    bb_mid = df["price"].rolling(20).mean()
    bb_std = df["price"].rolling(20).std()
    bb_u   = bb_mid + 2 * bb_std
    bb_l   = bb_mid - 2 * bb_std
    df["bb_position"] = ((df["price"] - bb_l) / (bb_u - bb_l)).clip(0, 1)

    # Ratio SMA
    sma20 = df["price"].rolling(20).mean()
    sma50 = df["price"].rolling(50).mean()
    df["sma_ratio"] = (sma20 / sma50) - 1

    # Retorno 5 días
    df["ret_5d"] = df["price"].pct_change(5)

    # Volatilidad histórica 20 días
    log_r = np.log(df["price"] / df["price"].shift(1))
    df["vol_20d"] = log_r.rolling(20).std()

    return df[FEATURE_NAMES]


def build_dataset(
    tickers: list[str], period: str, horizon: int = 5
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Descarga precios y construye X, y para todos los activos.
    Target: 1 si el rendimiento a 'horizon' días es positivo, 0 si negativo.
    """
    X_all, y_all, idx_all = [], [], []

    for ticker in tickers:
        logger.info(f"Descargando {ticker} ({period})…")
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if raw.empty:
            logger.warning(f"Sin datos para {ticker}, omitiendo.")
            continue

        prices = raw["Close"].squeeze().dropna()
        features = compute_features(prices)

        # Target: rendimiento logarítmico a horizon días
        log_ret = np.log(prices / prices.shift(1))
        future_ret = log_ret.shift(-horizon)  # rendimiento futuro (no hay leakage: es target)

        df_full = features.copy()
        df_full["target_ret"] = future_ret.values
        df_full["target"] = (df_full["target_ret"] > 0).astype(int)

        df_clean = df_full[FEATURE_NAMES + ["target"]].dropna()
        X_all.append(df_clean[FEATURE_NAMES].values)
        y_all.append(df_clean["target"].values)
        idx_all.extend([ticker] * len(df_clean))

    if not X_all:
        raise ValueError("No se pudo construir el dataset. Verifica los tickers y el período.")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    return X, y, idx_all


def train(tickers: list[str], period: str, horizon: int, test_size: float) -> None:
    # ── Construir dataset ──────────────────────────────────────────
    logger.info("Construyendo dataset…")
    X, y, _ = build_dataset(tickers, period, horizon)
    logger.info(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} features, "
                f"balance UP={y.mean():.2%}")

    # ── Split temporal (shuffle=False para evitar data leakage) ───
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=None
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Pipeline: StandardScaler + RandomForest ────────────────────
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    logger.info("Entrenando pipeline (StandardScaler + RandomForestClassifier)…")
    pipeline.fit(X_train, y_train)

    # ── Evaluación ─────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    logger.info(f"\n{'='*50}")
    logger.info(f"Accuracy en test: {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"\nReporte de clasificación:\n{classification_report(y_test, y_pred, target_names=['DOWN','UP'])}")
    logger.info(f"Matriz de confusión:\n{confusion_matrix(y_test, y_pred)}")

    # Feature importances
    fi = pipeline.named_steps["clf"].feature_importances_
    for fname, importance in sorted(zip(FEATURE_NAMES, fi), key=lambda x: -x[1]):
        logger.info(f"  {fname:15s}: {importance:.4f}")

    # ── Serialización ──────────────────────────────────────────────
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_OUT)
    logger.info(f"\n✅ Modelo guardado en {MODEL_OUT}")
    logger.info("   Reinicia el backend para que MLModelSingleton cargue el modelo actualizado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena el modelo ML de RiskLab.")
    parser.add_argument("--tickers", nargs="+",
                        default=["MSI", "XOM", "JNJ", "PG", "UL", "TSM"])
    parser.add_argument("--period",   default="3y",
                        help="Período de datos (ej: 1y, 2y, 3y, 5y)")
    parser.add_argument("--horizon",  type=int, default=5,
                        help="Días a predecir en el futuro")
    parser.add_argument("--test-size", type=float, default=0.30,
                        help="Fracción de datos para test (shuffle=False)")
    args = parser.parse_args()

    try:
        train(args.tickers, args.period, args.horizon, args.test_size)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
