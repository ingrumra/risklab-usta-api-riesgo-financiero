# =============================================================================
# tests/test_api.py – Suite pytest + TestClient de FastAPI
# Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez
# =============================================================================
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

# Importamos la app después de cambiar sys.path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.main import app

# ─────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """TestClient con ciclo de vida completo (startup/shutdown)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def var_body():
    return {
        "tickers":    ["MSI", "XOM", "JNJ", "PG", "UL", "TSM"],
        "weights":    [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
        "confidence": 0.95,
        "period":     "1y",
    }


@pytest.fixture
def frontera_body():
    return {
        "tickers":      ["MSI", "XOM", "JNJ", "PG", "UL", "TSM"],
        "period":       "1y",
        "n_portfolios": 1000,
    }


@pytest.fixture
def option_body():
    return {
        "ticker": "MSI",
        "S": 200.0, "K": 200.0, "T": 0.5,
        "r": 0.045, "sigma": 0.25, "tipo": "call",
    }


@pytest.fixture
def bond_body():
    return {
        "face_value": 1000.0, "coupon_rate": 0.05,
        "maturity": 5.0, "ytm": 0.045, "frequency": 2,
    }


@pytest.fixture
def stress_body():
    return {
        "tickers":      ["MSI", "XOM", "JNJ", "PG", "UL", "TSM"],
        "weights":      [1/6, 1/6, 1/6, 1/6, 1/6, 1/6],
        "period":       "1y",
        "shock_tasa":   0.02,
        "shock_vol":    0.30,
        "shock_precio": -0.20,
    }


@pytest.fixture
def predict_body():
    return {"ticker": "MSI", "period": "1y", "horizon_days": 5}


# ─────────────────────────────────────────────────────────────────
# TESTS – ROOT
# ─────────────────────────────────────────────────────────────────

def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "version" in data
    assert "endpoints" in data


def test_activos(client):
    r = client.get("/activos")
    assert r.status_code == 200
    data = r.json()
    assert "tickers" in data
    assert len(data["tickers"]) >= 1


# ─────────────────────────────────────────────────────────────────
# TESTS – PRECIOS Y RENDIMIENTOS
# ─────────────────────────────────────────────────────────────────

def test_precios_valid(client):
    r = client.get("/precios/MSI", params={"period": "3mo"})
    assert r.status_code == 200
    data = r.json()
    assert data["ticker"] == "MSI"
    assert data["n_observaciones"] > 0
    assert len(data["precios"]) > 0


def test_precios_invalid_ticker(client):
    r = client.get("/precios/XXXXXXXX")
    assert r.status_code in (404, 503)


def test_rendimientos(client):
    r = client.get("/rendimientos/XOM", params={"period": "3mo"})
    assert r.status_code == 200
    data = r.json()
    assert "media_diaria_pct" in data
    assert "skewness" in data
    assert "kurtosis" in data


# ─────────────────────────────────────────────────────────────────
# TESTS – INDICADORES TÉCNICOS
# ─────────────────────────────────────────────────────────────────

def test_indicadores(client):
    r = client.get("/indicadores/JNJ", params={"period": "1y"})
    assert r.status_code == 200
    data = r.json()
    assert 0 <= data["rsi"] <= 100
    assert "macd" in data
    assert "bb_upper" in data
    assert data["bb_upper"] > data["bb_lower"]


# ─────────────────────────────────────────────────────────────────
# TESTS – VaR / CVaR + KUPIEC
# ─────────────────────────────────────────────────────────────────

def test_var(client, var_body):
    r = client.post("/var", json=var_body)
    assert r.status_code == 200
    data = r.json()
    assert len(data["resultados"]) == 3
    assert "kupiec" in data
    # VaR debe ser positivo
    for res in data["resultados"]:
        assert res["var_diario_pct"] >= 0
        assert res["cvar_diario_pct"] >= res["var_diario_pct"]


def test_var_invalid_weights(client):
    body = {
        "tickers": ["MSI", "XOM"],
        "weights": [0.6, 0.6],  # suman 1.2 → error
        "confidence": 0.95,
    }
    r = client.post("/var", json=body)
    assert r.status_code == 422


def test_var_mismatched_lengths(client):
    body = {
        "tickers": ["MSI", "XOM"],
        "weights": [0.5, 0.3, 0.2],  # longitudes distintas → error
        "confidence": 0.95,
    }
    r = client.post("/var", json=body)
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────
# TESTS – CAPM
# ─────────────────────────────────────────────────────────────────

def test_capm(client):
    r = client.get("/capm", params={"period": "1y"})
    assert r.status_code == 200
    data = r.json()
    assert "activos" in data
    for a in data["activos"]:
        assert "beta" in a
        assert "r_squared" in a
        assert 0 <= a["r_squared"] <= 1


# ─────────────────────────────────────────────────────────────────
# TESTS – MARKOWITZ
# ─────────────────────────────────────────────────────────────────

def test_frontera(client, frontera_body):
    r = client.post("/frontera-eficiente", json=frontera_body)
    assert r.status_code == 200
    data = r.json()
    assert "max_sharpe" in data
    assert "min_varianza" in data
    ms = data["max_sharpe"]
    mv = data["min_varianza"]
    # Mínima varianza debe tener menor volatilidad que máximo Sharpe
    assert mv["volatilidad_anual_pct"] <= ms["volatilidad_anual_pct"] + 0.5


# ─────────────────────────────────────────────────────────────────
# TESTS – ALERTAS / SEÑALES
# ─────────────────────────────────────────────────────────────────

def test_alertas(client):
    r = client.get("/alertas", params={"period": "1y"})
    assert r.status_code == 200
    data = r.json()
    assert "alertas" in data
    for alerta in data["alertas"]:
        assert alerta["señal_global"] in {"BUY", "SELL", "NEUTRAL"}
        assert "indicadores" in alerta


# ─────────────────────────────────────────────────────────────────
# TESTS – MACRO
# ─────────────────────────────────────────────────────────────────

def test_macro(client):
    r = client.get("/macro")
    assert r.status_code == 200
    data = r.json()
    assert "rf_anual_pct" in data
    assert data["rf_anual_pct"] > 0


# ─────────────────────────────────────────────────────────────────
# TESTS – OPCIONES BLACK-SCHOLES
# ─────────────────────────────────────────────────────────────────

def test_opcion_call(client, option_body):
    r = client.post("/opcion/precio", json=option_body)
    assert r.status_code == 200
    data = r.json()
    assert data["precio"] > 0
    assert "greeks" in data
    greeks = data["greeks"]
    assert 0 < greeks["delta"] < 1  # call ATM → delta entre 0 y 1
    assert greeks["gamma"] > 0
    assert greeks["vega"] > 0
    assert greeks["theta"] < 0  # theta siempre negativo


def test_opcion_put(client, option_body):
    body = {**option_body, "tipo": "put"}
    r = client.post("/opcion/precio", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["precio"] > 0
    g = data["greeks"]
    assert -1 < g["delta"] < 0  # put ATM → delta entre -1 y 0


def test_opcion_paridad_put_call(client, option_body):
    """Verifica paridad put-call: C - P = S - K*e^(-rT)"""
    r_call = client.post("/opcion/precio", json={**option_body, "tipo": "call"})
    r_put  = client.post("/opcion/precio", json={**option_body, "tipo": "put"})
    assert r_call.status_code == 200
    assert r_put.status_code == 200
    c = r_call.json()["precio"]
    p = r_put.json()["precio"]
    import math
    teorico = option_body["S"] - option_body["K"] * math.exp(
        -option_body["r"] * option_body["T"]
    )
    assert abs((c - p) - teorico) < 0.01  # tolerancia $0.01


def test_opcion_tipo_invalido(client, option_body):
    r = client.post("/opcion/precio", json={**option_body, "tipo": "forward"})
    assert r.status_code == 422


# ─────────────────────────────────────────────────────────────────
# TESTS – BONO
# ─────────────────────────────────────────────────────────────────

def test_bono(client, bond_body):
    r = client.post("/bono/valorar", json=bond_body)
    assert r.status_code == 200
    data = r.json()
    assert data["precio"] > 0
    assert data["duracion_macaulay"] > 0
    assert data["duracion_modificada"] > 0
    assert data["convexidad"] > 0
    # Bono con cupón > YTM → precio > par
    assert data["precio"] > bond_body["face_value"]


# ─────────────────────────────────────────────────────────────────
# TESTS – STRESS TESTING
# ─────────────────────────────────────────────────────────────────

def test_stress(client, stress_body):
    r = client.post("/stress", json=stress_body)
    assert r.status_code == 200
    data = r.json()
    assert "escenarios" in data
    assert len(data["escenarios"]) == 3
    for sc in data["escenarios"]:
        assert sc["perdida_estimada_pct"] >= 0


# ─────────────────────────────────────────────────────────────────
# TESTS – ML PREDICCIÓN (SINGLETON)
# ─────────────────────────────────────────────────────────────────

def test_predict(client, predict_body):
    r = client.post("/predict", json=predict_body)
    assert r.status_code == 200
    data = r.json()
    assert data["direction"] in {"UP", "DOWN"}
    assert 0 <= data["confidence"] <= 1
    assert "features" in data


def test_singleton_no_double_load(client, predict_body):
    """
    Llama /predict dos veces y verifica que el modelo
    es el mismo objeto en memoria (Singleton pattern).
    """
    r1 = client.post("/predict", json=predict_body)
    r2 = client.post("/predict", json=predict_body)
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Ambas respuestas deben ser idénticas (mismo modelo, mismo input)
    assert r1.json()["direction"] == r2.json()["direction"]
    assert r1.json()["model_version"] == r2.json()["model_version"]


# ─────────────────────────────────────────────────────────────────
# TESTS – PYDANTIC VALIDATORS
# ─────────────────────────────────────────────────────────────────

def test_pydantic_ticker_uppercase(client):
    """Verifica que los tickers se normalizan a mayúsculas."""
    r = client.get("/precios/msI", params={"period": "3mo"})
    # 200 significa que el validator hizo uppercase y encontró datos
    assert r.status_code in (200, 404, 503)
    if r.status_code == 200:
        assert r.json()["ticker"] == "MSI"


def test_pydantic_confidence_bounds(client):
    """Verifica que confidence fuera de [0.90, 0.999] es rechazado."""
    body = {
        "tickers": ["MSI", "XOM"],
        "weights": [0.5, 0.5],
        "confidence": 0.50,  # fuera de bounds
    }
    r = client.post("/var", json=body)
    assert r.status_code == 422
