# =============================================================================
# backend/app/models/schemas.py – Modelos Pydantic de Request y Response
# Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez
# Proyecto Integrador – Teoría del Riesgo · USTA
# =============================================================================
from __future__ import annotations

import math
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────────────────────────

class VaRRequest(BaseModel):
    tickers:    list[str]   = Field(..., min_length=1, examples=[["MSI","XOM","JNJ","PG","UL","TSM"]])
    weights:    list[float] = Field(..., min_length=1)
    confidence: float       = Field(default=0.95, ge=0.90, le=0.999)
    period:     str         = Field(default="2y")

    @field_validator("tickers")
    @classmethod
    def tickers_upper(cls, v: list[str]) -> list[str]:
        return [t.strip().upper() for t in v]

    @field_validator("weights")
    @classmethod
    def weights_positive(cls, v: list[float]) -> list[float]:
        if any(w < 0 for w in v):
            raise ValueError("Todos los pesos deben ser no negativos.")
        return v

    @model_validator(mode="after")
    def check_lengths_and_sum(self) -> "VaRRequest":
        if len(self.tickers) != len(self.weights):
            raise ValueError("La cantidad de tickers debe coincidir con la de pesos.")
        total = sum(self.weights)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Los pesos deben sumar 1.0 (suman {total:.4f}).")
        return self


class FronteraRequest(BaseModel):
    tickers:      list[str] = Field(..., min_length=2)
    period:       str       = Field(default="2y")
    n_portfolios: int       = Field(default=10_000, ge=1_000, le=50_000)

    @field_validator("tickers")
    @classmethod
    def tickers_upper(cls, v: list[str]) -> list[str]:
        return [t.strip().upper() for t in v]


class OptionRequest(BaseModel):
    """Request para valoración Black-Scholes."""
    ticker: str   = Field(..., description="Ticker del subyacente (para referencia)")
    S:      float = Field(..., gt=0, description="Precio actual del subyacente (USD)")
    K:      float = Field(..., gt=0, description="Precio de ejercicio (strike)")
    T:      float = Field(..., gt=0, description="Tiempo al vencimiento en años")
    r:      float = Field(..., description="Tasa libre de riesgo anual (decimal)")
    sigma:  float = Field(..., gt=0, description="Volatilidad anual (decimal, ej: 0.25)")
    tipo:   str   = Field(..., description="'call' o 'put'")

    @field_validator("tipo")
    @classmethod
    def valid_tipo(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"call", "put"}:
            raise ValueError("tipo debe ser 'call' o 'put'.")
        return v

    @field_validator("ticker")
    @classmethod
    def ticker_upper(cls, v: str) -> str:
        return v.strip().upper()


class StressRequest(BaseModel):
    """Request para stress testing del portafolio."""
    tickers:    list[str]
    weights:    list[float]
    period:     str   = "2y"
    shock_tasa: float = Field(default=0.02,  description="Shock de tasa libre de riesgo (ej: 0.02 = +200 pb)")
    shock_vol:  float = Field(default=0.30,  description="Shock de volatilidad (ej: 0.30 = +30%)")
    shock_precio: float = Field(default=-0.20, description="Shock de precio (ej: -0.20 = caída de 20%)")

    @field_validator("tickers")
    @classmethod
    def upper(cls, v: list[str]) -> list[str]:
        return [t.strip().upper() for t in v]


class PredictRequest(BaseModel):
    ticker:        str = Field(..., description="Ticker a predecir")
    period:        str = Field(default="2y")
    horizon_days:  int = Field(default=5, ge=1, le=30)

    @field_validator("ticker")
    @classmethod
    def upper(cls, v: str) -> str:
        return v.strip().upper()


class BondRequest(BaseModel):
    """Request para valoración de renta fija."""
    face_value:   float = Field(default=1000.0, gt=0, description="Valor nominal")
    coupon_rate:  float = Field(..., ge=0, le=1, description="Tasa cupón anual (decimal)")
    maturity:     float = Field(..., gt=0, description="Años al vencimiento")
    ytm:          float = Field(..., gt=0, description="Yield to maturity (decimal)")
    frequency:    int   = Field(default=2, description="Pagos de cupón por año (2=semestral)")


# ─────────────────────────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────────────────────────

class PrecioItem(BaseModel):
    fecha:  str
    precio: float


class PreciosResponse(BaseModel):
    ticker:              str
    empresa:             str
    periodo:             str
    n_observaciones:     int
    precio_actual:       float
    variacion_diaria_pct: Optional[float]
    precios:             list[PrecioItem]


class RendimientosResponse(BaseModel):
    ticker:                     str
    empresa:                    str
    periodo:                    str
    media_diaria_pct:           float
    rendimiento_anualizado_pct: float
    volatilidad_diaria_pct:     float
    volatilidad_anualizada_pct: float
    skewness:                   float
    kurtosis:                   float
    min_diario_pct:             float
    max_diario_pct:             float
    n_observaciones:            int
    rendimientos:               list[dict]


class IndicadoresResponse(BaseModel):
    ticker:       str
    fecha_ultimo: str
    precio_actual: float
    rsi:          float
    macd:         float
    macd_signal:  float
    macd_hist:    float
    bb_upper:     float
    bb_middle:    float
    bb_lower:     float
    sma_20:       float
    sma_50:       float
    ema_20:       float
    stoch_k:      float
    stoch_d:      float


class VaRMethodResult(BaseModel):
    metodo:              str
    var_diario_pct:      float
    cvar_diario_pct:     float
    var_anualizado_pct:  float


class KupiecResult(BaseModel):
    violaciones_observadas: int
    violaciones_esperadas:  float
    tasa_violaciones_pct:   float
    tasa_esperada_pct:      float
    lr_statistic:           float
    p_valor:                float
    modelo_valido:          bool
    interpretacion_kupiec:  str


class VaRResponse(BaseModel):
    tickers:         list[str]
    pesos:           list[float]
    nivel_confianza: float
    resultados:      list[VaRMethodResult]
    kupiec:          KupiecResult


class CAPMAsset(BaseModel):
    ticker:                      str
    empresa:                     str
    beta:                        float
    alpha_diario:                float
    r_squared:                   float
    riesgo_sistematico_pct:      float
    riesgo_idiosincratico_pct:   float
    rendimiento_esperado_anual_pct: float
    clasificacion:               str


class CAPMResponse(BaseModel):
    rf_anual_pct:            float
    prima_mercado_anual_pct: float
    benchmark:               str
    activos:                 list[CAPMAsset]


class PortfolioOptimo(BaseModel):
    rendimiento_anual_pct:  float
    volatilidad_anual_pct:  float
    sharpe_ratio:           float
    pesos:                  dict[str, float]


class FronteraResponse(BaseModel):
    tickers:               list[str]
    n_portfolios_simulados: int
    max_sharpe:            PortfolioOptimo
    min_varianza:          PortfolioOptimo
    correlaciones:         dict


class AlertaActivo(BaseModel):
    ticker:        str
    empresa:       str
    indicadores:   dict[str, str]
    señal_global:  str
    votos_compra:  int
    votos_venta:   int
    interpretacion: str


class AlertasResponse(BaseModel):
    fecha_consulta: str
    alertas:        list[AlertaActivo]


class MacroResponse(BaseModel):
    fecha_consulta:  str
    rf_anual_pct:    float
    inflacion_us:    Optional[float] = None
    vix:             Optional[float] = None
    oro_usd:         Optional[float] = None
    brent_usd:       Optional[float] = None
    usd_cop:         Optional[float] = None
    dxy:             Optional[float] = None


class Greeks(BaseModel):
    delta: float
    gamma: float
    vega:  float
    theta: float
    rho:   float


class OptionResponse(BaseModel):
    ticker:      str
    tipo:        str
    S:           float
    K:           float
    T:           float
    r:           float
    sigma:       float
    precio:      float
    greeks:      Greeks
    paridad_put_call: Optional[float] = None


class YieldPoint(BaseModel):
    maturity: float
    yield_pct: float


class YieldCurveResponse(BaseModel):
    fecha_consulta:  str
    puntos:          list[YieldPoint]
    ns_params:       dict[str, float]
    ns_fitted:       list[YieldPoint]


class BondResponse(BaseModel):
    precio:              float
    duracion_macaulay:   float
    duracion_modificada: float
    convexidad:          float
    precio_shock_200pb:  float
    delta_precio_pct:    float


class StressResult(BaseModel):
    escenario:           str
    var_base_pct:        float
    perdida_estimada_pct: float
    descripcion:         str


class StressResponse(BaseModel):
    fecha_consulta: str
    portafolio:     dict[str, float]
    escenarios:     list[StressResult]


class PredictionResponse(BaseModel):
    ticker:        str
    horizon_days:  int
    model_version: str
    prediction_pct: float
    direction:     str
    confidence:    float
    features:      dict[str, float]
