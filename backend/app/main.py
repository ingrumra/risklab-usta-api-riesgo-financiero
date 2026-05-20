# =============================================================================
# backend/app/main.py – Aplicación FastAPI principal
# Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez
# Proyecto Integrador – Teoría del Riesgo · USTA
# =============================================================================
from __future__ import annotations

import datetime
import logging
import math

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from yfinance import ticker

from app.config import Settings
from app.db.database import create_tables, get_db, PredictionLog, StressLog
from app.dependencies import (
    get_data_service,
    get_ml_model,
    get_option_pricer,
    get_portfolio_analyzer,
    get_risk_calculator,
    get_settings_dep,
    get_stress_tester,
)
from app.ml.model import MLModelSingleton, extract_features
from app.models.schemas import (
    AlertaActivo,
    AlertasResponse,
    BondRequest,
    BondResponse,
    CAPMAsset,
    CAPMResponse,
    FronteraRequest,
    FronteraResponse,
    Greeks,
    IndicadoresResponse,
    KupiecResult,
    MacroResponse,
    OptionRequest,
    OptionResponse,
    PortfolioOptimo,
    PrecioItem,
    PreciosResponse,
    PredictRequest,
    PredictionResponse,
    RendimientosResponse,
    StressRequest,
    StressResponse,
    StressResult,
    VaRMethodResult,
    VaRRequest,
    VaRResponse,
    YieldCurveResponse,
    YieldPoint,
)
from app.services.data import DataService
from app.services.derivatives import Bond, OptionPricer, StressTester, YieldCurve
from app.services.portfolio import PortfolioAnalyzer
from app.services.risk import RiskCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# INICIALIZACIÓN
# ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RiskLab USTA – API de Riesgo Financiero",
    description=(
        "Backend del Proyecto Integrador de Teoría del Riesgo.\n\n"
        "Proporciona endpoints para análisis técnico, rendimientos, ARCH/GARCH, "
        "VaR (3 métodos + Kupiec), CAPM, Markowitz, renta fija (Nelson-Siegel), "
        "opciones (Black-Scholes + Greeks), stress testing y predicciones ML.\n\n"
        "**Autoras:** Alejandra Sepúlveda · Ingrid Umbacia Ramírez\n\n"
        "**Universidad Santo Tomás · Facultad de Estadística**"
    ),
    version="2.0.0",
    contact={"name": "Alejandra & Ingrid", "email": "estudiantes@usta.edu.co"},
    license_info={"name": "Académico – USTA 2025"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "CVX":  "Chevron",
    "JNJ":  "Johnson & Johnson",
    "PG":   "Procter & Gamble",
    "MSFT": "Microsoft",
    "TSM":  "TSMC",
}


@app.on_event("startup")
async def startup_event() -> None:
    """Crea las tablas SQLite al iniciar la app."""
    create_tables()
    logger.info("✅ Base de datos inicializada.")
    # Pre-carga del modelo ML (Singleton)
    get_ml_model()
    logger.info("✅ Modelo ML cargado en memoria.")


# ─────────────────────────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Info"])
async def root() -> dict:
    return {
        "proyecto":  "RiskLab USTA – Teoría del Riesgo v2.0",
        "autoras":   ["Alejandra Sepúlveda", "Ingrid Umbacia Ramírez"],
        "version":   "2.0.0",
        "docs":      "/docs",
        "redoc":     "/redoc",
        "endpoints": [
            "/activos", "/precios/{ticker}", "/rendimientos/{ticker}",
            "/indicadores/{ticker}", "/var", "/capm",
            "/frontera-eficiente", "/alertas", "/macro",
            "/curva-rendimiento", "/bono/valorar",
            "/opcion/precio", "/stress", "/predict",
        ],
    }


# ─────────────────────────────────────────────────────────────────
# ACTIVOS
# ─────────────────────────────────────────────────────────────────

@app.get("/activos", tags=["Portafolio"])
async def get_activos(settings: Settings = Depends(get_settings_dep)) -> dict:
    return {
        "tickers":      settings.tickers,
        "benchmark":    settings.benchmark,
        "empresas":     {t: NAMES.get(t, t) for t in settings.tickers},
        "total_activos": len(settings.tickers),
    }


# ─────────────────────────────────────────────────────────────────
# PRECIOS
# ─────────────────────────────────────────────────────────────────

@app.get("/precios/{ticker}", response_model=PreciosResponse, tags=["Precios"])
async def get_precios(
    ticker: str,
    period: str = "2y",
    data_svc: DataService = Depends(get_data_service),
) -> PreciosResponse:
    ticker = ticker.upper().strip()
    try:
        prices = data_svc.get_prices(ticker, period)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    var_diaria = None
    if len(prices) >= 2:
        var_diaria = round((float(prices.iloc[-1]) / float(prices.iloc[-2]) - 1) * 100, 4)

    return PreciosResponse(
        ticker=ticker,
        empresa=NAMES.get(ticker, ticker),
        periodo=period,
        n_observaciones=len(prices),
        precio_actual=round(float(prices.iloc[-1]), 4),
        variacion_diaria_pct=var_diaria,
        precios=[
            PrecioItem(fecha=str(idx.date()), precio=round(float(val), 4))
            for idx, val in zip(prices.index, prices.values)
        ],
    )


# ─────────────────────────────────────────────────────────────────
# RENDIMIENTOS
# ─────────────────────────────────────────────────────────────────

@app.get("/rendimientos/{ticker}", response_model=RendimientosResponse, tags=["Rendimientos"])
async def get_rendimientos(
    ticker: str,
    period: str = "2y",
    data_svc: DataService = Depends(get_data_service),
    calc:     RiskCalculator = Depends(get_risk_calculator),
) -> RendimientosResponse:
    ticker = ticker.upper().strip()
    try:
        prices = data_svc.get_prices(ticker, period)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    # ✅ FIX: Pylance ve que prices puede ser None si get_prices falla silenciosamente
    if not isinstance(prices, pd.Series) or prices.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No se obtuvieron datos para '{ticker}'."
        )

    from scipy import stats as sp_stats

    # ✅ FIX: cast explícito de lr a pd.Series antes de cualquier operación
    lr: pd.Series = pd.Series(calc.log_returns(prices))

    media       = float(lr.mean())
    volatilidad = float(lr.std())
    minimo      = float(lr.min())
    maximo      = float(lr.max())

    return RendimientosResponse(
        ticker=ticker,
        empresa=NAMES.get(ticker, ticker),
        periodo=period,
        media_diaria_pct=round(media * 100, 6),
        rendimiento_anualizado_pct=round(media * 252 * 100, 4),
        volatilidad_diaria_pct=round(volatilidad * 100, 6),
        volatilidad_anualizada_pct=round(volatilidad * (252 ** 0.5) * 100, 4),
        skewness=round(float(sp_stats.skew(lr)), 4),
        kurtosis=round(float(sp_stats.kurtosis(lr)), 4),
        min_diario_pct=round(minimo * 100, 4),
        max_diario_pct=round(maximo * 100, 4),
        n_observaciones=len(lr),
        rendimientos=[
            {"fecha": str(idx.date()), "log_ret_pct": round(float(val) * 100, 6)}
            for idx, val in zip(lr.index, lr.values)
        ],
    )


# ─────────────────────────────────────────────────────────────────
# INDICADORES TÉCNICOS
# ─────────────────────────────────────────────────────────────────

@app.get("/indicadores/{ticker}", response_model=IndicadoresResponse, tags=["Análisis Técnico"])
async def get_indicadores(
    ticker: str,
    period: str = "1y",
    data_svc: DataService = Depends(get_data_service),
) -> IndicadoresResponse:
    ticker = ticker.upper().strip()
    try:
        prices = data_svc.get_prices(ticker, period)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    if len(prices) < 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insuficientes datos ({len(prices)} obs.). Mínimo: 50.",
        )

    from app.services.portfolio import TechnicalIndicators
    ind = TechnicalIndicators.all_indicators(prices)
    return IndicadoresResponse(ticker=ticker, fecha_ultimo=str(prices.index[-1].date()), **ind)


# ─────────────────────────────────────────────────────────────────
# VaR / CVaR + KUPIEC
# ─────────────────────────────────────────────────────────────────

@app.post("/var", response_model=VaRResponse, tags=["Riesgo"])
async def calcular_var(
    req:      VaRRequest,
    data_svc: DataService = Depends(get_data_service),
    calc:     RiskCalculator = Depends(get_risk_calculator),
) -> VaRResponse:
    try:
        prices = data_svc.get_multiple_prices(req.tickers, req.period)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    weights = np.array(req.weights)

   # ✅ FIX: cast explícito en cada paso para que Pylance infiera los tipos
    log_ret:  pd.DataFrame = pd.DataFrame(calc.log_returns(prices))
    port_ret: pd.Series    = pd.Series(log_ret.dot(weights))

    n      = len(port_ret)
    split  = max(int(0.7 * n), 50)
    train: pd.Series = pd.Series(port_ret.iloc[:split])
    test:  pd.Series = pd.Series(port_ret.iloc[split:])

    v_par,  cv_par  = calc.var_parametric(train, req.confidence)
    v_hist, cv_hist = calc.var_historical(train, req.confidence)
    v_mc,   cv_mc   = calc.var_montecarlo(train, req.confidence)
    kupiec_raw = calc.kupiec_test(test, v_hist, req.confidence)

    s252 = 252 ** 0.5
    return VaRResponse(
        tickers=req.tickers,
        pesos=req.weights,
        nivel_confianza=req.confidence,
        resultados=[
            VaRMethodResult(metodo="Paramétrico (Normal)",
                            var_diario_pct=round(v_par * 100, 4),
                            cvar_diario_pct=round(cv_par * 100, 4),
                            var_anualizado_pct=round(v_par * s252 * 100, 4)),
            VaRMethodResult(metodo="Histórico",
                            var_diario_pct=round(v_hist * 100, 4),
                            cvar_diario_pct=round(cv_hist * 100, 4),
                            var_anualizado_pct=round(v_hist * s252 * 100, 4)),
            VaRMethodResult(metodo="Montecarlo (10,000 sim.)",
                            var_diario_pct=round(v_mc * 100, 4),
                            cvar_diario_pct=round(cv_mc * 100, 4),
                            var_anualizado_pct=round(v_mc * s252 * 100, 4)),
        ],
        kupiec=KupiecResult(**kupiec_raw),
    )


# ─────────────────────────────────────────────────────────────────
# CAPM
# ─────────────────────────────────────────────────────────────────

@app.get("/capm", response_model=CAPMResponse, tags=["CAPM"])
async def get_capm(
    period:   str = "2y",
    data_svc: DataService = Depends(get_data_service),
    analyzer: PortfolioAnalyzer = Depends(get_portfolio_analyzer),
    settings: Settings = Depends(get_settings_dep),
) -> CAPMResponse:
    import traceback
    try:
        prices = data_svc.get_multiple_prices(settings.tickers, period)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error precios: {str(e)}")

    try:
        bench_ticker = yf.Ticker(settings.benchmark)
        bench_raw = bench_ticker.history(period=period)

        # ✅ FIX línea 345: cast explícito a pd.Series desde el inicio
        # evita el error de __getitem__ con tuple slice
        bench_close = bench_raw["Close"]
        if not isinstance(bench_close, pd.Series):
            bench_close = pd.Series(bench_close.values[:, 0], index=bench_close.index)
        bench_prices: pd.Series = bench_close.dropna()

        # ✅ FIX línea 362: tz_localize sobre Index, no sobre Series
        # dropna() sobre NDArray no tiene .dropna — hay que hacerlo antes
        bench_prices.index = pd.DatetimeIndex(
            pd.to_datetime(bench_prices.index).tz_localize(None)
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Error benchmark: {str(e)}")

    try:
        macro     = data_svc.get_macro()
        rf_annual = float(macro.get("rf_annual") or 0.045)
        rf_daily  = rf_annual / 252
        logger.info(f"PRICES INDEX sample: {prices.index[:3].tolist()}")
        logger.info(f"BENCH INDEX sample: {bench_prices.index[:3].tolist()}")
        logger.info(f"PRICES dtype: {prices.index.dtype}")
        logger.info(f"BENCH dtype: {bench_prices.index.dtype}")
        results_raw = analyzer.capm(prices, bench_prices, rf_daily)

        # ✅ FIX línea 569: cast explícito a pd.Series antes de .dropna()
        bench_ret: pd.Series = pd.Series(
            np.log(bench_prices / bench_prices.shift(1))
        ).dropna()
        prima_pct = round((float(bench_ret.mean()) - rf_daily) * 252 * 100, 4)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error calculo CAPM: {traceback.format_exc()}"
        )

    return CAPMResponse(
        rf_anual_pct=round(rf_annual * 100, 4),
        prima_mercado_anual_pct=prima_pct,
        benchmark=settings.benchmark,
        activos=[CAPMAsset(**r) for r in results_raw],
    )


# ─────────────────────────────────────────────────────────────────
# FRONTERA EFICIENTE – MARKOWITZ
# ─────────────────────────────────────────────────────────────────

@app.post("/frontera-eficiente", response_model=FronteraResponse, tags=["Markowitz"])
async def get_frontera(
    req:      FronteraRequest,
    data_svc: DataService = Depends(get_data_service),
    analyzer: PortfolioAnalyzer = Depends(get_portfolio_analyzer),
) -> FronteraResponse:
    try:
        prices = data_svc.get_multiple_prices(req.tickers, req.period)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    macro  = data_svc.get_macro()
    rf     = float(macro.get("rf_annual") or 0.045)
    result = analyzer.simulate_frontier(prices, rf=rf, n_portfolios=req.n_portfolios)

    return FronteraResponse(
        tickers=req.tickers,
        n_portfolios_simulados=req.n_portfolios,
        max_sharpe=PortfolioOptimo(**result["max_sharpe"]),
        min_varianza=PortfolioOptimo(**result["min_varianza"]),
        correlaciones=result["correlaciones"],
    )


# ─────────────────────────────────────────────────────────────────
# SEÑALES / ALERTAS
# ─────────────────────────────────────────────────────────────────

@app.get("/alertas", response_model=AlertasResponse, tags=["Señales"])
async def get_alertas(
    period:   str = "1y",
    rsi_ob:   int = 70,
    rsi_os:   int = 30,
    stoch_ob: int = 80,
    stoch_os: int = 20,
    data_svc: DataService = Depends(get_data_service),
    analyzer: PortfolioAnalyzer = Depends(get_portfolio_analyzer),
    settings: Settings = Depends(get_settings_dep),
) -> AlertasResponse:
    try:
        prices = data_svc.get_multiple_prices(settings.tickers, period)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    signals = analyzer.generate_signals(prices, rsi_ob, rsi_os, stoch_ob, stoch_os)
    return AlertasResponse(
        fecha_consulta=str(datetime.date.today()),
        alertas=[AlertaActivo(**s) for s in signals],
    )


# ─────────────────────────────────────────────────────────────────
# MACRO
# ─────────────────────────────────────────────────────────────────

@app.get("/macro", response_model=MacroResponse, tags=["Macro"])
async def get_macro(data_svc: DataService = Depends(get_data_service)) -> MacroResponse:
    try:
        macro = data_svc.get_macro()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"Error al obtener datos macro: {e}")
    return MacroResponse(
        fecha_consulta=str(datetime.date.today()),
        rf_anual_pct=round(float(macro.get("rf_annual") or 0.045) * 100, 4),
        inflacion_us=macro.get("inflation_us"),
        vix=macro.get("vix"),
        oro_usd=macro.get("gold"),
        brent_usd=macro.get("oil"),
        usd_cop=macro.get("usdcop"),
        dxy=macro.get("dxy"),
    )


# ─────────────────────────────────────────────────────────────────
# CURVA DE RENDIMIENTO + NELSON-SIEGEL ★
# ─────────────────────────────────────────────────────────────────

@app.get("/curva-rendimiento", response_model=YieldCurveResponse, tags=["Renta Fija"])
async def get_curva_rendimiento(
    data_svc: DataService = Depends(get_data_service),
) -> YieldCurveResponse:
    """
    Obtiene puntos de la curva de tesoros US (FRED o Yahoo Finance)
    y ajusta el modelo Nelson-Siegel.
    """
    try:
        raw = data_svc.get_yield_curve()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    mats   = raw["maturities"]
    yields = raw["yields_pct"]

    if len(mats) < 3:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Insuficientes puntos de la curva (mínimo 3).")

    yc = YieldCurve()
    try:
        yc.fit_nelson_siegel(mats, yields)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error en ajuste Nelson-Siegel: {e}")

    return YieldCurveResponse(
        fecha_consulta=str(datetime.date.today()),
        puntos=[YieldPoint(maturity=m, yield_pct=y) for m, y in zip(mats, yields)],
        ns_params={**yc.params, "lambda": yc.params.get("lam", 0)},
        ns_fitted=[YieldPoint(maturity=p["maturity"], yield_pct=p["yield_pct"])
                   for p in yc.fitted_curve()],
    )


# ─────────────────────────────────────────────────────────────────
# VALORACIÓN DE BONOS ★
# ─────────────────────────────────────────────────────────────────


@app.post("/bono/valorar", response_model=BondResponse, tags=["Renta Fija"])
async def valorar_bono(req: BondRequest) -> BondResponse:
    """Valoración de bono con cupón: precio, duración, convexidad y shock ±200 pb."""
    bond = Bond(req.face_value, req.coupon_rate, req.maturity, req.ytm, req.frequency)
    precio_base = bond.precio()
    precio_shock = bond.precio_shock(0.02)  # +200 pb
    delta_pct = (precio_shock / precio_base - 1) * 100

    return BondResponse(
        precio=round(precio_base, 4),
        duracion_macaulay=round(bond.macaulay_duration(), 4),
        duracion_modificada=round(bond.modified_duration(), 4),
        convexidad=round(bond.convexity(), 4),
        precio_shock_200pb=round(precio_shock, 4),
        delta_precio_pct=round(delta_pct, 4),
    )


# ─────────────────────────────────────────────────────────────────
# OPCIONES BLACK-SCHOLES + GREEKS ★
# ─────────────────────────────────────────────────────────────────

@app.post("/opcion/precio", response_model=OptionResponse, tags=["Opciones"])
async def valorar_opcion(
    req:     OptionRequest,
    pricer:  OptionPricer = Depends(get_option_pricer),
) -> OptionResponse:
    """Valoración Black-Scholes de opción europea (call/put) + 5 Greeks + paridad put-call."""
    precio = pricer.black_scholes(req.S, req.K, req.T, req.r, req.sigma, req.tipo)
    greeks_dict = pricer.greeks(req.S, req.K, req.T, req.r, req.sigma, req.tipo)
    paridad = pricer.paridad_put_call(req.S, req.K, req.T, req.r, req.sigma)

    return OptionResponse(
        ticker=req.ticker,
        tipo=req.tipo,
        S=req.S,
        K=req.K,
        T=req.T,
        r=req.r,
        sigma=req.sigma,
        precio=round(precio, 4),
        greeks=Greeks(**greeks_dict),
        paridad_put_call=paridad if req.tipo == "call" else None,
    )


# ─────────────────────────────────────────────────────────────────
# STRESS TESTING ★
# ─────────────────────────────────────────────────────────────────

@app.post("/stress", response_model=StressResponse, tags=["Stress Testing"])
async def stress_test(
    req:      StressRequest,
    data_svc: DataService = Depends(get_data_service),
    calc:     RiskCalculator = Depends(get_risk_calculator),
    db:       Session = Depends(get_db),
) -> StressResponse:
    """
    Stress testing del portafolio bajo tres escenarios:
    shock de tasa, shock de volatilidad y crash de precio.
    Los resultados se persisten en SQLite.
    """
    try:
        prices = data_svc.get_multiple_prices(req.tickers, req.period)
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    weights = np.array(req.weights)

    # ✅ FIX: cast explícito para que Pylance no infiera np_ndarray
    log_ret:  pd.DataFrame = pd.DataFrame(calc.log_returns(prices))
    port_ret: pd.Series    = pd.Series(log_ret.dot(weights))

    tester = StressTester()
    scenarios_raw = tester.run(port_ret, req.shock_tasa, req.shock_vol, req.shock_precio)

    # Persistir en BD
    for sc in scenarios_raw:
        db.add(StressLog(
            scenario_name=sc["escenario"],
            portfolio_var=sc["var_base_pct"],
            max_loss_pct=sc["perdida_estimada_pct"],
        ))
    db.commit()

    return StressResponse(
        fecha_consulta=str(datetime.date.today()),
        portafolio={t: round(float(w) * 100, 2) for t, w in zip(req.tickers, weights)},
        escenarios=[StressResult(**s) for s in scenarios_raw],
    )


# ─────────────────────────────────────────────────────────────────
# ML – PREDICCIÓN ★★
# ─────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse, tags=["Machine Learning"])
async def predict(
    req:      PredictRequest,
    data_svc: DataService = Depends(get_data_service),
    ml_model: MLModelSingleton = Depends(get_ml_model),
    db:       Session = Depends(get_db),
) -> PredictionResponse:
    """
    Predicción de dirección de rendimiento (UP/DOWN) usando el modelo ML.
    Patrón Singleton: el modelo se carga una sola vez.
    """
    try:
        prices = data_svc.get_prices(req.ticker, req.period)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))

    features  = extract_features(prices, req.ticker)
    result    = ml_model.predict(features)
    pred_ret  = (0.015 if result["direction"] == "UP" else -0.012) * req.horizon_days

    # Persistir en BD
    import json
    db.add(PredictionLog(
        model_version=result.get("model_version", "v1"),
        ticker=req.ticker,
        horizon_days=req.horizon_days,
        features_json=json.dumps({k: round(float(v), 6) for k, v in features.items()}),
        prediction=pred_ret,
        direction=result["direction"],
        confidence=result["confidence"],
    ))
    db.commit()

    return PredictionResponse(
        ticker=req.ticker,
        horizon_days=req.horizon_days,
        model_version="v1",
        prediction_pct=round(pred_ret * 100, 4),
        direction=result["direction"],
        confidence=result["confidence"],
        features={k: round(float(v), 6) for k, v in features.items()},
    )
