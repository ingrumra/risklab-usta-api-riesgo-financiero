# =============================================================================
# backend/app/dependencies.py – Inyección de dependencias con Depends()
# Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez
# =============================================================================
from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.services.data import DataService
from app.services.risk import RiskCalculator
from app.services.portfolio import TechnicalIndicators, PortfolioAnalyzer
from app.services.derivatives import Bond, YieldCurve, OptionPricer, StressTester
from app.ml.model import MLModelSingleton


def get_settings_dep() -> Settings:
    """Dependencia de configuración."""
    return get_settings()


@lru_cache
def get_data_service() -> DataService:
    """Singleton de DataService."""
    return DataService()


@lru_cache
def get_risk_calculator() -> RiskCalculator:
    """Singleton de RiskCalculator."""
    return RiskCalculator()


@lru_cache
def get_technical_indicators() -> TechnicalIndicators:
    return TechnicalIndicators()


@lru_cache
def get_portfolio_analyzer() -> PortfolioAnalyzer:
    return PortfolioAnalyzer()


@lru_cache
def get_option_pricer() -> OptionPricer:
    return OptionPricer()


@lru_cache
def get_stress_tester() -> StressTester:
    return StressTester()


@lru_cache
def get_ml_model() -> MLModelSingleton:
    """
    Singleton del modelo ML.
    El decorador @lru_cache garantiza que MLModelSingleton.__new__
    solo se llama una vez; el patrón Singleton lo refuerza internamente.
    """
    return MLModelSingleton()
