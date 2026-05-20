# =============================================================================
# backend/app/services/risk.py – Cálculos de riesgo financiero
# Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez
# =============================================================================
from __future__ import annotations

import math
import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class RiskCalculator:
    """
    Encapsula todos los cálculos de riesgo financiero:
    log-rendimientos, VaR, CVaR, backtesting Kupiec, EWMA.
    """

    @staticmethod
    def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Calcula log-rendimientos a partir de precios."""
        return np.log(prices / prices.shift(1)).dropna()

    # ── VaR ──────────────────────────────────────────────────────

    @staticmethod
    def var_parametric(returns: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
        """VaR paramétrico bajo distribución normal."""
        mu, sigma = float(returns.mean()), float(returns.std())
        z = stats.norm.ppf(1 - confidence)
        var = -(mu + z * sigma)
        cvar = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence))
        return max(var, 0.0), max(cvar, 0.0)

    @staticmethod
    def var_historical(returns: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
        """VaR histórico (percentil empírico)."""
        var  = -float(returns.quantile(1 - confidence))
        cvar = -float(returns[returns <= returns.quantile(1 - confidence)].mean())
        return max(var, 0.0), max(cvar, 0.0)

    @staticmethod
    def var_montecarlo(returns: pd.Series, confidence: float = 0.95,
                       n_sim: int = 10_000) -> tuple[float, float]:
        """VaR Montecarlo (10,000 simulaciones por defecto)."""
        mu, sigma = float(returns.mean()), float(returns.std())
        np.random.seed(42)
        sims = np.random.normal(mu, sigma, n_sim)
        var  = -float(np.percentile(sims, (1 - confidence) * 100))
        cvar = -float(sims[sims <= np.percentile(sims, (1 - confidence) * 100)].mean())
        return max(var, 0.0), max(cvar, 0.0)

    @staticmethod
    def kupiec_test(test_returns: pd.Series, var: float,
                    confidence: float = 0.95) -> dict:
        """
        Test de Kupiec (Proportion of Failures).
        LR ~ chi²(1). H0: tasa de violaciones = 1 - confidence.
        """
        T = len(test_returns)
        alpha = 1 - confidence
        violations = int((test_returns < -var).sum())
        p_hat = violations / T if T > 0 else 0.0

        if violations == 0:
            lr = 2 * T * math.log(1 / (1 - alpha)) if alpha < 1 else 0.0
        elif violations == T:
            lr = 2 * T * math.log(1 / alpha) if alpha > 0 else 0.0
        else:
            try:
                lr = -2 * (
                    violations * math.log(alpha / p_hat)
                    + (T - violations) * math.log((1 - alpha) / (1 - p_hat))
                )
            except (ZeroDivisionError, ValueError):
                lr = 0.0

        p_val = float(1 - stats.chi2.cdf(lr, df=1))
        valid = p_val > 0.05

        return {
            "violaciones_observadas":  violations,
            "violaciones_esperadas":   round(alpha * T, 2),
            "tasa_violaciones_pct":    round(p_hat * 100, 4),
            "tasa_esperada_pct":       round(alpha * 100, 4),
            "lr_statistic":            round(lr, 6),
            "p_valor":                 round(p_val, 6),
            "modelo_valido":           valid,
            "interpretacion_kupiec":   (
                f"✅ Test de Kupiec: p-valor={p_val:.4f} > 0.05 → El modelo VaR es estadísticamente válido."
                if valid else
                f"❌ Test de Kupiec: p-valor={p_val:.4f} ≤ 0.05 → El número de violaciones ({violations}) "
                f"es estadísticamente inusual. Considerar VaR dinámico con GARCH."
            ),
        }

    # ── EWMA ────────────────────────────────────────────────────

    @staticmethod
    def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
        """
        Volatilidad EWMA (RiskMetrics λ=0.94 para datos diarios).
        σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}
        """
        var_series = returns.ewm(alpha=1 - lam, adjust=False).var()
        return np.sqrt(var_series)

    # ── Correlaciones y covarianza ────────────────────────────────

    @staticmethod
    def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        return returns.corr()


class VolatilityModel:
    """
    Clase polimórfica para modelos de volatilidad: ARCH/GARCH.
    Métodos .fit() y .forecast() compartidos (Semana 2 - POO).
    """

    def __init__(self, model_name: str = "GARCH(1,1)"):
        self.model_name = model_name
        self._result = None

    def fit(self, returns: pd.Series) -> "VolatilityModel":
        """Ajusta el modelo a los rendimientos."""
        from arch import arch_model
        cfg = {
            "ARCH(1)":       dict(vol="ARCH",  p=1),
            "GARCH(1,1)":    dict(vol="Garch", p=1, q=1),
            "EGARCH(1,1)":   dict(vol="EGARCH",p=1, q=1),
            "GJR-GARCH(1,1)":dict(vol="Garch", p=1, o=1, q=1),
        }.get(self.model_name, dict(vol="Garch", p=1, q=1))

        model = arch_model(returns * 100, dist="normal", **cfg)
        self._result = model.fit(disp="off")
        return self

    def forecast(self, horizon: int = 10) -> list[float]:
        """Pronóstico de volatilidad para los próximos `horizon` días."""
        if self._result is None:
            raise RuntimeError("Llamar .fit() antes de .forecast()")
        fc = self._result.forecast(horizon=horizon)
        return np.sqrt(fc.variance.iloc[-1].values).tolist()

    @property
    def aic(self) -> float:
        return float(self._result.aic) if self._result else float("nan")

    @property
    def bic(self) -> float:
        return float(self._result.bic) if self._result else float("nan")

    @property
    def log_likelihood(self) -> float:
        return float(self._result.loglikelihood) if self._result else float("nan")

    @property
    def conditional_volatility(self) -> list[float]:
        if self._result is None:
            return []
        return self._result.conditional_volatility.tolist()
