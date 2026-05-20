# =============================================================================
# backend/app/services/derivatives.py – Renta Fija, Opciones, Stress Testing
# Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez
# =============================================================================
from __future__ import annotations

import math
import logging
import numpy as np
import pandas as pd
from scipy import stats, optimize

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────
# RENTA FIJA
# ─────────────────────────────────────────────────────────────────

class Bond:
    """
    Valoración de bonos con cupón a tasa de mercado (YTM).
    Incluye Duración de Macaulay, Modificada y Convexidad.
    """

    def __init__(self, face_value: float, coupon_rate: float,
                 maturity: float, ytm: float, frequency: int = 2):
        self.F = face_value
        self.c = coupon_rate
        self.T = maturity
        self.y = ytm
        self.m = frequency  # pagos por año
        self.n = int(maturity * frequency)  # total de pagos
        self.C = face_value * coupon_rate / frequency  # cupón por período

    def _cash_flows(self) -> tuple[list[float], list[float]]:
        times = [(t + 1) / self.m for t in range(self.n)]
        cfs   = [self.C] * self.n
        cfs[-1] += self.F
        return times, cfs

    def precio(self) -> float:
        """Precio del bono (valor presente de los flujos)."""
        times, cfs = self._cash_flows()
        r = self.y / self.m
        return sum(cf / (1 + r) ** (t * self.m) for cf, t in zip(cfs, times))

    def macaulay_duration(self) -> float:
        """Duración de Macaulay (años)."""
        times, cfs = self._cash_flows()
        r = self.y / self.m
        P = self.precio()
        pv_times = sum(t * cf / (1 + r) ** (t * self.m) for cf, t in zip(cfs, times))
        return pv_times / P if P > 0 else 0.0

    def modified_duration(self) -> float:
        """Duración Modificada = D_Mac / (1 + y/m)."""
        return self.macaulay_duration() / (1 + self.y / self.m)

    def convexity(self) -> float:
        """Convexidad del bono."""
        times, cfs = self._cash_flows()
        r = self.y / self.m
        P = self.precio()
        conv = sum(
            t * (t + 1 / self.m) * cf / (1 + r) ** (t * self.m + 2)
            for cf, t in zip(cfs, times)
        )
        return conv / P if P > 0 else 0.0

    def precio_shock(self, delta_ytm: float) -> float:
        """Aproximación cuadrática: precio tras shock de tasa."""
        P     = self.precio()
        D_mod = self.modified_duration()
        conv  = self.convexity()
        delta_P = P * (-D_mod * delta_ytm + 0.5 * conv * delta_ytm ** 2)
        return round(P + delta_P, 4)


# ─────────────────────────────────────────────────────────────────
# CURVA DE RENDIMIENTO – NELSON-SIEGEL
# ─────────────────────────────────────────────────────────────────

class YieldCurve:
    """
    Ajuste de la curva de rendimientos con el modelo Nelson-Siegel.
    Usa scipy.optimize.least_squares con bounds en λ > 0.
    """

    def __init__(self):
        self.params: dict[str, float] = {}
        self._maturities: list[float] = []
        self._yields: list[float] = []

    @staticmethod
    def _ns_rate(tau: float, beta0: float, beta1: float,
                 beta2: float, lam: float) -> float:
        """Tasa Nelson-Siegel para madurez tau."""
        if tau <= 0 or lam <= 0:
            return beta0
        x = tau / lam
        ex = math.exp(-x)
        factor = (1 - ex) / x
        return beta0 + beta1 * factor + beta2 * (factor - ex)

    def fit_nelson_siegel(self, maturities: list[float],
                          yields: list[float]) -> "YieldCurve":
        """Ajusta Nelson-Siegel a los puntos observados."""
        self._maturities = maturities
        self._yields = yields
        mat = np.array(maturities)
        yld = np.array(yields)

        def residuals(params):
            b0, b1, b2, lam = params
            fitted = np.array([self._ns_rate(t, b0, b1, b2, lam) for t in mat])
            return fitted - yld

        y0 = yld.mean()
        x0 = [y0, yld[0] - y0, 0.0, 1.5]
        res = optimize.least_squares(
            residuals, x0,
            bounds=([-np.inf, -np.inf, -np.inf, 0.01],
                    [np.inf,  np.inf,  np.inf, 30.0])
        )
        b0, b1, b2, lam = res.x
        self.params = {"beta0": round(b0, 6), "beta1": round(b1, 6),
                       "beta2": round(b2, 6), "lam": round(lam, 6)}
        return self

    def spot_rate(self, tau: float) -> float:
        """Tasa spot para una madurez dada (años)."""
        if not self.params:
            raise RuntimeError("Llamar .fit_nelson_siegel() primero.")
        return self._ns_rate(tau, self.params["beta0"], self.params["beta1"],
                             self.params["beta2"], self.params["lam"])

    def fitted_curve(self, n_points: int = 50) -> list[dict]:
        """Curva ajustada en n_points entre 0.1 y 30 años."""
        if not self.params:
            return []
        taus = np.linspace(0.1, 30, n_points)
        return [{"maturity": round(t, 2), "yield_pct": round(self.spot_rate(t), 4)}
                for t in taus]


# ─────────────────────────────────────────────────────────────────
# OPCIONES – BLACK-SCHOLES + GREEKS
# ─────────────────────────────────────────────────────────────────

class OptionPricer:
    """
    Valoración de opciones europeas con Black-Scholes.
    Incluye las cinco Greeks y paridad put-call.
    """

    @staticmethod
    def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

    def black_scholes(self, S: float, K: float, T: float,
                      r: float, sigma: float, tipo: str = "call") -> float:
        """Precio Black-Scholes de una opción europea."""
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        if tipo == "call":
            return S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)
        else:  # put
            return K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

    def greeks(self, S: float, K: float, T: float,
               r: float, sigma: float, tipo: str = "call") -> dict[str, float]:
        """Calcula las cinco Greeks: Δ, Γ, ν, Θ, ρ."""
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        n_d1   = stats.norm.pdf(d1)
        sqrt_T = math.sqrt(T)

        # Delta
        delta = stats.norm.cdf(d1) if tipo == "call" else stats.norm.cdf(d1) - 1
        # Gamma (igual para call y put)
        gamma = n_d1 / (S * sigma * sqrt_T)
        # Vega (igual para call y put, por 1% de σ)
        vega  = S * n_d1 * sqrt_T / 100
        # Theta (por día calendario)
        if tipo == "call":
            theta = (-(S * n_d1 * sigma / (2 * sqrt_T))
                     - r * K * math.exp(-r * T) * stats.norm.cdf(d2)) / 365
        else:
            theta = (-(S * n_d1 * sigma / (2 * sqrt_T))
                     + r * K * math.exp(-r * T) * stats.norm.cdf(-d2)) / 365
        # Rho (por 1% de r)
        if tipo == "call":
            rho = K * T * math.exp(-r * T) * stats.norm.cdf(d2) / 100
        else:
            rho = -K * T * math.exp(-r * T) * stats.norm.cdf(-d2) / 100

        return {
            "delta": round(delta, 6),
            "gamma": round(gamma, 6),
            "vega":  round(vega, 6),
            "theta": round(theta, 6),
            "rho":   round(rho, 6),
        }

    def paridad_put_call(self, S: float, K: float, T: float, r: float,
                          sigma: float) -> float:
        """Verifica paridad put-call: C - P = S - K·e^(-rT). Retorna diferencia."""
        call = self.black_scholes(S, K, T, r, sigma, "call")
        put  = self.black_scholes(S, K, T, r, sigma, "put")
        teorico = S - K * math.exp(-r * T)
        return round((call - put) - teorico, 6)


# ─────────────────────────────────────────────────────────────────
# STRESS TESTING
# ─────────────────────────────────────────────────────────────────

class StressTester:
    """Aplica escenarios de stress al portafolio óptimo."""

    @staticmethod
    def run(returns: pd.Series,
            shock_tasa: float  = 0.02,
            shock_vol: float   = 0.30,
            shock_precio: float = -0.20) -> list[dict]:
        """
        Tres escenarios de stress:
          1. Shock de tasa libre de riesgo (+200 pb)
          2. Shock de volatilidad (+30%)
          3. Shock de precio (-20%)
        """
        from app.services.risk import RiskCalculator
        calc = RiskCalculator()

        var_base, _ = calc.var_historical(returns, 0.95)

        scenarios = []

        # Escenario 1: Shock de tasa
        stressed_rate = var_base * (1 + shock_tasa * 5)
        scenarios.append({
            "escenario":           f"Shock de tasa +{shock_tasa*100:.0f} pb",
            "var_base_pct":        round(var_base * 100, 4),
            "perdida_estimada_pct": round(stressed_rate * 100, 4),
            "descripcion": (
                f"Un incremento de {shock_tasa*100:.0f} pb en las tasas de interés "
                f"incrementa el VaR del portafolio en ~{shock_tasa*500:.0f}%, "
                f"elevando la pérdida estimada al {stressed_rate*100:.2f}%. "
                f"Los activos de mayor duración (bonos) son los más afectados."
            ),
        })

        # Escenario 2: Shock de volatilidad (estilo COVID/GFC)
        shocked_returns = returns * (1 + shock_vol)
        var_vol, _ = calc.var_historical(shocked_returns, 0.95)
        scenarios.append({
            "escenario":           f"Crisis de volatilidad +{shock_vol*100:.0f}%",
            "var_base_pct":        round(var_base * 100, 4),
            "perdida_estimada_pct": round(var_vol * 100, 4),
            "descripcion": (
                f"Un choque de volatilidad del +{shock_vol*100:.0f}% (análogo a COVID-19 o GFC) "
                f"eleva el VaR al {var_vol*100:.2f}%. "
                f"El CVaR (Expected Shortfall) sería aún mayor, capturando las pérdidas en la cola izquierda."
            ),
        })

        # Escenario 3: Crash de precio
        crashed_returns = returns + shock_precio / 252
        var_crash, _ = calc.var_historical(crashed_returns, 0.95)
        scenarios.append({
            "escenario":           f"Crash de precio {shock_precio*100:.0f}%",
            "var_base_pct":        round(var_base * 100, 4),
            "perdida_estimada_pct": round(var_crash * 100, 4),
            "descripcion": (
                f"Una caída de {abs(shock_precio)*100:.0f}% en los precios (shock sistémico) "
                f"genera una pérdida estimada del {var_crash*100:.2f}% diario. "
                f"Este escenario es comparable al Black Monday (1987) o Lehman Brothers (2008)."
            ),
        })

        return scenarios
