# =============================================================================
# backend/app/services/portfolio.py – Análisis técnico y portafolio
# Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez
# =============================================================================
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

NAMES: dict[str, str] = {
    "MSI": "Motorola Solutions",
    "XOM": "ExxonMobil",
    "JNJ": "Johnson & Johnson",
    "PG":  "Procter & Gamble",
    "UL":  "Unilever",
    "TSM": "TSMC",
}


class TechnicalIndicators:
    """Calcula indicadores técnicos (RSI, MACD, Bollinger, SMA, EMA, Estocástico)."""

    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> float:
        delta = prices.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss
        rsi_s = 100 - (100 / (1 + rs))
        return round(float(rsi_s.dropna().iloc[-1]), 4)

    @staticmethod
    def macd(prices: pd.Series) -> tuple[float, float, float]:
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal    = macd_line.ewm(span=9).mean()
        hist      = macd_line - signal
        return (
            round(float(macd_line.iloc[-1]), 4),
            round(float(signal.iloc[-1]), 4),
            round(float(hist.iloc[-1]), 4),
        )

    @staticmethod
    def bollinger(prices: pd.Series, period: int = 20) -> tuple[float, float, float]:
        mid = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        return (
            round(float((mid + 2 * std).iloc[-1]), 4),
            round(float(mid.iloc[-1]), 4),
            round(float((mid - 2 * std).iloc[-1]), 4),
        )

    @staticmethod
    def stochastic(prices: pd.Series, period: int = 14) -> tuple[float, float]:
        low14  = prices.rolling(period).min()
        high14 = prices.rolling(period).max()
        k = 100 * (prices - low14) / (high14 - low14)
        d = k.rolling(3).mean()
        return round(float(k.iloc[-1]), 4), round(float(d.iloc[-1]), 4)

    @classmethod
    def all_indicators(cls, prices: pd.Series) -> dict:
        sma20 = round(float(prices.rolling(20).mean().iloc[-1]), 4)
        sma50 = round(float(prices.rolling(50).mean().iloc[-1]), 4)
        ema20 = round(float(prices.ewm(span=20, adjust=False).mean().iloc[-1]), 4)
        bb_u, bb_m, bb_l = cls.bollinger(prices)
        macd_v, sig_v, hist_v = cls.macd(prices)
        stoch_k, stoch_d = cls.stochastic(prices)
        return {
            "precio_actual": round(float(prices.iloc[-1]), 4),
            "rsi":           cls.rsi(prices),
            "macd":          macd_v,
            "macd_signal":   sig_v,
            "macd_hist":     hist_v,
            "bb_upper":      bb_u,
            "bb_middle":     bb_m,
            "bb_lower":      bb_l,
            "sma_20":        sma20,
            "sma_50":        sma50,
            "ema_20":        ema20,
            "stoch_k":       stoch_k,
            "stoch_d":       stoch_d,
        }


class SignalGenerator:
    """
    Genera señales de compra/venta por activo.
    Métodos por regla + agregador .evaluate_all() → SignalReport.
    """

    def __init__(self, rsi_ob: int = 70, rsi_os: int = 30,
                 stoch_ob: int = 80, stoch_os: int = 20):
        self.rsi_ob   = rsi_ob
        self.rsi_os   = rsi_os
        self.stoch_ob = stoch_ob
        self.stoch_os = stoch_os

    def rsi_extreme(self, rsi: float) -> str:
        if rsi > self.rsi_ob:  return "SELL"
        if rsi < self.rsi_os:  return "BUY"
        return "NEUTRAL"

    def macd_cross(self, macd: float, signal: float) -> str:
        return "BUY" if macd > signal else "SELL"

    def bollinger_signal(self, price: float, bb_upper: float, bb_lower: float) -> str:
        if price >= bb_upper:  return "SELL"
        if price <= bb_lower:  return "BUY"
        return "NEUTRAL"

    def sma_cross(self, sma_short: float, sma_long: float) -> str:
        return "BUY" if sma_short > sma_long else "SELL"

    def stoch_signal(self, k: float, d: float) -> str:
        if k > self.stoch_ob and d > self.stoch_ob:  return "SELL"
        if k < self.stoch_os and d < self.stoch_os:  return "BUY"
        return "NEUTRAL"

    def evaluate_all(self, prices: pd.Series, ticker: str) -> dict:
        ind = TechnicalIndicators.all_indicators(prices)
        signals = {
            "RSI":      self.rsi_extreme(ind["rsi"]),
            "MACD":     self.macd_cross(ind["macd"], ind["macd_signal"]),
            "Bollinger": self.bollinger_signal(ind["precio_actual"], ind["bb_upper"], ind["bb_lower"]),
            "SMA":      self.sma_cross(ind["sma_20"], ind["sma_50"]),
            "Estocástico": self.stoch_signal(ind["stoch_k"], ind["stoch_d"]),
        }
        buys  = sum(1 for s in signals.values() if s == "BUY")
        sells = sum(1 for s in signals.values() if s == "SELL")
        if buys > sells:   overall = "BUY"
        elif sells > buys: overall = "SELL"
        else:              overall = "NEUTRAL"

        interp = (
            f"{ticker}: {buys} indicadores alcistas vs {sells} bajistas. "
            f"Señal global: {'COMPRA 🟢' if overall=='BUY' else ('VENTA 🔴' if overall=='SELL' else 'NEUTRAL 🟡')}. "
            f"RSI={ind['rsi']:.1f}, MACD={'alcista' if signals['MACD']=='BUY' else 'bajista'}, "
            f"Bollinger={'sobre banda sup.' if signals['Bollinger']=='SELL' else ('bajo banda inf.' if signals['Bollinger']=='BUY' else 'dentro bandas')}."
        )
        return {
            "ticker":       ticker,
            "empresa":      NAMES.get(ticker, ticker),
            "indicadores":  signals,
            "señal_global": overall,
            "votos_compra": buys,
            "votos_venta":  sells,
            "interpretacion": interp,
        }


class PortfolioAnalyzer:
    """Análisis de portafolio: CAPM, frontera eficiente Markowitz."""

    @staticmethod
    def capm(prices: pd.DataFrame, bench: pd.Series, rf_daily: float) -> list[dict]:
        """Beta y rendimiento esperado CAPM para cada activo."""
        # Calcular log-rendimientos
        log_ret = np.log(prices / prices.shift(1)).dropna()
        bench_r = np.log(bench / bench.shift(1)).dropna()

        # Normalizar índices a solo fecha (sin timezone)
        log_ret.index = pd.to_datetime(log_ret.index).normalize().tz_localize(None)
        bench_r.index = pd.to_datetime(bench_r.index).normalize().tz_localize(None)

        # Intersección de fechas comunes
        common_idx = log_ret.index.intersection(bench_r.index)

        if len(common_idx) < 30:
            raise ValueError(f"Muy pocas fechas comunes entre portafolio y benchmark: {len(common_idx)}")

        log_ret = log_ret.loc[common_idx]
        bench_a = bench_r.loc[common_idx]

        results = []
        for ticker in prices.columns:
            r = log_ret[ticker].values
            b = bench_a.values
            sl, ic, r_val, _, _ = stats.linregress(b, r)
            beta     = round(float(sl), 4)
            alpha_d  = round(float(ic), 6)
            r2       = round(float(r_val ** 2), 4)
            rm_daily = float(b.mean())
            re_ann   = ((rf_daily + beta * (rm_daily - rf_daily)) * 252) * 100
            sis_pct  = round(r2 * 100, 2)
            idi_pct  = round((1 - r2) * 100, 2)
            clasif   = ("Agresivo" if beta > 1.1 else
                        ("Defensivo" if beta < 0.9 else "Neutral"))
            results.append({
                "ticker":                       ticker,
                "empresa":                      NAMES.get(ticker, ticker),
                "beta":                         beta,
                "alpha_diario":                 alpha_d,
                "r_squared":                    r2,
                "riesgo_sistematico_pct":       sis_pct,
                "riesgo_idiosincratico_pct":    idi_pct,
                "rendimiento_esperado_anual_pct": round(re_ann, 4),
                "clasificacion":                clasif,
            })
        return results

    @staticmethod
    def simulate_frontier(prices: pd.DataFrame, rf: float = 0.045,
                           n_portfolios: int = 10_000) -> dict:
        """
        Simula portafolios aleatorios y construye la frontera eficiente.
        Para máx Sharpe y mín varianza usa también cvxpy (QP).
        """
        log_ret = np.log(prices / prices.shift(1)).dropna()
        mu  = log_ret.mean().values * 252
        cov = log_ret.cov().values * 252
        n   = len(prices.columns)
        tickers = list(prices.columns)

        np.random.seed(42)
        records = []
        for _ in range(n_portfolios):
            w = np.random.dirichlet(np.ones(n))
            ret = float(w @ mu)
            vol = float(np.sqrt(w @ cov @ w))
            sh  = (ret - rf) / vol if vol > 0 else 0.0
            records.append((ret, vol, sh, w.tolist()))

        # Máximo Sharpe y mínima varianza
        best_sharpe = max(records, key=lambda x: x[2])
        min_var     = min(records, key=lambda x: x[1])

        def portfolio_dict(rec: tuple) -> dict:
            ret, vol, sh, w = rec
            return {
                "rendimiento_anual_pct":  round(ret * 100, 4),
                "volatilidad_anual_pct":  round(vol * 100, 4),
                "sharpe_ratio":           round(sh, 4),
                "pesos": {t: round(wi * 100, 2) for t, wi in zip(tickers, w)},
            }

        corr = log_ret.corr().to_dict()
        return {
            "max_sharpe":   portfolio_dict(best_sharpe),
            "min_varianza": portfolio_dict(min_var),
            "correlaciones": corr,
            "puntos": [{"ret": round(r * 100, 3), "vol": round(v * 100, 3), "sh": round(s, 3)}
                       for r, v, s, _ in records[:2000]],  # 2000 para viz
        }

    def generate_signals(self, prices: pd.DataFrame,
                          rsi_ob: int = 70, rsi_os: int = 30,
                          stoch_ob: int = 80, stoch_os: int = 20) -> list[dict]:
        sg = SignalGenerator(rsi_ob, rsi_os, stoch_ob, stoch_os)
        results = []
        for ticker in prices.columns:
            try:
                results.append(sg.evaluate_all(prices[ticker], ticker))
            except Exception as e:
                logger.warning(f"Error señales {ticker}: {e}")
        return results
