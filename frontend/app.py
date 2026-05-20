# =============================================================================
# frontend/app.py – Tablero Streamlit (consume el backend FastAPI)
# Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez
# Proyecto Integrador – Teoría del Riesgo · USTA
#
# NAVEGACIÓN REDISEÑADA:
#   La barra lateral solo contiene el logo/info y la selección de período.
#   Los módulos se navegan por TABS (grupos) en la zona principal.
# =============================================================================
from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
BACKEND_URL = "http://127.0.0.1:8000"
TICKERS = ["AAPL", "CVX", "JNJ", "PG", "MSFT", "TSM"]
NAMES = {
    "AAPL": "Apple Inc.",
    "CVX":  "Chevron",
    "JNJ":  "Johnson & Johnson",
    "PG":   "Procter & Gamble",
    "MSFT": "Microsoft",
    "TSM":  "TSMC",
}

# ── Paleta de colores (más clara y cómoda) ────────────────────────
NAVY   = "#001A4D"
PURPLE = "#3D008D"
PINK   = "#D91A72"
GOLD   = "#E8A800"
TEAL   = "#0891B2"
GREEN  = "#059669"
ORANGE = "#D97706"
LIGHT_BG  = "#F8FAFC"
CARD_BG   = "#FFFFFF"
TEXT_MAIN = "#1E293B"
TEXT_MUTED= "#64748B"

CHART_COLORS = [PURPLE, PINK, TEAL, ORANGE, GREEN, NAVY]
TEMPLATE_CHARTS = "plotly_white"

st.set_page_config(
    page_title="RiskLab USTA – Teoría del Riesgo",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",  # sidebar colapsada por defecto
)

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── BASE ── */
.stApp {{background-color:{LIGHT_BG}; color:{TEXT_MAIN};}}

/* ── FORZAR TEXTO OSCURO SOLO EN ZONA PRINCIPAL ── */
.main p, .main span, .main label,
.main li, .main td, .main th {{
    color:{TEXT_MAIN} !important;
}}
/* Los div generales NO se tocan para no romper banners/tabs */

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {{
    background:linear-gradient(180deg,{NAVY} 0%,#002868 100%) !important;
    border-right:3px solid {GOLD};
    min-width:220px; max-width:260px;
}}
[data-testid="stSidebar"] * {{color:#FFFFFF !important;}}

/* ── HEADINGS – solo los que NO están en banners ── */
.main h1 {{color:{NAVY} !important; font-weight:800;
     border-bottom:3px solid {PINK}; padding-bottom:8px;}}
.main h2 {{color:{PURPLE} !important; font-weight:700;}}
.main h3 {{color:{NAVY} !important; font-weight:600;}}

/* ── TABS ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background:linear-gradient(135deg,{NAVY} 0%,{PURPLE} 100%) !important;
    border-radius:12px; padding:6px; gap:4px;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    border-radius:8px; padding:8px 14px;
    font-size:0.82rem; font-weight:700;
}}
[data-testid="stTabs"] [data-baseweb="tab"] p,
[data-testid="stTabs"] [data-baseweb="tab"] span,
[data-testid="stTabs"] [data-baseweb="tab"] div {{
    color:#FFFFFF !important;
    opacity: 0.85;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    background:rgba(255,255,255,0.22) !important;
}}
[data-testid="stTabs"] [aria-selected="true"] p,
[data-testid="stTabs"] [aria-selected="true"] span,
[data-testid="stTabs"] [aria-selected="true"] div {{
    color:#FFFFFF !important;
    opacity: 1 !important;
}}
[data-testid="stTabs"] [data-baseweb="tab"]:hover p,
[data-testid="stTabs"] [data-baseweb="tab"]:hover span,
[data-testid="stTabs"] [data-baseweb="tab"]:hover div {{
    color:#FFFFFF !important;
    opacity: 1 !important;
}}

/* ── SELECTBOX / INPUTS – texto siempre oscuro ── */
[data-testid="stSelectbox"] > div > div {{
    background:#FFFFFF !important;
    color:{TEXT_MAIN} !important;
    border:1.5px solid #CBD5E1 !important;
}}
[data-testid="stSelectbox"] span,
[data-testid="stSelectbox"] div,
[data-testid="stSelectbox"] p {{
    color:{TEXT_MAIN} !important;
}}
[data-baseweb="select"] * {{
    color:{TEXT_MAIN} !important;
    background:#FFFFFF !important;
}}
[data-baseweb="popover"] * {{
    color:{TEXT_MAIN} !important;
    background:#FFFFFF !important;
}}

/* ── SLIDER ── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] p,
[data-testid="stSlider"] span {{
    color:{TEXT_MAIN} !important;
}}

/* ── NUMBER INPUT ── */
[data-testid="stNumberInput"] label,
[data-testid="stNumberInput"] p,
[data-testid="stNumberInput"] input {{
    color:{TEXT_MAIN} !important;
    background:#FFFFFF !important;
}}

/* ── CARDS ── */
.rl-card {{
    background:#FFFFFF;
    border-radius:14px; padding:20px 24px;
    box-shadow:0 2px 12px rgba(0,26,77,0.07);
    margin-bottom:14px;
    border-top:4px solid {PURPLE};
}}
.rl-card * {{color:{TEXT_MAIN} !important;}}
.rl-card h3 {{color:{NAVY} !important;}}

/* ── BANNER ── */
.module-banner {{
    background:linear-gradient(135deg,{NAVY} 0%,{PURPLE} 100%);
    border-radius:12px; padding:16px 24px; margin-bottom:18px;
}}
.module-banner h2, .module-banner p {{color:white !important;}}

/* ── CONTEXT BOX ── */
.ctx-box {{
    background:#EEF2FF;
    border:1.5px solid rgba(61,0,141,0.20);
    border-left:5px solid {PURPLE};
    border-radius:10px; padding:14px 18px; margin:10px 0 16px 0;
}}
.ctx-box, .ctx-box * {{color:{TEXT_MAIN} !important;}}

/* ── SEÑALES / SEMÁFORO (NO TOCAR) ── */
.sem-buy {{
    background:#D1FAE5; border:1.5px solid #059669;
    border-radius:10px; padding:10px 12px; text-align:center;
}}
.sem-buy * {{color:#065F46 !important;}}
.sem-sell {{
    background:#FEE2E2; border:1.5px solid #DC2626;
    border-radius:10px; padding:10px 12px; text-align:center;
}}
.sem-sell * {{color:#7F1D1D !important;}}
.sem-neutral {{
    background:#FEF3C7; border:1.5px solid #D97706;
    border-radius:10px; padding:10px 12px; text-align:center;
}}
.sem-neutral * {{color:#78350F !important;}}

/* ── MÉTRICAS ── */
[data-testid="metric-container"] {{
    background:#FFFFFF;
    border:1px solid rgba(61,0,141,0.10);
    border-radius:10px; padding:12px 16px;
}}
[data-testid="metric-container"] label,
[data-testid="metric-container"] div,
[data-testid="metric-container"] p {{
    color:{TEXT_MAIN} !important;
}}

/* ── BOTONES ── */
.stButton > button {{
    color:#FFFFFF !important;
    background:{PURPLE} !important;
    border-radius:8px; border:none; font-weight:600;
}}
.stButton > button:hover {{
    background:{NAVY} !important;
    color:#FFFFFF !important;
}}

/* ── DATAFRAME / TABLAS ── */
[data-testid="stDataFrame"] * {{
    color:{TEXT_MAIN} !important;
}}
.stDataFrame th {{
    background:{PURPLE} !important;
    color:#FFFFFF !important;
}}

/* ── EXPANDER ── */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary *,
[data-testid="stExpander"] p,
[data-testid="stExpander"] div {{
    color:{TEXT_MAIN} !important;
}}

/* ── SELECT_SLIDER ── */
[data-testid="stSlider"] div[role="slider"] {{
    background:{PURPLE} !important;
}}

/* ── CODE ── */
pre, code {{
    color:{NAVY} !important;
    background:#F1F5F9 !important;
}}

/* ── INFO / WARNING / ERROR BOXES ── */
[data-testid="stAlert"] * {{
    color:{TEXT_MAIN} !important;
}}

/* ── HEADER PRINCIPAL – máxima especificidad ── */
.risklab-header span {{
    color:#FFFFFF !important;
    -webkit-text-fill-color:#FFFFFF !important;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def api_get(endpoint: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ No se pudo conectar al backend. Asegúrate de que FastAPI esté corriendo en localhost:8000")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Error del backend: {e.response.status_code} – {e.response.text}")
        return None
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")
        return None


def api_post(endpoint: str, body: dict) -> dict | None:
    try:
        r = requests.post(f"{BACKEND_URL}{endpoint}", json=body, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ No se pudo conectar al backend FastAPI.")
        return None
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", e.response.text) if e.response else str(e)
        st.error(f"❌ Error del backend ({e.response.status_code}): {detail}")
        return None
    except Exception as e:
        st.error(f"❌ Error inesperado: {e}")
        return None


def banner(title: str, subtitle: str = "") -> None:
    sub = f"<p style='color:rgba(255,255,255,0.85);margin:4px 0 0 0;font-size:0.88rem;'>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="module-banner"><h2 style="margin:0;">{title}</h2>{sub}</div>',
                unsafe_allow_html=True)


def ctx(html: str) -> None:
    st.markdown(f'<div class="ctx-box">{html}</div>', unsafe_allow_html=True)


def card(html: str, color: str = PURPLE) -> None:
    st.markdown(
        f'<div class="rl-card" style="border-top-color:{color};">{html}</div>',
        unsafe_allow_html=True
    )


def chart_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        template=TEMPLATE_CHARTS,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color="#1E293B", size=12, family="Calibri"),
        title_font=dict(color="#1E293B", size=14),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            font=dict(color="#1E293B"),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        xaxis=dict(
            color="#1E293B",
            tickfont=dict(color="#1E293B"),
            title_font=dict(color="#1E293B"),
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
        ),
        yaxis=dict(
            color="#1E293B",
            tickfont=dict(color="#1E293B"),
            title_font=dict(color="#1E293B"),
            gridcolor="#F1F5F9",
            linecolor="#CBD5E1",
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_color="#1E293B",
            bordercolor="#CBD5E1",
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# SIDEBAR – solo período y resumen
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0 8px;">
        <div style="font-size:2rem;">📊</div>
        <div style="font-weight:800; font-size:1.05rem; color:#FDB913;">RiskLab USTA</div>
        <div style="font-size:0.72rem; color:rgba(255,255,255,0.65); margin-top:2px;">
            Teoría del Riesgo · Python
        </div>
    </div>
    <hr style="border-color:rgba(255,255,255,0.2); margin:8px 0;">
    """, unsafe_allow_html=True)

    period = st.selectbox(
        "📅 Período de análisis:",
        ["1mo","3mo","6mo","1y","2y","3y","5y"],
        index=4,
        help="Período de datos históricos para todos los módulos."
    )

    st.markdown(f"""
    <hr style="border-color:rgba(255,255,255,0.2); margin:10px 0;">
    <div style="font-size:0.72rem; color:rgba(255,255,255,0.6); text-align:center;">
        <b>Portafolio:</b><br>MSI · XOM · JNJ<br>PG · UL · TSM<br>
        <br><b>Backend:</b> FastAPI + SQLAlchemy<br>
        <b>Frontend:</b> Streamlit
    </div>
    <hr style="border-color:rgba(255,255,255,0.2); margin:10px 0;">
    <div style="font-size:0.68rem; color:rgba(255,255,255,0.5); text-align:center;">
        Alejandra Sepúlveda<br>Ingrid Umbacia Ramírez<br>USTA · 2025
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# HEADER PRINCIPAL
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="risklab-header" style="background:linear-gradient(135deg,#001A4D 0%,#3D008D 100%);
     border-radius:16px; padding:22px 32px; margin-bottom:20px;">
    <span style="display:block; font-size:1.7rem; font-weight:800;
          color:#FFFFFF !important; -webkit-text-fill-color:#FFFFFF !important;
          font-family:Calibri,sans-serif; margin:0 0 6px 0;">
        📊 RiskLab USTA – Sistema Integral de Análisis de Riesgo
    </span>
    <span style="display:block; font-size:0.9rem;
          color:rgba(255,255,255,0.85) !important;
          -webkit-text-fill-color:rgba(255,255,255,0.85) !important;
          font-family:Calibri,sans-serif;">
        Alejandra Sepúlveda · Ingrid Umbacia Ramírez &nbsp;|&nbsp;
        Universidad Santo Tomás · Teoría del Riesgo &amp; Python para APIs e IA
    </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# NAVEGACIÓN POR GRUPOS DE TABS
# ─────────────────────────────────────────────────────────────────
grupo_tab, renta_tab, api_tab = st.tabs([
    "📈 Módulos de Mercado",
    "🏦 Renta Fija · Opciones · Stress",
    "🔌 API · ML · Macro",
])

# =================================================================
# GRUPO 1: MÓDULOS DE MERCADO
# =================================================================
with grupo_tab:
    m1, m2, m3, m4, m5, m6, m7 = st.tabs([
        "📉 Análisis Técnico",
        "📊 Rendimientos",
        "🌊 ARCH/GARCH",
        "🛡️ CAPM",
        "⚠️ VaR/CVaR",
        "🎯 Markowitz",
        "🚦 Señales",
    ])

    # ── M1 – ANÁLISIS TÉCNICO ────────────────────────────────────
    with m1:
        banner("📉 Módulo 1 – Análisis Técnico",
               "RSI · MACD · Bandas de Bollinger · SMA · EMA · Estocástico")

        ticker_sel = st.selectbox("Activo:", TICKERS,
                                  format_func=lambda x: f"{x} – {NAMES[x]}", key="at_tick")

        with st.spinner("Cargando datos..."):
            ind_data  = api_get(f"/indicadores/{ticker_sel}", {"period": period})
            prec_data = api_get(f"/precios/{ticker_sel}",   {"period": period})

        if not ind_data or not prec_data:
            st.stop()

        prices_list = prec_data.get("precios", [])
        df_prices = pd.DataFrame(prices_list)
        df_prices["fecha"] = pd.to_datetime(df_prices["fecha"])
        df_prices = df_prices.set_index("fecha").sort_index()
        p = df_prices["precio"]

        # Indicadores locales para gráfico
        sma_s = p.rolling(20).mean()
        sma_l = p.rolling(50).mean()
        ema20 = p.ewm(span=20, adjust=False).mean()
        delta = p.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rsi_s = 100 - (100 / (1 + gain / loss))
        ema12 = p.ewm(span=12).mean(); ema26 = p.ewm(span=26).mean()
        macd_l = ema12 - ema26; macd_sig = macd_l.ewm(span=9).mean(); macd_h = macd_l - macd_sig
        bb_mid = p.rolling(20).mean(); bb_u = bb_mid + 2*p.rolling(20).std(); bb_l_b = bb_mid - 2*p.rolling(20).std()
        sk = 100*(p - p.rolling(14).min())/(p.rolling(14).max()-p.rolling(14).min())
        sd = sk.rolling(3).mean()

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            row_heights=[0.42, 0.20, 0.20, 0.18],
                            subplot_titles=[f"Precio – {ticker_sel}", "RSI (14)", "MACD", "Estocástico"])
        fig.add_trace(go.Scatter(x=p.index, y=p, name="Precio", line=dict(color=NAVY, width=1.8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=sma_s.index, y=sma_s, name="SMA20", line=dict(color=PURPLE, width=1.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=sma_l.index, y=sma_l, name="SMA50", line=dict(color=PINK, width=1.5, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=ema20.index, y=ema20, name="EMA20", line=dict(color=GOLD, width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb_u.index, y=bb_u, name="BB Sup", line=dict(color="rgba(61,0,141,0.35)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=bb_l_b.index, y=bb_l_b, name="BB Inf",
                                  line=dict(color="rgba(61,0,141,0.35)", width=1),
                                  fill="tonexty", fillcolor="rgba(61,0,141,0.04)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=rsi_s.index, y=rsi_s, name="RSI", line=dict(color=TEAL, width=1.8)), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="red", dash="dash", width=1), row=2, col=1)
        fig.add_hline(y=30, line=dict(color=GREEN, dash="dash", width=1), row=2, col=1)
        colors_m = [PURPLE if v >= 0 else PINK for v in macd_h.fillna(0)]
        fig.add_trace(go.Bar(x=macd_h.index, y=macd_h, name="Hist.", marker_color=colors_m, opacity=0.65), row=3, col=1)
        fig.add_trace(go.Scatter(x=macd_l.index, y=macd_l, name="MACD", line=dict(color=NAVY, width=1.8)), row=3, col=1)
        fig.add_trace(go.Scatter(x=macd_sig.index, y=macd_sig, name="Señal", line=dict(color=PINK, width=1.8)), row=3, col=1)
        fig.add_trace(go.Scatter(x=sk.index, y=sk, name="%K", line=dict(color=PURPLE, width=1.8)), row=4, col=1)
        fig.add_trace(go.Scatter(x=sd.index, y=sd, name="%D", line=dict(color=GOLD, width=1.8)), row=4, col=1)
        fig.add_hline(y=80, line=dict(color="red", dash="dash", width=1), row=4, col=1)
        fig.add_hline(y=20, line=dict(color=GREEN, dash="dash", width=1), row=4, col=1)
        chart_layout(fig, 820)
        st.plotly_chart(fig, use_container_width=True)

        # Valores actuales
        st.markdown("### 📋 Valores Actuales del Backend")
        rsi_v = ind_data["rsi"]; macd_v = ind_data["macd"]; sig_v = ind_data["macd_signal"]
        px_v  = ind_data["precio_actual"]; bbu_v = ind_data["bb_upper"]; bbl_v = ind_data["bb_lower"]
        smas_v = ind_data["sma_20"]; smal_v = ind_data["sma_50"]

        ci1, ci2 = st.columns(2)
        with ci1:
            cls_r = "sem-sell" if rsi_v > 70 else ("sem-buy" if rsi_v < 30 else "sem-neutral")
            lbl_r = "🔴 SOBRECOMPRADO" if rsi_v > 70 else ("🟢 SOBREVENDIDO" if rsi_v < 30 else "🟡 NEUTRAL")
            st.markdown(f'<div class="{cls_r}"><b>RSI = {rsi_v:.1f}</b><br>{lbl_r}</div>', unsafe_allow_html=True)
            cls_m = "sem-buy" if macd_v > sig_v else "sem-sell"
            lbl_m = "🟢 Momentum alcista" if macd_v > sig_v else "🔴 Momentum bajista"
            st.markdown(f'<div class="{cls_m}" style="margin-top:10px;"><b>MACD → {lbl_m}</b><br>MACD={macd_v:.3f} | Señal={sig_v:.3f}</div>', unsafe_allow_html=True)
        with ci2:
            if px_v >= bbu_v:
                st.markdown(f'<div class="sem-sell"><b>Bollinger → 🔴 Sobre banda superior</b><br>${px_v:.2f} ≥ ${bbu_v:.2f}</div>', unsafe_allow_html=True)
            elif px_v <= bbl_v:
                st.markdown(f'<div class="sem-buy"><b>Bollinger → 🟢 Bajo banda inferior</b><br>${px_v:.2f} ≤ ${bbl_v:.2f}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="sem-neutral"><b>Bollinger → 🟡 Dentro de bandas</b><br>${bbl_v:.2f} – ${bbu_v:.2f}</div>', unsafe_allow_html=True)
            cls_s = "sem-buy" if smas_v > smal_v else "sem-sell"
            lbl_s = "🟢 Golden Cross / Alcista" if smas_v > smal_v else "🔴 Death Cross / Bajista"
            st.markdown(f'<div class="{cls_s}" style="margin-top:10px;"><b>Medias Móviles → {lbl_s}</b><br>SMA20=${smas_v:.2f} | SMA50=${smal_v:.2f}</div>', unsafe_allow_html=True)

    # ── M2 – RENDIMIENTOS ────────────────────────────────────────
    with m2:
        banner("📊 Módulo 2 – Rendimientos y Propiedades Empíricas",
               "Log-rendimientos · Estadísticos · Pruebas de normalidad · Hechos estilizados")
        ticker_sel2 = st.selectbox("Activo:", TICKERS, format_func=lambda x: f"{x} – {NAMES[x]}", key="rend_tick")

        with st.spinner("Cargando rendimientos..."):
            rend_data = api_get(f"/rendimientos/{ticker_sel2}", {"period": period})
            prec_data2 = api_get(f"/precios/{ticker_sel2}", {"period": period})

        if not rend_data:
            st.stop()

        st.markdown("### 📐 Estadísticos Descriptivos")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Media diaria", f"{rend_data['media_diaria_pct']:.4f}%")
        c1.metric("Ret. Anualizado", f"{rend_data['rendimiento_anualizado_pct']:.2f}%")
        c2.metric("Desv. Estándar diaria", f"{rend_data['volatilidad_diaria_pct']:.4f}%")
        c2.metric("Volatilidad Anualizada", f"{rend_data['volatilidad_anualizada_pct']:.2f}%")
        c3.metric("Mínimo diario", f"{rend_data['min_diario_pct']:.3f}%")
        c3.metric("Máximo diario", f"{rend_data['max_diario_pct']:.3f}%")
        c4.metric("Skewness", f"{rend_data['skewness']:.4f}")
        c4.metric("Kurtosis (exceso)", f"{rend_data['kurtosis']:.4f}")

        lr_df = pd.DataFrame(rend_data["rendimientos"])
        lr_df["fecha"] = pd.to_datetime(lr_df["fecha"])
        lr = lr_df["log_ret_pct"].values / 100

        st.markdown("### 📈 Serie Temporal de Log-Rendimientos")
        fig_lr = go.Figure(go.Scatter(x=lr_df["fecha"], y=lr_df["log_ret_pct"],
                                       line=dict(color=PINK, width=0.9), name="Log-Ret (%)"))
        chart_layout(fig_lr, 300)
        st.plotly_chart(fig_lr, use_container_width=True)

        col_h1, col_h2 = st.columns(2)
        with col_h1:
            mu, sigma = np.mean(lr), np.std(lr)
            x_norm = np.linspace(lr.min(), lr.max(), 200)
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=lr*100, nbinsx=80, name="Frecuencia",
                                             histnorm="probability density",
                                             marker_color=PURPLE, opacity=0.7))
            fig_hist.add_trace(go.Scatter(x=x_norm*100, y=stats.norm.pdf(x_norm, mu, sigma)/100,
                                           name="Normal teórica", line=dict(color=PINK, width=2.5)))
            chart_layout(fig_hist.update_layout(title="Histograma + Normal Teórica"), 380)
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_h2:
            (tq, sq), (sl, ic, _) = stats.probplot(lr, dist="norm")
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(x=tq, y=sq, mode="markers",
                                         marker=dict(color=PURPLE, size=4, opacity=0.55)))
            fig_qq.add_trace(go.Scatter(x=tq, y=sl*np.array(tq)+ic, mode="lines",
                                         line=dict(color=PINK, width=2.5), name="Normal"))
            chart_layout(fig_qq.update_layout(title="Q-Q Plot vs Normal"), 380)
            st.plotly_chart(fig_qq, use_container_width=True)

        st.markdown("### 🧪 Pruebas de Normalidad")
        jb_s, jb_p = stats.jarque_bera(lr)
        sw_s, sw_p = stats.shapiro(lr[-500:]) if len(lr) > 500 else stats.shapiro(lr)
        ct1, ct2 = st.columns(2)
        for col, nm, s_val, p_val in [(ct1,"Jarque-Bera",jb_s,jb_p),(ct2,"Shapiro-Wilk",sw_s,sw_p)]:
            res = "🔴 Rechaza normalidad (p < 0.05)" if p_val < 0.05 else "🟢 No rechaza normalidad"
            col.markdown(f"""
            <div class="rl-card" style="border-top-color:{PINK};">
                <b>{nm}</b><br>
                Estadístico: <code>{s_val:.4f}</code><br>
                p-valor: <code>{p_val:.2e}</code><br>
                <b>{res}</b>
            </div>""", unsafe_allow_html=True)

        ctx(f"""
        <strong>📌 Hechos Estilizados – {ticker_sel2}:</strong>
        <ol style="margin:8px 0 0 16px;">
        <li><b>No normalidad</b> confirmada por Jarque-Bera (p={jb_p:.2e}). Colas más gruesas que la normal.</li>
        <li><b>Kurtosis = {rend_data['kurtosis']:.2f}:</b> Leptocurtosis → eventos extremos más frecuentes (fat tails).</li>
        <li><b>Skewness = {rend_data['skewness']:.3f}:</b> {"Asimetría negativa → pérdidas extremas más frecuentes." if rend_data['skewness'] < -0.05 else "Asimetría positiva o aproximadamente simétrica."}</li>
        <li><b>Implicación para riesgo:</b> El VaR paramétrico normal subestima el riesgo real.</li>
        </ol>
        """)

    # ── M3 – ARCH/GARCH ──────────────────────────────────────────
    with m3:
        banner("🌊 Módulo 3 – Modelos ARCH/GARCH",
               "EWMA · ARCH(1) · GARCH(1,1) · EGARCH · GJR-GARCH · AIC/BIC · Pronóstico")
        from arch import arch_model as _arch_model

        ticker_sel3 = st.selectbox("Activo:", TICKERS, format_func=lambda x: f"{x} – {NAMES[x]}", key="garch_tick")

        with st.spinner("Descargando precios..."):
            prec_data3 = api_get(f"/precios/{ticker_sel3}", {"period": period})

        if not prec_data3:
            st.stop()

        df_p3 = pd.DataFrame(prec_data3["precios"])
        df_p3["fecha"] = pd.to_datetime(df_p3["fecha"])
        df_p3 = df_p3.set_index("fecha").sort_index()
        p3 = df_p3["precio"]
        lr_t = (np.log(p3 / p3.shift(1)).dropna() * 100)

        # EWMA
        lam = st.slider("Lambda EWMA:", 0.80, 0.99, 0.94, 0.01, key="ewma_lam")
        ewma_vol = lr_t.ewm(alpha=1 - lam, adjust=False).std()
        fig_ewma = go.Figure()
        fig_ewma.add_trace(go.Scatter(x=ewma_vol.index, y=ewma_vol, name=f"EWMA (λ={lam})",
                                       line=dict(color=TEAL, width=1.8)))
        chart_layout(fig_ewma.update_layout(title=f"Volatilidad EWMA – {ticker_sel3}",
                                             yaxis_title="Volatilidad (%)"), 300)
        st.plotly_chart(fig_ewma, use_container_width=True)

        ctx(f"""<strong>¿Por qué ARCH/GARCH?</strong> Los rendimientos financieros presentan
        <b>agrupamiento de volatilidad</b>: períodos de alta agitación se agrupan.
        EWMA (λ={lam}) asigna mayor peso a observaciones recientes.
        Los modelos GARCH capturan esta heteroscedasticidad condicional con mayor rigor estadístico.""")

        @st.cache_data(ttl=3600)
        def fit_models(vals: list, key: str) -> dict:
            s = pd.Series(vals)
            out = {}
            for nm, kw in [("ARCH(1)", dict(vol="ARCH", p=1)),
                            ("GARCH(1,1)", dict(vol="Garch", p=1, q=1)),
                            ("EGARCH(1,1)", dict(vol="EGARCH", p=1, q=1)),
                            ("GJR-GARCH(1,1)", dict(vol="Garch", p=1, o=1, q=1))]:
                try:
                    r = _arch_model(s, dist="normal", **kw).fit(disp="off")
                    out[nm] = {"aic": r.aic, "bic": r.bic, "ll": r.loglikelihood,
                                "cond_vol": r.conditional_volatility.tolist(),
                                "std_resid": r.std_resid.tolist(),
                                "forecast_var": r.forecast(horizon=10).variance.iloc[-1].tolist()}
                except Exception:
                    out[nm] = None
            return out

        with st.spinner("Ajustando modelos GARCH..."):
            models = fit_models(lr_t.values.tolist(), ticker_sel3)

        rows_m = []
        for nm, r in models.items():
            if r:
                rows_m.append({"Modelo": nm, "Log-L": f"{r['ll']:.2f}",
                                "AIC": f"{r['aic']:.2f}", "BIC": f"{r['bic']:.2f}"})
        if rows_m:
            df_m = pd.DataFrame(rows_m)
            aics = pd.to_numeric(df_m["AIC"])
            best_aic = df_m.loc[aics.idxmin(), "Modelo"]
            best_bic = df_m.loc[pd.to_numeric(df_m["BIC"]).idxmin(), "Modelo"]
            df_m["✅ AIC"] = df_m["Modelo"].apply(lambda x: "✅" if x == best_aic else "")
            df_m["✅ BIC"] = df_m["Modelo"].apply(lambda x: "✅" if x == best_bic else "")
            st.dataframe(df_m, use_container_width=True)
            ctx(f"Mejor por AIC: <b>{best_aic}</b> | Mejor por BIC: <b>{best_bic}</b>. Menor valor = mejor ajuste.")

        if models.get("GARCH(1,1)"):
            g11 = models["GARCH(1,1)"]
            cond_vol = pd.Series(g11["cond_vol"], index=lr_t.index[:len(g11["cond_vol"])])
            std_res  = pd.Series(g11["std_resid"], index=lr_t.index[:len(g11["std_resid"])])

            fig_cv = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    subplot_titles=["Volatilidad Condicional – GARCH(1,1)", "Residuos Estand."])
            fig_cv.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol,
                                         line=dict(color=PINK, width=1.8), name="Vol. Cond."), row=1, col=1)
            fig_cv.add_trace(go.Scatter(x=std_res.index, y=std_res,
                                         line=dict(color=PURPLE, width=0.9), name="Residuos"), row=2, col=1)
            fig_cv.add_hline(y=3, line=dict(color="red", dash="dash"), row=2, col=1)
            fig_cv.add_hline(y=-3, line=dict(color="red", dash="dash"), row=2, col=1)
            chart_layout(fig_cv, 480)
            st.plotly_chart(fig_cv, use_container_width=True)

            fcast_vol = np.sqrt(g11["forecast_var"])
            fig_fc = go.Figure(go.Scatter(x=list(range(1, 11)), y=fcast_vol,
                                           mode="lines+markers",
                                           line=dict(color=PURPLE, width=2.5),
                                           marker=dict(size=9, color=PINK)))
            chart_layout(fig_fc.update_layout(title="Pronóstico Volatilidad GARCH – 10 días",
                                               xaxis_title="Días adelante", yaxis_title="Volatilidad (%)"), 320)
            st.plotly_chart(fig_fc, use_container_width=True)
            ctx(f"GARCH(1,1) pronostica volatilidad promedio de <b>{np.mean(fcast_vol):.3f}%</b> diario "
                f"(≈ <b>{np.mean(fcast_vol)*np.sqrt(252):.2f}% anualizado</b>).")

    # ── M4 – CAPM ────────────────────────────────────────────────
    with m4:
        banner("🛡️ Módulo 4 – CAPM y Riesgo Sistemático",
               "Beta · SML · Rf desde API · Riesgo sistemático vs. idiosincrático")
        with st.spinner("Consultando CAPM..."):
            capm_data = api_get("/capm", {"period": period})
        if not capm_data:
            st.stop()

        ctx(f"""Rf = <b>{capm_data['rf_anual_pct']:.2f}%</b> (T-Bill 13W, backend).
        Prima de mercado estimada: <b>{capm_data['prima_mercado_anual_pct']:.2f}%</b> anual vs. S&P 500.""")

        activos = capm_data["activos"]
        df_capm = pd.DataFrame(activos)
        st.markdown("### 📋 Tabla CAPM")
        st.dataframe(df_capm[["ticker","empresa","beta","alpha_diario","r_squared",
                               "riesgo_sistematico_pct","riesgo_idiosincratico_pct",
                               "rendimiento_esperado_anual_pct","clasificacion"]],
                     use_container_width=True)

        betas = [a["beta"] for a in activos]
        col_b = [PINK if b > 1.1 else PURPLE if b < 0.9 else GOLD for b in betas]
        fig_b = go.Figure(go.Bar(x=[a["ticker"] for a in activos], y=betas,
                                  marker_color=col_b, text=[f"{b:.3f}" for b in betas],
                                  textposition="outside"))
        fig_b.add_hline(y=1.0, line=dict(color="gray", dash="dash"), annotation_text="β=1.0")
        chart_layout(fig_b.update_layout(title="Beta por Activo"), 380)
        st.plotly_chart(fig_b, use_container_width=True)

        ctx("""<strong>Interpretación:</strong><ul style="margin:6px 0 0 16px;">
        <li><b>β > 1:</b> Activo agresivo – amplifica movimientos del mercado (mayor riesgo sistemático).</li>
        <li><b>β < 1:</b> Activo defensivo – amortigua las caídas del mercado.</li>
        <li><b>R²:</b> Proporción del riesgo total que es sistemático (no diversificable por Markowitz).</li>
        <li><b>Alpha de Jensen:</b> Exceso de rendimiento ajustado por riesgo sistemático.</li>
        </ul>""")

    # ── M5 – VaR / CVaR ──────────────────────────────────────────
    with m5:
        banner("⚠️ Módulo 5 – VaR y CVaR",
               "Paramétrico · Histórico · Montecarlo · CVaR · Test de Kupiec")
        st.markdown("### ⚖️ Composición del Portafolio")
        col_w = st.columns(len(TICKERS))
        weights = []
        for i, t in enumerate(TICKERS):
            w = col_w[i].number_input(t, min_value=0.0, max_value=1.0,
                                       value=round(1/len(TICKERS), 4), step=0.01,
                                       format="%.3f", key=f"w_{t}")
            weights.append(w)
        ws = sum(weights)
        weights_norm = [w/ws for w in weights] if ws > 0 else weights
        conf = st.select_slider("Nivel de confianza:", [0.90, 0.95, 0.99], value=0.95)

        if st.button("📊 Calcular VaR vía Backend"):
            body = {"tickers": TICKERS, "weights": weights_norm, "confidence": conf, "period": period}
            with st.spinner("Calculando VaR..."):
                var_data = api_post("/var", body)

            if var_data:
                st.markdown("### 📊 Resultados")
                df_var = pd.DataFrame(var_data["resultados"])
                st.dataframe(df_var, use_container_width=True)

                # Gráfico comparativo VaR
                fig_var = go.Figure()
                for i, row in df_var.iterrows():
                    fig_var.add_trace(go.Bar(name=row["metodo"],
                                              x=["VaR Diario %", "CVaR Diario %", "VaR Anualizado %"],
                                              y=[row["var_diario_pct"], row["cvar_diario_pct"], row["var_anualizado_pct"]],
                                              marker_color=CHART_COLORS[i]))
                chart_layout(fig_var.update_layout(title="Comparativa VaR por Método",
                                                    barmode="group"), 380)
                st.plotly_chart(fig_var, use_container_width=True)

                kup = var_data["kupiec"]
                ck1, ck2, ck3, ck4 = st.columns(4)
                ck1.metric("Violaciones obs.", kup["violaciones_observadas"],
                            f"Esperadas: {kup['violaciones_esperadas']:.1f}")
                ck2.metric("Tasa violaciones", f"{kup['tasa_violaciones_pct']:.3f}%",
                            f"Esp: {kup['tasa_esperada_pct']:.3f}%")
                ck3.metric("LR Kupiec", f"{kup['lr_statistic']:.4f}", "Crítico χ²(1)=3.841")
                ck4.metric("p-valor", f"{kup['p_valor']:.4f}",
                            "✅ Válido" if kup["modelo_valido"] else "❌ Rechazado")
                st.info(kup.get("interpretacion_kupiec", ""))

    # ── M6 – MARKOWITZ ───────────────────────────────────────────
    with m6:
        banner("🎯 Módulo 6 – Frontera Eficiente de Markowitz",
               "Correlación · 10,000 portafolios · Máx Sharpe · Mín Varianza")
        n_sim = st.slider("Portafolios a simular:", 1000, 20000, 10000, 1000, key="nsim")

        if st.button("🎲 Calcular Frontera vía Backend"):
            body = {"tickers": TICKERS, "period": period, "n_portfolios": n_sim}
            with st.spinner("Simulando frontera eficiente..."):
                front_data = api_post("/frontera-eficiente", body)

            if front_data:
                corr_df = pd.DataFrame(front_data["correlaciones"])
                fig_corr = px.imshow(corr_df, color_continuous_scale=["#D91A72","white","#3D008D"],
                                      zmin=-1, zmax=1, text_auto=".3f",
                                      title="Matriz de Correlaciones")
                fig_corr.update_layout(height=420, template=TEMPLATE_CHARTS)
                st.plotly_chart(fig_corr, use_container_width=True)

                ms = front_data["max_sharpe"]; mv = front_data["min_varianza"]
                col_o1, col_o2 = st.columns(2)
                with col_o1:
                    st.markdown(f"""
                    <div class="rl-card" style="border-top-color:{PINK};">
                        <h3 style="color:{PINK};">★ Portafolio Máximo Sharpe</h3>
                        Rendimiento: <b>{ms['rendimiento_anual_pct']:.2f}%</b><br>
                        Volatilidad: <b>{ms['volatilidad_anual_pct']:.2f}%</b><br>
                        Sharpe: <b>{ms['sharpe_ratio']:.4f}</b>
                    </div>""", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(list(ms["pesos"].items()), columns=["Activo","Peso (%)"]),
                                 use_container_width=True)
                with col_o2:
                    st.markdown(f"""
                    <div class="rl-card" style="border-top-color:{PURPLE};">
                        <h3 style="color:{PURPLE};">◆ Portafolio Mínima Varianza</h3>
                        Rendimiento: <b>{mv['rendimiento_anual_pct']:.2f}%</b><br>
                        Volatilidad: <b>{mv['volatilidad_anual_pct']:.2f}%</b><br>
                        Sharpe: <b>{mv['sharpe_ratio']:.4f}</b>
                    </div>""", unsafe_allow_html=True)
                    st.dataframe(pd.DataFrame(list(mv["pesos"].items()), columns=["Activo","Peso (%)"]),
                                 use_container_width=True)

    # ── M7 – SEÑALES ─────────────────────────────────────────────
    with m7:
        banner("🚦 Módulo 7 – Panel de Señales y Alertas",
               "RSI · MACD · Bollinger · Medias Móviles · Estocástico · Umbrales configurables")
        cu1, cu2, cu3, cu4 = st.columns(4)
        rsi_ob   = cu1.slider("RSI sobrecompra", 60, 85, 70, key="rsi_ob")
        rsi_os   = cu2.slider("RSI sobreventa",  15, 40, 30, key="rsi_os")
        stoch_ob = cu3.slider("Estoc. sobrecompra", 65, 90, 80, key="st_ob")
        stoch_os = cu4.slider("Estoc. sobreventa",  10, 35, 20, key="st_os")
        st.markdown("---")

        with st.spinner("Consultando señales..."):
            alertas_data = api_get("/alertas", {
                "period": period, "rsi_ob": rsi_ob, "rsi_os": rsi_os,
                "stoch_ob": stoch_ob, "stoch_os": stoch_os
            })

        if not alertas_data:
            st.stop()

        for alerta in alertas_data["alertas"]:
            overall = alerta["señal_global"]
            nb, ns = alerta["votos_compra"], alerta["votos_venta"]
            ov_label = "🟢 COMPRA" if overall=="BUY" else ("🔴 VENTA" if overall=="SELL" else "🟡 NEUTRAL")

            with st.expander(
                f"**{alerta['ticker']} – {alerta['empresa']}** &nbsp;·&nbsp; {ov_label} ({nb}✅/{ns}❌)",
                expanded=False
            ):
                cols_s = st.columns(5)
                for i, (ind_name, sig) in enumerate(alerta["indicadores"].items()):
                    cls = "sem-buy" if sig=="BUY" else ("sem-sell" if sig=="SELL" else "sem-neutral")
                    icon = "🟢" if sig=="BUY" else ("🔴" if sig=="SELL" else "🟡")
                    lbl  = "COMPRA" if sig=="BUY" else ("VENTA" if sig=="SELL" else "NEUTRAL")
                    cols_s[i].markdown(f"""
                    <div class="{cls}">
                        <div style="font-size:0.75rem; font-weight:700;">{ind_name}</div>
                        <div style="font-size:1.3rem;">{icon}</div>
                        <div style="font-size:0.78rem; font-weight:600;">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown(f"""<div class="ctx-box" style="margin-top:10px;">
                    <b>{alerta['interpretacion']}</b></div>""", unsafe_allow_html=True)


# =================================================================
# GRUPO 2: RENTA FIJA · OPCIONES · STRESS
# =================================================================
with renta_tab:
    rf_tab, op_tab, st_tab, mac_tab, bench_tab = st.tabs([
        "📈 Curva + Nelson-Siegel",
        "💹 Bono – Duración/Convexidad",
        "⚡ Opciones Black-Scholes",
        "🔥 Stress Testing",
        "🌍 Macro y Benchmark",
    ])

    # ── CURVA DE RENDIMIENTO ─────────────────────────────────────
    with rf_tab:
        banner("📈 Módulo – Curva de Rendimiento US + Nelson-Siegel",
               "Puntos FRED/Yahoo · Ajuste NS · DGS3MO, DGS1, DGS2, DGS5, DGS10, DGS30")
        with st.spinner("Obteniendo curva de rendimiento..."):
            curva_data = api_get("/curva-rendimiento")

        if curva_data:
            puntos = curva_data["puntos"]
            ns_fit = curva_data["ns_fitted"]
            ns_par = curva_data["ns_params"]

            mats_obs = [p["maturity"] for p in puntos]
            ylds_obs = [p["yield_pct"] for p in puntos]
            mats_fit = [p["maturity"] for p in ns_fit]
            ylds_fit = [p["yield_pct"] for p in ns_fit]

            fig_yc = go.Figure()
            fig_yc.add_trace(go.Scatter(x=mats_obs, y=ylds_obs, mode="markers+lines",
                                         name="Puntos observados",
                                         marker=dict(color=NAVY, size=10),
                                         line=dict(color=NAVY, width=1.5, dash="dot")))
            fig_yc.add_trace(go.Scatter(x=mats_fit, y=ylds_fit, mode="lines",
                                         name="Nelson-Siegel ajustado",
                                         line=dict(color=PINK, width=2.8)))
            chart_layout(fig_yc.update_layout(
                title="Curva de Rendimiento US – Spot vs. Nelson-Siegel",
                xaxis_title="Madurez (años)", yaxis_title="Rendimiento (%)"
            ), 420)
            st.plotly_chart(fig_yc, use_container_width=True)

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("β₀ (Nivel)", f"{ns_par.get('beta0',0):.4f}%")
            p2.metric("β₁ (Pendiente)", f"{ns_par.get('beta1',0):.4f}%")
            p3.metric("β₂ (Curvatura)", f"{ns_par.get('beta2',0):.4f}%")
            p4.metric("λ (Decay)", f"{ns_par.get('lambda',0):.4f}")

            ctx("""<b>Modelo Nelson-Siegel:</b> r(τ) = β₀ + β₁·[(1-e<sup>-τ/λ</sup>)/(τ/λ)] + β₂·[(1-e<sup>-τ/λ</sup>)/(τ/λ) - e<sup>-τ/λ</sup>]
            <br>β₀ = nivel largo plazo · β₁ = componente de corto plazo · β₂ = joroba (hump) · λ = velocidad de decaimiento.""")
        else:
            st.warning("No se pudo obtener la curva de rendimiento.")

    # ── BONO ────────────────────────────────────────────────────
    with op_tab:
        banner("💹 Valoración de Bono – Duración y Convexidad",
               "Precio · Duración Macaulay · Duración Modificada · Convexidad · Shock ±200 pb")
        cb1, cb2, cb3 = st.columns(3)
        face_v   = cb1.number_input("Valor nominal (USD)", value=1000.0, step=100.0)
        coup_r   = cb2.number_input("Tasa cupón anual (%)", value=5.0, step=0.5) / 100
        mat_b    = cb3.number_input("Años al vencimiento", value=5.0, step=0.5)
        ytm_b    = cb1.number_input("YTM (%)", value=4.5, step=0.25) / 100
        freq_b   = cb2.selectbox("Frecuencia cupones", [1, 2, 4], index=1,
                                   format_func=lambda x: {1:"Anual",2:"Semestral",4:"Trimestral"}[x])

        if st.button("💰 Valorar Bono"):
            body = {"face_value": face_v, "coupon_rate": coup_r, "maturity": mat_b,
                    "ytm": ytm_b, "frequency": freq_b}
            bono_data = api_post("/bono/valorar", body)
            if bono_data:
                bb1, bb2, bb3 = st.columns(3)
                bb1.metric("Precio del Bono", f"${bono_data['precio']:,.2f}")
                bb2.metric("Duración Macaulay", f"{bono_data['duracion_macaulay']:.4f} años")
                bb3.metric("Duración Modificada", f"{bono_data['duracion_modificada']:.4f}")
                bb1.metric("Convexidad", f"{bono_data['convexidad']:.4f}")
                bb2.metric("Precio con shock +200pb", f"${bono_data['precio_shock_200pb']:,.2f}")
                bb3.metric("Δ Precio %", f"{bono_data['delta_precio_pct']:.4f}%")

                ctx(f"""<b>Interpretación:</b> Un bono con duración modificada de
                <b>{bono_data['duracion_modificada']:.4f}</b> años experimenta una variación de precio de
                aproximadamente <b>-{bono_data['duracion_modificada']:.2f}%</b> por cada +100 pb en el YTM.
                La convexidad corrige este estimado cuadrático: precio ajustado con +200 pb = ${bono_data['precio_shock_200pb']:,.2f}
                (variación real: {bono_data['delta_precio_pct']:.2f}%).""")

    # ── OPCIONES BLACK-SCHOLES ───────────────────────────────────
    with st_tab:
        banner("⚡ Módulo – Opciones Europeas (Black-Scholes + Greeks)",
               "Call · Put · Δ · Γ · ν · Θ · ρ · Paridad Put-Call · Volatilidad implícita")
        ot1, ot2, ot3 = st.columns(3)
        opt_tick = ot1.selectbox("Ticker subyacente", TICKERS, key="opt_tick")
        opt_tipo = ot2.selectbox("Tipo", ["call", "put"])
        opt_S    = ot3.number_input("Precio subyacente S ($)", value=200.0, step=10.0)
        opt_K    = ot1.number_input("Strike K ($)", value=200.0, step=5.0)
        opt_T    = ot2.number_input("Tiempo T (años)", value=0.5, step=0.1, min_value=0.01)
        opt_r    = ot3.number_input("Rf r (%)", value=4.5, step=0.25) / 100
        opt_sig  = ot1.number_input("Volatilidad σ (%)", value=25.0, step=1.0) / 100

        if st.button("⚡ Valorar Opción"):
            body = {"ticker": opt_tick, "S": opt_S, "K": opt_K, "T": opt_T,
                    "r": opt_r, "sigma": opt_sig, "tipo": opt_tipo}
            opt_data = api_post("/opcion/precio", body)
            if opt_data:
                oc1, oc2 = st.columns([1, 2])
                with oc1:
                    st.markdown(f"""
                    <div class="rl-card" style="border-top-color:{PINK};">
                        <h3>Precio {opt_tipo.upper()}</h3>
                        <div style="font-size:2rem; font-weight:800; color:{PINK};">
                            ${opt_data['precio']:.4f}
                        </div>
                    </div>""", unsafe_allow_html=True)
                with oc2:
                    g = opt_data["greeks"]
                    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
                    gc1.metric("Δ Delta", f"{g['delta']:.4f}")
                    gc2.metric("Γ Gamma", f"{g['gamma']:.6f}")
                    gc3.metric("ν Vega", f"{g['vega']:.4f}")
                    gc4.metric("Θ Theta", f"{g['theta']:.4f}")
                    gc5.metric("ρ Rho", f"{g['rho']:.4f}")

                if opt_data.get("paridad_put_call") is not None:
                    ctx(f"""<b>Verificación Paridad Put-Call:</b> C - P = S - K·e<sup>-rT</sup>
                    → Diferencia = <b>{opt_data['paridad_put_call']:.6f}</b>
                    (≈0 confirma la paridad). La paridad no se cumple para opciones americanas.""")

                ctx(f"""<b>Interpretación de Greeks:</b><br>
                <b>Delta ({g['delta']:.4f}):</b> Por cada $1 de subida del subyacente, la opción cambia ${g['delta']:.4f}.<br>
                <b>Gamma ({g['gamma']:.6f}):</b> Aceleración del Delta; alto cuando S ≈ K (at-the-money).<br>
                <b>Vega ({g['vega']:.4f}):</b> Cambio de precio por +1% en σ implícita (clave en stress testing).<br>
                <b>Theta ({g['theta']:.4f}):</b> Pérdida de valor por día (decaimiento temporal).<br>
                <b>Rho ({g['rho']:.4f}):</b> Sensibilidad al cambio en la tasa de interés.""")

    # ── STRESS TESTING ───────────────────────────────────────────
    with mac_tab:
        banner("🔥 Módulo – Stress Testing del Portafolio",
               "Shock de tasa · Volatilidad · Precio · Comparativa con VaR base")
        st.markdown("### ⚙️ Parámetros de Stress")
        ss1, ss2, ss3 = st.columns(3)
        s_tasa  = ss1.slider("Shock tasa (pb)", 50, 400, 200, 50) / 10000
        s_vol   = ss2.slider("Shock volatilidad (%)", 10, 100, 30, 5) / 100
        s_precio= ss3.slider("Shock precio (%)", -50, -5, -20, 5) / 100

        st.markdown("#### Pesos del portafolio para stress")
        cw = st.columns(len(TICKERS))
        st_weights = []
        for i, t in enumerate(TICKERS):
            w = cw[i].number_input(t, 0.0, 1.0, round(1/len(TICKERS),3), 0.01, format="%.3f", key=f"sw_{t}")
            st_weights.append(w)
        sw_sum = sum(st_weights)
        st_weights_norm = [w/sw_sum for w in st_weights] if sw_sum > 0 else st_weights

        if st.button("🔥 Ejecutar Stress Test"):
            body = {"tickers": TICKERS, "weights": st_weights_norm, "period": period,
                    "shock_tasa": s_tasa, "shock_vol": s_vol, "shock_precio": s_precio}
            with st.spinner("Ejecutando stress test..."):
                stress_data = api_post("/stress", body)

            if stress_data:
                sc_list = stress_data["escenarios"]
                fig_st = go.Figure()
                fig_st.add_trace(go.Bar(
                    x=[s["escenario"] for s in sc_list],
                    y=[s["var_base_pct"] for s in sc_list],
                    name="VaR Base", marker_color=TEAL, opacity=0.8
                ))
                fig_st.add_trace(go.Bar(
                    x=[s["escenario"] for s in sc_list],
                    y=[s["perdida_estimada_pct"] for s in sc_list],
                    name="Pérdida Stress", marker_color=PINK, opacity=0.9
                ))
                chart_layout(fig_st.update_layout(
                    title="Stress Testing: VaR Base vs Pérdida Estimada (%)",
                    barmode="group", yaxis_title="Pérdida (%)"
                ), 380)
                st.plotly_chart(fig_st, use_container_width=True)

                for sc in sc_list:
                    with st.expander(f"📌 {sc['escenario']}"):
                        cs1, cs2 = st.columns(2)
                        cs1.metric("VaR Base", f"{sc['var_base_pct']:.4f}%")
                        cs2.metric("Pérdida Estimada", f"{sc['perdida_estimada_pct']:.4f}%",
                                   delta=f"{sc['perdida_estimada_pct']-sc['var_base_pct']:+.4f}%")
                        st.write(sc["descripcion"])

    # ── MACRO Y BENCHMARK ────────────────────────────────────────
    with bench_tab:
        banner("🌍 Módulo – Macro y Benchmark",
               "Rf · VIX · Oro · Brent · USD/COP · Rendimiento acumulado vs S&P 500")
        import yfinance as yf

        with st.spinner("Consultando macro..."):
            macro_data = api_get("/macro")

        if macro_data:
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("📈 Rf anual", f"{macro_data['rf_anual_pct']:.2f}%")
            c2.metric("😱 VIX", f"{macro_data.get('vix') or 'N/D'}")
            c3.metric("🥇 Oro USD/oz", f"${macro_data.get('oro_usd'):,.0f}" if macro_data.get('oro_usd') else "N/D")
            c4.metric("🛢️ Brent", f"${macro_data.get('brent_usd'):.1f}" if macro_data.get('brent_usd') else "N/D")
            c5.metric("💵 USD/COP", f"{macro_data.get('usd_cop'):,.0f}" if macro_data.get('usd_cop') else "N/D")
            c6.metric("🇺🇸 Inflación US", f"{macro_data.get('inflacion_us'):.2f}%" if macro_data.get('inflacion_us') else "N/D")

        st.markdown("---")
        st.markdown("### 📈 Rendimiento Acumulado vs. S&P 500")
        with st.spinner("Descargando datos benchmark..."):
            try:
                raw_p  = yf.download(TICKERS, period=period, auto_adjust=True, progress=False)["Close"].dropna()
                raw_b  = yf.download("^GSPC", period=period, auto_adjust=True, progress=False)["Close"].dropna()
                if isinstance(raw_b, pd.DataFrame): raw_b = raw_b.iloc[:,0]
                port_ret  = np.log(raw_p/raw_p.shift(1)).dropna().mean(axis=1)
                bench_s   = np.log(raw_b/raw_b.shift(1)).dropna()
                cidx = port_ret.index.intersection(bench_s.index)
                pr = port_ret.loc[cidx]; br = bench_s.loc[cidx]
                cum_p = ((1+pr).cumprod())*100/((1+pr).cumprod().iloc[0])
                cum_b = ((1+br).cumprod())*100/((1+br).cumprod().iloc[0])

                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(x=cum_p.index, y=cum_p, name="Portafolio",
                                              line=dict(color=PURPLE, width=2.8)))
                fig_cum.add_trace(go.Scatter(x=cum_b.index, y=cum_b, name="S&P 500",
                                              line=dict(color=PINK, width=2.8, dash="dash")))
                chart_layout(fig_cum.update_layout(
                    title="Rendimiento Acumulado – Base 100", yaxis_title="Índice (Base 100)"
                ), 420)
                st.plotly_chart(fig_cum, use_container_width=True)

                n = len(pr)
                ann_p = (1+(1+pr).prod()-1)**(252/n)-1 if n > 0 else 0
                ann_b = (1+(1+br).prod()-1)**(252/n)-1 if n > 0 else 0
                rf = (macro_data["rf_anual_pct"]/100) if macro_data else 0.045
                vol_p = pr.std()*np.sqrt(252); vol_b = br.std()*np.sqrt(252)
                sharpe_p = (ann_p-rf)/vol_p if vol_p > 0 else 0
                sharpe_b = (ann_b-rf)/vol_b if vol_b > 0 else 0
                from scipy import stats as sp_stats
                sl_j, al_j, *_ = sp_stats.linregress(br.values, pr.values)
                alpha_j = al_j * 252
                active = pr.values - br.values
                te = pd.Series(active).std()*np.sqrt(252)
                ir = (pr.mean()-br.mean())*252/te if te > 0 else 0

                def mdd(r):
                    c=(1+pd.Series(r)).cumprod()
                    return float(((c-c.cummax())/c.cummax()).min())

                perf_df = pd.DataFrame({
                    "Métrica": ["Ret. Anualizado","Volatilidad Anual","Sharpe Ratio",
                                 "Máx. Drawdown","Alpha Jensen (anual)","Beta","Tracking Error","Info Ratio"],
                    "Portafolio": [f"{ann_p*100:.2f}%",f"{vol_p*100:.2f}%",f"{sharpe_p:.4f}",
                                   f"{mdd(pr)*100:.2f}%",f"{alpha_j*100:.4f}%",f"{sl_j:.4f}",
                                   f"{te*100:.2f}%",f"{ir:.4f}"],
                    "S&P 500": [f"{ann_b*100:.2f}%",f"{vol_b*100:.2f}%",f"{sharpe_b:.4f}",
                                f"{mdd(br)*100:.2f}%","—","1.0","—","—"],
                })
                st.dataframe(perf_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error al calcular benchmark: {e}")


# =================================================================
# GRUPO 3: API · ML · MACRO
# =================================================================
with api_tab:
    api_m, ml_m = st.tabs([
        "🔌 API FastAPI – Demostración",
        "🤖 ML – Predicción de Dirección",
    ])

    # ── API FASTAPI ──────────────────────────────────────────────
    with api_m:
        banner("🔌 Backend FastAPI – Demostración",
               "Endpoints · Pydantic · Depends() · BaseSettings · SQLAlchemy ORM · /docs · /redoc")

        ctx("""<strong>Arquitectura:</strong> El frontend <b>nunca</b> llama directamente a Yahoo Finance:
        siempre consume el backend FastAPI. El backend expone <code>/docs</code> (Swagger UI) y
        <code>/redoc</code> generados automáticamente por FastAPI + Pydantic.""")

        st.markdown("### 🔍 Estado del Backend")
        with st.spinner("Verificando conexión..."):
            root_data = api_get("/")
        if root_data:
            st.success("✅ Backend FastAPI activo y respondiendo")
            st.json(root_data)
        else:
            st.error("❌ Backend no disponible. Ejecuta: `uvicorn app.main:app --reload` desde backend/")

        st.markdown("### 📋 Endpoints Disponibles")
        endpoints_df = pd.DataFrame([
            {"/activos":         "GET",  "Descripción": "Lista activos del portafolio"},
            {"/precios/{ticker}":"GET",  "Descripción": "Precios históricos"},
            {"/rendimientos/{ticker}":"GET","Descripción":"Estadísticas log-rendimientos"},
            {"/indicadores/{ticker}":"GET","Descripción":"RSI, MACD, Bollinger, etc."},
            {"/var":             "POST", "Descripción": "VaR y CVaR (3 métodos) + Kupiec"},
            {"/capm":            "GET",  "Descripción": "Beta, Alpha, CAPM"},
            {"/frontera-eficiente":"POST","Descripción":"Frontera Eficiente Markowitz"},
            {"/alertas":         "GET",  "Descripción": "Señales de compra/venta"},
            {"/macro":           "GET",  "Descripción": "Indicadores macro en tiempo real"},
            {"/curva-rendimiento":"GET", "Descripción": "⭐ Curva spot + Nelson-Siegel (FRED)"},
            {"/bono/valorar":    "POST", "Descripción": "⭐ Valoración bono: precio, duración, convexidad"},
            {"/opcion/precio":   "POST", "Descripción": "⭐ Black-Scholes + 5 Greeks + paridad"},
            {"/stress":          "POST", "Descripción": "⭐ Stress testing (3 escenarios)"},
            {"/predict":         "POST", "Descripción": "⭐⭐ Predicción ML (Singleton)"},
        ].copy() if False else [
            {"Endpoint": k, "Método": v, "Descripción": d}
            for k, v, d in [
                ("/activos","GET","Lista activos del portafolio"),
                ("/precios/{ticker}","GET","Precios históricos"),
                ("/rendimientos/{ticker}","GET","Estadísticas log-rendimientos"),
                ("/indicadores/{ticker}","GET","RSI, MACD, Bollinger, etc."),
                ("/var","POST","VaR y CVaR (3 métodos) + Kupiec"),
                ("/capm","GET","Beta, Alpha, CAPM"),
                ("/frontera-eficiente","POST","Frontera Eficiente Markowitz"),
                ("/alertas","GET","Señales de compra/venta"),
                ("/macro","GET","Indicadores macro en tiempo real"),
                ("/curva-rendimiento","GET","⭐ Curva spot + Nelson-Siegel"),
                ("/bono/valorar","POST","⭐ Precio, duración, convexidad"),
                ("/opcion/precio","POST","⭐ Black-Scholes + 5 Greeks + paridad"),
                ("/stress","POST","⭐ Stress testing (3 escenarios)"),
                ("/predict","POST","⭐⭐ Predicción ML (Singleton)"),
            ]
        ])
        st.dataframe(endpoints_df, use_container_width=True)

        st.markdown("### 🧪 Prueba un Endpoint en Vivo")
        ep = st.selectbox("Endpoint:", ["/activos", "/macro", "/capm"])
        if st.button("▶️ Ejecutar"):
            with st.spinner(f"Consultando {ep}..."):
                data = api_get(ep, {"period": period})
            if data:
                st.json(data)

        st.markdown("### 💡 Conceptos Python Aplicados")
        st.markdown(f"""
        <div class="rl-card">
        <table style="width:100%; font-size:0.83rem; border-collapse:collapse;">
        <tr style="background:{PURPLE}; color:white;">
            <th style="padding:8px; color:white;">Concepto (semana)</th>
            <th style="padding:8px; color:white;">Implementación</th>
            <th style="padding:8px; color:white;">Archivo</th>
        </tr>
        <tr><td style="padding:6px;"><b>Decoradores (S1)</b></td><td>@log_execution_time, @cache_result(ttl=3600)</td><td>services/data.py</td></tr>
        <tr style="background:#F8FAFC;"><td style="padding:6px;"><b>Type hints (S1)</b></td><td>Todas las funciones anotadas</td><td>Todos los archivos</td></tr>
        <tr><td style="padding:6px;"><b>POO (S2)</b></td><td>DataService, RiskCalculator, Bond, YieldCurve, OptionPricer</td><td>services/</td></tr>
        <tr style="background:#F8FAFC;"><td style="padding:6px;"><b>Pydantic (S2/4/5)</b></td><td>Request + Response models con @field_validator, @model_validator</td><td>models/schemas.py</td></tr>
        <tr><td style="padding:6px;"><b>HTTPException (S2)</b></td><td>400, 404, 422, 503 con detalle</td><td>main.py</td></tr>
        <tr style="background:#F8FAFC;"><td style="padding:6px;"><b>async/await (S4)</b></td><td>Todas las rutas son async def</td><td>main.py</td></tr>
        <tr><td style="padding:6px;"><b>Depends() (S6)</b></td><td>Inyección DataService, RiskCalculator, MLSingleton, DB Session</td><td>dependencies.py</td></tr>
        <tr style="background:#F8FAFC;"><td style="padding:6px;"><b>BaseSettings (S6)</b></td><td>Settings(BaseSettings) + .env + @lru_cache</td><td>config.py</td></tr>
        <tr><td style="padding:6px;"><b>SQLAlchemy ORM (S7)</b></td><td>PrecioCache, MacroCache, PredictionLog, StressLog</td><td>db/database.py</td></tr>
        <tr style="background:#F8FAFC;"><td style="padding:6px;"><b>Singleton ML (S8-10)</b></td><td>MLModelSingleton.__new__() + @lru_cache en Depends</td><td>ml/model.py</td></tr>
        </table>
        </div>
        """, unsafe_allow_html=True)

    # ── ML PREDICCIÓN ────────────────────────────────────────────
    with ml_m:
        banner("🤖 Módulo – Predicción ML (Singleton Pattern)",
               "Pipeline: StandardScaler + RandomForest · train_test_split(shuffle=False) · /predict · PredictionLog")

        ctx("""<b>¿Cómo funciona el modelo ML?</b>
        <ol style="margin:8px 0 0 16px;">
        <li><b>Features:</b> RSI normalizado, MACD/precio, posición en Bollinger Band, ratio SMA, retorno 5d, volatilidad 20d.</li>
        <li><b>Target:</b> Dirección del rendimiento a N días (UP/DOWN). Sin leakage: train_test_split(shuffle=False).</li>
        <li><b>Pipeline:</b> StandardScaler → RandomForestClassifier(100 árboles).</li>
        <li><b>Serialización:</b> joblib.dump() → model_v1.joblib. Singleton carga el modelo una sola vez en memoria.</li>
        <li><b>Auditoría:</b> Cada predicción se persiste en la tabla PredictionLog (SQLite).</li>
        </ol>""")

        ml_tick = st.selectbox("Activo a predecir:", TICKERS,
                                format_func=lambda x: f"{x} – {NAMES[x]}", key="ml_tick")
        ml_hor = st.slider("Horizonte (días):", 1, 30, 5, key="ml_hor")

        if st.button("🤖 Predecir"):
            body = {"ticker": ml_tick, "period": period, "horizon_days": ml_hor}
            with st.spinner("Ejecutando predicción (verificar logs: modelo solo carga 1 vez)..."):
                pred_data = api_post("/predict", body)
            if pred_data:
                dc1, dc2, dc3 = st.columns(3)
                color_dir = GREEN if pred_data["direction"] == "UP" else PINK
                dc1.metric("Dirección", f"{'📈 UP' if pred_data['direction']=='UP' else '📉 DOWN'}",
                            f"Confianza: {pred_data['confidence']*100:.1f}%")
                dc2.metric("Retorno estimado", f"{pred_data['prediction_pct']:.4f}%")
                dc3.metric("Modelo versión", pred_data["model_version"])

                st.markdown("#### 🔍 Features Extraídas")
                feats_df = pd.DataFrame(list(pred_data["features"].items()),
                                         columns=["Feature", "Valor"])
                feats_df["Descripción"] = feats_df["Feature"].map({
                    "rsi": "RSI / 100 (0-1)",
                    "macd": "MACD / Precio actual",
                    "bb_position": "Posición dentro de Bandas Bollinger (0=inf, 1=sup)",
                    "sma_ratio": "SMA20/SMA50 - 1 (ratio de momentum)",
                    "ret_5d": "Retorno acumulado 5 días",
                    "vol_20d": "Volatilidad histórica 20 días",
                })
                st.dataframe(feats_df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,{NAVY} 0%,{PURPLE} 100%);
     border-radius:14px; padding:18px 28px; margin-top:28px; text-align:center;">
    <p style="color:#fff; font-weight:700; font-size:0.95rem; margin:0;">
        Alejandra Sepúlveda &nbsp;·&nbsp; Ingrid Umbacia Ramírez
    </p>
    <p style="color:rgba(255,255,255,0.6); font-size:0.75rem; margin:4px 0 0 0;">
        Teoría del Riesgo · Universidad Santo Tomás · Backend: FastAPI + SQLAlchemy · Frontend: Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
