# 📊 RiskLab USTA – Sistema Integral de Análisis de Riesgo Financiero

**Proyecto Integrador – Teoría del Riesgo · Python para APIs e IA**  
Universidad Santo Tomás · Facultad de Estadística · 2025

**Autoras:** Alejandra Sepúlveda · Ingrid Umbacia Ramírez

---

## 🏗️ Arquitectura del Sistema

```
risklab/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app principal (todos los endpoints)
│   │   ├── config.py            # BaseSettings + .env + @lru_cache
│   │   ├── dependencies.py      # Inyección de dependencias con Depends()
│   │   ├── models/
│   │   │   └── schemas.py       # Modelos Pydantic (Request + Response)
│   │   ├── services/
│   │   │   ├── data.py          # DataService: Yahoo Finance + caché
│   │   │   ├── risk.py          # RiskCalculator: VaR, CVaR, Kupiec, EWMA
│   │   │   ├── portfolio.py     # TechnicalIndicators, SignalGenerator, CAPM, Markowitz
│   │   │   ├── derivatives.py   # Bond, YieldCurve (NS), OptionPricer (BS), StressTester
│   │   │   └── fred.py          # Integración FRED API
│   │   ├── ml/
│   │   │   └── model.py         # MLModelSingleton + extract_features
│   │   └── db/
│   │       └── database.py      # SQLAlchemy ORM: PrecioCache, MacroCache, PredictionLog
│   └── Dockerfile               # Multi-stage build (python:3.11.9-slim-bookworm)
├── frontend/
│   └── app.py                   # Tablero Streamlit (8 módulos + navegación por tabs)
├── tests/
│   └── test_api.py              # pytest + TestClient (30+ tests)
├── .github/workflows/ci.yml    # GitHub Actions CI
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Instalación y Ejecución Local

### 1. Clonar el repositorio

```bash
git clone https://github.com/<usuario>/risklab-usta.git
cd risklab-usta
```

### 2. Crear entorno virtual

```bash
python3.11 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env y agregar tu FRED_API_KEY (gratuita en https://fred.stlouisfed.org/)
```

### 4. Ejecutar el backend (Terminal 1)

```bash
cd backend
uvicorn app.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
# ReDoc:      http://localhost:8000/redoc
```

### 5. Ejecutar el frontend (Terminal 2)

```bash
cd frontend
streamlit run app.py --server.port 8501
# Tablero: http://localhost:8501
```

---

## 📋 Endpoints de la API

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Información general |
| `/activos` | GET | Lista activos del portafolio |
| `/precios/{ticker}` | GET | Precios históricos |
| `/rendimientos/{ticker}` | GET | Log-rendimientos + estadísticos |
| `/indicadores/{ticker}` | GET | RSI, MACD, Bollinger, SMA, EMA |
| `/var` | POST | VaR/CVaR (paramétrico, histórico, MC) + Kupiec |
| `/capm` | GET | Beta, Alpha, riesgo sistemático |
| `/frontera-eficiente` | POST | Frontera de Markowitz |
| `/alertas` | GET | Señales de compra/venta |
| `/macro` | GET | Rf, VIX, Oro, Brent, USD/COP |
| `/curva-rendimiento` | GET | ⭐ Curva spot + Nelson-Siegel |
| `/bono/valorar` | POST | ⭐ Precio, duración, convexidad |
| `/opcion/precio` | POST | ⭐ Black-Scholes + 5 Greeks + paridad |
| `/stress` | POST | ⭐ Stress testing (3 escenarios) |
| `/predict` | POST | ⭐⭐ Predicción ML (Singleton) |

---

## 🧪 Tests

```bash
pytest tests/ -v --tb=short
```

---

## 🐳 Docker

```bash
# Build y run con docker compose
docker compose up --build

# Solo build
docker build -f backend/Dockerfile -t risklab-usta:latest .
```

---

## ☁️ Deploy en Render

1. Conectar el repositorio en [render.com](https://render.com)
2. Crear un **Web Service** apuntando a `backend/`
3. Configurar las variables de entorno en Render (igual que `.env`)
4. El comando de inicio es: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

> **Nota:** El free-tier de Render duerme el servicio tras 15 min sin tráfico.  
> Antes de la sustentación ejecutar: `curl https://<tu-url>.onrender.com/`

---

## 🤖 Modelo ML

El modelo predice la **dirección del rendimiento** (UP/DOWN) a N días:

- **Features:** RSI, MACD/precio, posición Bollinger, SMA ratio, retorno 5d, volatilidad 20d
- **Pipeline:** `StandardScaler → RandomForestClassifier(100 árboles)`
- **Split:** `train_test_split(shuffle=False)` para evitar data leakage temporal
- **Serialización:** `joblib.dump()` → `model_v1.joblib`
- **Patrón Singleton:** El modelo se carga **una sola vez** en memoria
- **Auditoría:** Cada predicción se persiste en `PredictionLog` (SQLite)

---

## 🛡️ Uso de Herramientas de IA

Este proyecto fue desarrollado con apoyo de herramientas de IA (Claude de Anthropic) para:
- Estructurar el código inicial del backend y las clases de servicios
- Generar las fórmulas matemáticas de Black-Scholes y Nelson-Siegel
- Diseñar los tests unitarios con TestClient

Todo el código fue revisado, entendido y adaptado por las autoras.  
Las decisiones metodológicas (selección de modelos GARCH, parámetros CAPM, diseño del pipeline ML) son de autoría propia.

---

## 📚 Variables de Entorno

| Variable | Descripción | Dónde obtenerla |
|---|---|---|
| `FRED_API_KEY` | Clave API de FRED | https://fred.stlouisfed.org/docs/api/api_key.html |
| `TICKERS` | Activos del portafolio | Configurar según preferencia |
| `BENCHMARK` | Benchmark (ej: ^GSPC) | Yahoo Finance symbols |
| `DATABASE_URL` | URL SQLite | `sqlite:///./risklab.db` (default) |

---

## 👩‍💻 Autoras

| Nombre | Rol |
|---|---|
| Alejandra Sepúlveda | Backend FastAPI · Renta Fija · Opciones · Stress |
| Ingrid Umbacia Ramírez | ML · Frontend Streamlit · GARCH · VaR |

**Universidad Santo Tomás · Facultad de Estadística · 2025**
