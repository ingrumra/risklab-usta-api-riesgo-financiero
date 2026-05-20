# =============================================================================
# Dockerfile – Multi-stage build
# Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez
#
# Stage 1 (builder): instala dependencias y compila wheels
# Stage 2 (runtime): imagen final liviana ~150 MB (vs ~600 MB sin multi-stage)
# Base: python:3.11.9-slim-bookworm (requerimiento del proyecto)
# =============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.11.9-slim-bookworm AS builder

WORKDIR /build

# Instalar dependencias de sistema para compilar paquetes C
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero (capa de caché)
COPY requirements.txt .

# Instalar en directorio local (wheel cache para stage 2)
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11.9-slim-bookworm AS runtime

WORKDIR /app

# Copiar solo los paquetes instalados del stage 1
COPY --from=builder /install /usr/local

# Copiar el código fuente del backend
COPY backend/app ./app
COPY requirements.txt .

# Variables de entorno de producción
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Puerto expuesto
EXPOSE 8000

# Crear usuario no-root por seguridad
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')"

# Comando de inicio
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
