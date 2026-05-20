#!/usr/bin/env bash
# ============================================================
# run_backend.sh – Inicia el backend FastAPI
# Ejecutar desde la raíz del proyecto: bash run_backend.sh
# Terminal 1 de Antigravity IDE
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"

echo "============================================================"
echo " RiskLab USTA – Backend FastAPI"
echo " Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez"
echo "============================================================"

# Verificar Python 3.11
PYTHON_VERSION=$(python3 --version 2>&1 | grep -o '3\.[0-9]*')
if [[ "$PYTHON_VERSION" != "3.11" && "$PYTHON_VERSION" != "3.12" ]]; then
    echo "⚠️  Se requiere Python 3.11 (detectado: $PYTHON_VERSION)"
fi

# Verificar entorno virtual
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Creando entorno virtual..."
    python3.11 -m venv "$SCRIPT_DIR/.venv"
fi

# Activar entorno
source "$SCRIPT_DIR/.venv/bin/activate"

# Instalar dependencias
echo "Instalando dependencias..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Copiar .env si no existe
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    cp "$SCRIPT_DIR/.env.example" "$SCRIPT_DIR/.env"
    echo "⚠️  Se creó .env desde .env.example. Agrega tu FRED_API_KEY si la tienes."
fi

echo ""
echo "✅ Iniciando backend en http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo "   Redoc: http://localhost:8000/redoc"
echo ""

cd "$BACKEND_DIR"
uvicorn app.main:app --reload --port 8000 --host 127.0.0.1
