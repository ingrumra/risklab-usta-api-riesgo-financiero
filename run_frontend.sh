#!/usr/bin/env bash
# ============================================================
# run_frontend.sh – Inicia el tablero Streamlit
# Ejecutar desde la raíz del proyecto: bash run_frontend.sh
# Terminal 2 de Antigravity IDE
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "============================================================"
echo " RiskLab USTA – Frontend Streamlit"
echo " Autoras: Alejandra Sepúlveda · Ingrid Umbacia Ramírez"
echo "============================================================"

# Activar entorno virtual
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
else
    echo "❌ Entorno virtual no encontrado. Ejecuta primero run_backend.sh"
    exit 1
fi

echo ""
echo "⚠️  Asegúrate de que el backend esté corriendo en http://localhost:8000"
echo "   (ejecuta run_backend.sh en otra terminal)"
echo ""
echo "✅ Iniciando tablero en http://localhost:8501"
echo ""

cd "$FRONTEND_DIR"
streamlit run app.py \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false \
    --theme.base light
