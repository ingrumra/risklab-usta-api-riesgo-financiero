# tests/conftest.py – Configuración pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

# Override the database URL to use an in-memory SQLite for tests
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_risklab.db")
os.environ.setdefault("FRED_API_KEY", "")
