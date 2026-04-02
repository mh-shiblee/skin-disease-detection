import subprocess
import sys
import os

# Start FastAPI backend
backend = subprocess.Popen([
    sys.executable, "-m", "uvicorn",
    "backend.main:app",
    "--host", "0.0.0.0",
    "--port", "8000"
])

# Start Streamlit frontend
frontend = subprocess.Popen([
    sys.executable, "-m", "streamlit",
    "run", "frontend/app.py",
    "--server.port", "8501",
    "--server.address", "0.0.0.0",
    "--server.headless", "true"
])

backend.wait()
frontend.wait()
