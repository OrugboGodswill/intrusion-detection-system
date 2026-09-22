#!/bin/bash
set -e

echo "Starting Sentinel IDS Backend API on port 5000..."
python backend/app.py &

echo "Waiting for Backend API to initialize..."
sleep 3

PORT="${PORT:-8501}"
echo "Starting Sentinel IDS Streamlit Frontend on port ${PORT}..."
exec streamlit run frontend/streamlit_app.py --server.port="${PORT}" --server.address=0.0.0.0
