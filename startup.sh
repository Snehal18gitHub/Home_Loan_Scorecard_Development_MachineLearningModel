#!/bin/bash
set -e

echo "Starting Flask prediction API..."
gunicorn --bind 0.0.0.0:5000 --timeout 600 --access-logfile - --error-logfile - flask_api:app &

echo "Starting Streamlit frontend..."
exec python -m streamlit run app.py 
  --server.address 0.0.0.0 
  --server.port 8000 
  --server.headless true