
#!/bin/bash
set -e

echo "Starting Flask prediction API..."
gunicorn --bind 127.0.0.1:5000 --timeout 600 flask_api:app &

echo "Starting Streamlit frontend..."
exec streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8000 \
    --server.headless true
