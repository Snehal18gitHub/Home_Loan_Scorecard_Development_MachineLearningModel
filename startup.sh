
#!/bin/bash

gunicorn --bind 127.0.0.1:5000 --timeout 600 flask_api:app &

exec python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8000 --server.headless true
