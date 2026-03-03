#!/bin/sh

# 1. Start the FastAPI Backend in the background (&)
# We use 'python -m uvicorn' to ensure it finds the API folder correctly
python -m uvicorn API.main:app --host 0.0.0.0 --port 8000 &

# 2. Start the Streamlit Frontend
streamlit run API/app.py --server.port 8501 --server.address 0.0.0.0