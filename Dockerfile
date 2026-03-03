FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything (API, Model, Data, etc.)
COPY . .

# Set environment variable so Python finds your modules
ENV PYTHONPATH=/app/API
# Tell Streamlit where the API is (inside the same container, it's localhost)
ENV API_URL=http://localhost:8000

# Fix line endings and permissions for the script
RUN chmod +x start.sh

EXPOSE 8000
EXPOSE 8501

CMD ["./start.sh"]