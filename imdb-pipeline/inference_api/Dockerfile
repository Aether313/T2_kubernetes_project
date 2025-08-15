FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir tensorflow joblib

# Copy application code
COPY app.py .
COPY inference.py .

# Copy models/preproc data
COPY data /data

# Expose port 8002 for FastAPI
EXPOSE 8002

# Start FastAPI using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]







