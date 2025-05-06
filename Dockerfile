# Use official Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY README.md .

# Set environment variable for Python
ENV PYTHONPATH=/app/src

# Entrypoint to CLI
ENTRYPOINT ["python", "-m", "src.cli"]
