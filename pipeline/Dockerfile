# Cloud Run Job: Data Validation and Conversion
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements-pipeline.txt .
RUN pip install --no-cache-dir -r requirements-pipeline.txt

# Copy pipeline code
COPY scripts/data_utils.py scripts/
COPY pipeline/ pipeline/

# Set entrypoint
ENTRYPOINT ["python", "-m", "pipeline"]
