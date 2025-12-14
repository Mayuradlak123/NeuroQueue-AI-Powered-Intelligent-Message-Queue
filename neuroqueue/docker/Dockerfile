FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Project
COPY . .

# Environment
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose
EXPOSE 8000

# Run
CMD ["python", "main.py"]
