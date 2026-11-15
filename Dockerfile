# 3.12: pandas 3 e numpy 2.4 (requirements travados do IN1165) exigem Python >= 3.11
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose API port
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "mapi_ai.app:app", "--host", "0.0.0.0", "--port", "8000"]
