FROM python:3.11-slim

# Prevent Python from writing pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and pre-trained models
COPY src/ ./src/
COPY models/ ./models/

# Expose FastAPI default port
EXPOSE 8000

# Command to run FastAPI server
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
