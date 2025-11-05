FROM python:3.12-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies needed to build some Python packages and for DB backends
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	build-essential \
	git \
	curl \
	libsqlite3-dev \
	libpq-dev \
	libffi-dev \
	ca-certificates \
	&& rm -rf /var/lib/apt/lists/*

# Copy only dependency files first to leverage Docker cache
COPY pyproject.toml requirements.txt /app/

# Upgrade pip and install Python dependencies. Add torch/transformers explicitly to
# ensure CPU wheels are fetched where available for sentence-transformers usage.
RUN python -m pip install --upgrade pip setuptools wheel \
	&& pip install --no-cache-dir -r requirements.txt \
	&& pip install --no-cache-dir torch transformers || true

# Copy the rest of the application
COPY . /app

# Use a non-root user for better security
RUN useradd -m appuser || true \
	&& chown -R appuser:appuser /app

USER appuser

# The application package root is under /app/src so make that the working dir
WORKDIR /app/src

# Env var for GROQ API
ARG GROQ_API_KEY
ENV GROQ_API_KEY=${GROQ_API_KEY}

RUN python data_ingestion_pipeline.py
RUN python model_training_pipeline.py
RUN python model_evaluation_pipeline.py
RUN python model_inference_pipeline.py
RUN python rag_pipeline.py

# Default port used by the API (matches config/params.yaml)
EXPOSE 8000

# Run the API using uvicorn. We import the app module from the package root (src)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
