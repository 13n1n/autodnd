# Use Python 3.13 as base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Install dependencies using uv (excluding dev dependencies for production)
RUN uv sync --frozen --no-dev

# Copy application code
COPY . .

# Create directory for game state dumps (if needed)
RUN mkdir -p /var/autodnd && chmod 777 /var/autodnd

# Expose port 5000
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV AUTODND_DUMP_DIR=/var/autodnd
ENV FLASK_APP=main.py

# Run the application using waitress (production WSGI server)
CMD ["uv", "run", "python", "-m", "autodnd"]

