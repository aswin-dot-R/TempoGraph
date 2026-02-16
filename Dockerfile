FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update && apt-get install -y python3.10 python3.10-venv python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /home/appuser && \
    chmod 755 /home/appuser

# Switch to the user
USER appuser
WORKDIR /home/appuser

# Install Python dependencies
COPY requirements.txt .
RUN python3.10 -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy source code and configs
COPY src/ ./src/
COPY configs/ ./configs/
COPY ui/ ./ui/
COPY tests/ ./tests/

# Expose ports
EXPOSE 8000 8501

# Set default command to run both API and UI
CMD bash -c "uvicorn src.api:app --host 0.0.0.0 --port 8000 & streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0"