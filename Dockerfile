# TempoGraph Docker image
# VLM stays external — set TEMPOGRAPH_VLM_URL=http://your-llm:8085 at docker run.
# Build whisper.cpp (CPU) inside the image so the pipeline doesn't need the host GPU.

FROM python:3.12-slim

# System deps: ffmpeg for video encoding + build tools for whisper.cpp
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Build whisper.cpp (CPU backend) at /opt/whisper.cpp
RUN git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git /tmp/whisper.cpp && \
    cd /tmp/whisper.cpp && \
    mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_BACKEND=cpu -DGGML_BLAS=OFF && \
    make -j"$(nproc)" && \
    cp whisper-cli /opt/whisper.cpp/build/bin/whisper-cli && \
    cd / && rm -rf /tmp/whisper.cpp

# Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Copy the TempoGraph source
COPY . /app

# Point whisper binary at the CPU build
ENV TEMPOGRAPH_WHISPER_BIN=/opt/whisper.cpp/build/bin/whisper-cli

# Streamlit web UI port
EXPOSE 8501

WORKDIR /app

ENTRYPOINT ["streamlit", "run", "ui/app.py"]
