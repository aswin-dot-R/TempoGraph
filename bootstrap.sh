#!/usr/bin/env bash
# TempoGraph v2 — one-shot bootstrap.
#
# Detects what's already on the system, installs only what's missing.
# Idempotent — safe to re-run.
#
# Usage:
#   bash bootstrap.sh                     # core install (Python deps + whisper.cpp + base.en model)
#   bash bootstrap.sh --with-llm          # also build llama-cpp + download Qwen3.5-VL-9B (~10 GB)
#   bash bootstrap.sh --whisper-model X   # change default whisper model (e.g. small.en)
#   bash bootstrap.sh --no-build          # skip whisper.cpp build (use existing or CPU-only fallback)

set -euo pipefail

# ─── args ──────────────────────────────────────────────────────────────
WITH_LLM=0
NO_BUILD=0
WHISPER_MODEL="base.en"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-llm)        WITH_LLM=1; shift ;;
    --no-build)        NO_BUILD=1; shift ;;
    --whisper-model)   WHISPER_MODEL="$2"; shift 2 ;;
    -h|--help)
      grep -E "^# " "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

cd "$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$PWD"

log()  { printf '\033[36m[bootstrap]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[bootstrap]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[31m[bootstrap]\033[0m %s\n' "$*" >&2; exit 1; }

# ─── 0. system prereqs ─────────────────────────────────────────────────
log "Checking system prerequisites..."
need_pkg=()
for cmd in git python3 ffmpeg cmake; do
  command -v "$cmd" >/dev/null 2>&1 || need_pkg+=("$cmd")
done
if (( ${#need_pkg[@]} )); then
  err "Missing system commands: ${need_pkg[*]}. Install via apt/brew/etc., then re-run."
fi
log "  ✓ git, python3, ffmpeg, cmake all present"

# ─── 1. python env ─────────────────────────────────────────────────────
PYTHON="${TEMPOGRAPH_PYTHON:-python3}"
log "Using Python: $($PYTHON --version) at $(command -v $PYTHON)"

PY_VER="$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
log "  Python $PY_VER"

# Detect or create venv
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  log "  ✓ conda env active: $(basename "$CONDA_PREFIX")"
elif [[ -d ".venv" ]]; then
  log "  ✓ existing .venv found, activating"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PYTHON="$(command -v python)"
else
  log "  no env detected — creating .venv"
  $PYTHON -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PYTHON="$(command -v python)"
fi

# ─── 2. python deps ────────────────────────────────────────────────────
log "Installing Python deps (pip install -r requirements.txt)..."
$PYTHON -m pip install --upgrade pip --quiet
$PYTHON -m pip install -r requirements.txt --quiet
$PYTHON -m pip install pytest --quiet
log "  ✓ pip install complete"

# Verify torch GPU access
$PYTHON - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        print(f"[bootstrap]   ✓ torch sees CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[bootstrap]   ⚠ torch installed but no CUDA — YOLO/Depth will run on CPU (slow).")
except ImportError:
    print("[bootstrap]   ⚠ torch not installed by requirements.txt — please pip install torch manually.")
PY

# ─── 3. whisper.cpp ────────────────────────────────────────────────────
WHISPER_DIR="${TEMPOGRAPH_WHISPER_DIR:-$HOME/whisper.cpp}"
WHISPER_BIN="$WHISPER_DIR/build/bin/whisper-cli"
WHISPER_MODEL_PATH="$WHISPER_DIR/models/ggml-${WHISPER_MODEL}.bin"

if (( NO_BUILD == 0 )); then
  if [[ ! -d "$WHISPER_DIR" ]]; then
    log "Cloning whisper.cpp into $WHISPER_DIR..."
    git clone --depth 1 https://github.com/ggml-org/whisper.cpp "$WHISPER_DIR"
  else
    log "  ✓ whisper.cpp already cloned at $WHISPER_DIR"
  fi

  if [[ ! -x "$WHISPER_BIN" ]]; then
    # Backend selection: prefer Vulkan > CUDA > CPU
    BACKEND_FLAG=""
    if command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary 2>/dev/null | grep -q "deviceName"; then
      BACKEND_FLAG="-DGGML_VULKAN=1"
      log "  Vulkan detected — building whisper.cpp with -DGGML_VULKAN=1"
    elif command -v nvcc >/dev/null 2>&1; then
      BACKEND_FLAG="-DGGML_CUDA=1"
      log "  CUDA detected — building whisper.cpp with -DGGML_CUDA=1"
    else
      log "  no GPU backend — building whisper.cpp CPU-only"
    fi
    log "Building whisper.cpp (this takes 1–3 minutes)..."
    cmake -B "$WHISPER_DIR/build" -S "$WHISPER_DIR" $BACKEND_FLAG > /tmp/whisper-cmake.log 2>&1 \
      || { tail -30 /tmp/whisper-cmake.log; err "whisper.cpp configure failed; see /tmp/whisper-cmake.log"; }
    cmake --build "$WHISPER_DIR/build" -j --config Release > /tmp/whisper-build.log 2>&1 \
      || { tail -30 /tmp/whisper-build.log; err "whisper.cpp build failed; see /tmp/whisper-build.log"; }
    log "  ✓ whisper-cli built at $WHISPER_BIN"
  else
    log "  ✓ whisper-cli already built at $WHISPER_BIN"
  fi
else
  log "Skipping whisper.cpp build (--no-build)"
fi

# Whisper model download
if [[ ! -f "$WHISPER_MODEL_PATH" ]]; then
  log "Downloading whisper model '$WHISPER_MODEL' (~75 MB to ~3 GB)..."
  ( cd "$WHISPER_DIR" && bash models/download-ggml-model.sh "$WHISPER_MODEL" )
  log "  ✓ model at $WHISPER_MODEL_PATH"
else
  log "  ✓ whisper model '$WHISPER_MODEL' already downloaded"
fi

# ─── 4. optional: llama-cpp + Qwen3.5-VL ───────────────────────────────
if (( WITH_LLM == 1 )); then
  log "--with-llm: setting up llama.cpp + Qwen3.5-VL-9B (~10 GB) ..."
  LLAMA_DIR="${TEMPOGRAPH_LLAMA_DIR:-$HOME/llama.cpp}"
  MODEL_DIR="${TEMPOGRAPH_QWEN_DIR:-$HOME/qwen-models}"
  mkdir -p "$MODEL_DIR"

  if [[ ! -d "$LLAMA_DIR" ]]; then
    log "  Cloning llama.cpp into $LLAMA_DIR..."
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  fi

  if [[ ! -x "$LLAMA_DIR/build/bin/llama-server" ]]; then
    BACKEND=""
    if command -v vulkaninfo >/dev/null 2>&1; then
      BACKEND="-DGGML_VULKAN=1"
      log "  building llama.cpp with Vulkan"
    elif command -v hipcc >/dev/null 2>&1; then
      BACKEND="-DGGML_HIP=1"
      log "  building llama.cpp with HIP (AMD ROCm)"
    elif command -v nvcc >/dev/null 2>&1; then
      BACKEND="-DGGML_CUDA=1"
      log "  building llama.cpp with CUDA"
    else
      log "  building llama.cpp CPU-only (slow inference)"
    fi
    cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" $BACKEND > /tmp/llama-cmake.log 2>&1 \
      || { tail -30 /tmp/llama-cmake.log; err "llama.cpp configure failed"; }
    cmake --build "$LLAMA_DIR/build" -j --target llama-server --config Release > /tmp/llama-build.log 2>&1 \
      || { tail -30 /tmp/llama-build.log; err "llama.cpp build failed"; }
    log "  ✓ llama-server built at $LLAMA_DIR/build/bin/llama-server"
  fi

  QWEN_GGUF="$MODEL_DIR/Qwen3.5-9B-Q8_0.gguf"
  QWEN_MMPROJ="$MODEL_DIR/mmproj-Qwen3.5-9B-BF16.gguf"
  if [[ ! -f "$QWEN_GGUF" || ! -f "$QWEN_MMPROJ" ]]; then
    if ! command -v huggingface-cli >/dev/null 2>&1; then
      $PYTHON -m pip install huggingface_hub --quiet
    fi
    log "  Downloading Qwen3.5-VL-9B from HuggingFace (~9.5 GB + 600 MB mmproj)..."
    huggingface-cli download lmstudio-community/Qwen3.5-9B-GGUF \
      Qwen3.5-9B-Q8_0.gguf mmproj-Qwen3.5-9B-BF16.gguf \
      --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
    log "  ✓ Qwen weights at $MODEL_DIR"
  else
    log "  ✓ Qwen weights already present at $MODEL_DIR"
  fi

  # Generate a systemd --user unit pointing at these paths
  UNIT_PATH="$HOME/.config/systemd/user/qwen-tempograph.service"
  if [[ ! -f "$UNIT_PATH" ]]; then
    mkdir -p "$(dirname "$UNIT_PATH")"
    cat > "$UNIT_PATH" <<UNIT
[Unit]
Description=Qwen3.5-VL llama-server for TempoGraph
After=network.target

[Service]
Type=simple
ExecStart=$LLAMA_DIR/build/bin/llama-server \\
  -m $QWEN_GGUF \\
  --mmproj $QWEN_MMPROJ \\
  -ngl 99 -c 100000 -np 1 \\
  --host 0.0.0.0 --port 8082
Restart=on-failure
RestartSec=10
WorkingDirectory=$LLAMA_DIR

[Install]
WantedBy=default.target
UNIT
    systemctl --user daemon-reload
    log "  ✓ systemd unit created at $UNIT_PATH"
    log "    enable + start it with: systemctl --user enable --now qwen-tempograph.service"
  else
    log "  ✓ systemd unit already exists at $UNIT_PATH"
  fi
else
  log "Skipping LLM setup (use --with-llm to install Qwen3.5-VL + llama.cpp)."
  log "  Without it: pipeline can run YOLO/Depth/Audio + frame selection but VLM stage will fail unless an Ollama-compatible server is reachable on port 8082."
fi

# ─── 5. .streamlit/config.toml (4 GB upload cap) ───────────────────────
mkdir -p .streamlit
if [[ ! -f .streamlit/config.toml ]]; then
  cat > .streamlit/config.toml <<'TOML'
[server]
maxUploadSize = 4096
maxMessageSize = 4096

[browser]
gatherUsageStats = false
TOML
  log "  ✓ .streamlit/config.toml created"
fi

# ─── 6. summary ────────────────────────────────────────────────────────
cat <<EOF

\033[32m✓ TempoGraph v2 bootstrap complete\033[0m

  Python:    $($PYTHON --version) at $(command -v $PYTHON)
  whisper.cpp: $WHISPER_BIN
  whisper model: $WHISPER_MODEL ($WHISPER_MODEL_PATH)
$([ "$WITH_LLM" -eq 1 ] && echo "  llama-server: $LLAMA_DIR/build/bin/llama-server" || echo "  LLM:       not installed (re-run with --with-llm to set up)")

Run the UI:
  make run

Run the CLI on a video:
  make run-cli VIDEO=clip.mp4

Run the test suite:
  make test
EOF
