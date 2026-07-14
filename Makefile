.PHONY: install install-llm run run-cli test smoke clean help

PYTHON ?= python3
VIDEO ?=

help:
	@echo "TempoGraph v2 — Make targets"
	@echo ""
	@echo "  make install          Bootstrap: pip deps + whisper.cpp + base.en model (~1 GB total)"
	@echo "  make install-llm      ALSO: build llama.cpp + download Qwen3.5-VL-9B (~10 GB more)"
	@echo "  make run              Launch the Streamlit UI on http://localhost:8501"
	@echo "  make run-cli VIDEO=clip.mp4   Run the CLI on a video"
	@echo "  make test             Run the unit tests"
	@echo "  make smoke            Run the end-to-end smoke test (skip-VLM)"
	@echo "  make clean            Remove __pycache__/ and *.pyc"

install:
	bash bootstrap.sh

install-llm:
	bash bootstrap.sh --with-llm

run:
	$(PYTHON) -m streamlit run ui/app.py

run-cli:
	@if [ -z "$(VIDEO)" ]; then \
	  echo "usage: make run-cli VIDEO=path/to/video.mp4"; exit 2; \
	fi
	$(PYTHON) -m src.pipeline_v2 \
	  --video "$(VIDEO)" \
	  --output "results/$$(basename $(VIDEO))" \
	  --camera auto --vlm-frame-mode keyframes \
	  --audio --whisper-model base.en \
	  --vlm-autostart-service qwen-tempograph.service --vlm-autostop

test:
	$(PYTHON) -m pytest tests/ -v

smoke:
	$(PYTHON) scripts/smoke_test_v2.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
