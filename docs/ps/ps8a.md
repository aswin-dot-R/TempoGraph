# PS8a — Ship hardening: portability + config (35B lane)

TempoGraph must run on a stranger's machine. Today it cannot: server
URLs and whisper paths are hardcoded to this workstation, requirements
contains an uninstallable pin, and same-named videos overwrite each
other's results. Tests in parallel (`docs/ps/ps8b-tests` section of
`docs/ps/ps8b.md` — the 9B writes `tests/test_settings.py`).

## Scope fence

- **Files you may create:** `src/settings.py`
- **Files you may modify:** `src/pipeline_v2.py`, `src/aggregator.py`,
  `src/summarizer.py`, `ui/app.py`, `ui/pages/Results.py`,
  `src/modules/whisper_cpp.py`, `requirements.txt`
- No test files.

## Forbidden

`sudo`, `systemctl`, `pip install`, git push, starting/stopping any
llama-server. Interpreter: `/home/ashie/anaconda3/bin/python3`.

## FROZEN CONTRACTS

New `src/settings.py` — env-var driven, zero deps:

```python
@dataclass(frozen=True)
class Settings:
    vlm_url: str          # TEMPOGRAPH_VLM_URL,       default "http://127.0.0.1:8085"
    vlm_model: str        # TEMPOGRAPH_VLM_MODEL,     default "ornith-1.0-9b-Q4_K_M.gguf"
    walker_url: str       # TEMPOGRAPH_WALKER_URL,    default = vlm_url
    verifier_url: str     # TEMPOGRAPH_VERIFIER_URL,  default "http://127.0.0.1:8096"
    whisper_binary: str   # TEMPOGRAPH_WHISPER_BIN,   default "~/whisper.cpp/build/bin/whisper-cli" (expanduser)
    whisper_model_dir: str # TEMPOGRAPH_WHISPER_MODELS, default "~/whisper.cpp/models" (expanduser)
    results_dir: str      # TEMPOGRAPH_RESULTS_DIR,   default "results"

def get_settings() -> Settings:
    """Read env each call (no caching — tests monkeypatch os.environ)."""
```

## Tasks

1. **`src/settings.py`** per contract.
2. **Thread it through**: every hardcoded `http://127.0.0.1:8082/8085/8096`,
   `/home/ashie/whisper.cpp/...` in the scoped files becomes a
   `get_settings()` default (constructor params keep working — settings
   only replace the hardcoded fallback defaults, so explicit kwargs and
   all existing tests stay valid).
3. **Run-dir collisions (footgun #3)**: in the UI drop flow, if
   `results/<video_name>` already contains a `tempograph.db` from a
   DIFFERENT source (compare `run_meta.video_path`), suffix the new run
   dir with `-2`, `-3`, ... Same source path keeps the same dir (resume
   still works).
4. **`requirements.txt`**: remove the unsatisfiable
   `depth-anything-v2>=1.0.0` line (footgun #1); add a commented
   `# transformers>=4.40  # optional: enables the depth stage`.
5. Grep-audit: after your changes,
   `grep -rn "home/ashie" src/ ui/ --include=*.py` must return ONLY
   lines inside `src/settings.py` defaults (the expanduser fallbacks
   there must use `~`, so ideally zero hits).

## ACCEPTANCE (paste output)

```bash
/home/ashie/anaconda3/bin/python3 -m pytest -q      # suite green
grep -rn "home/ashie" src/ ui/ --include=*.py | grep -v settings.py | wc -l   # 0
TEMPOGRAPH_VLM_URL=http://example:1234 /home/ashie/anaconda3/bin/python3 -c "
from src.settings import get_settings; print(get_settings().vlm_url)"        # http://example:1234
grep -c "depth-anything" requirements.txt                                     # 0
timeout 12 /home/ashie/anaconda3/bin/python3 -m streamlit run ui/app.py \
  --server.headless true --server.port 8599 & sleep 8; curl -sf localhost:8599 >/dev/null && echo UI_BOOTS
/home/ashie/anaconda3/bin/python3 -m black src/ ui/
git status --porcelain
```

(do not commit — gate reviewer merges both lanes)
