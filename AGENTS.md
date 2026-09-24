# AGENTS.md — chatterbox-nano-api

Single-file FastAPI server (`app.py`, ~143 lines) wrapping the four ONNX graphs of Chatterbox-Nano for text-to-speech with voice cloning. CPU only, shipped as a Docker image. No tests, no lint, no typecheck, no CI.

**`/generate` currently returns 501.** The autoregressive LM loop was deliberately removed because the old single-forward-pass version produced garbage audio. Port the loop from the model repo's `run_onnx.py`. Startup, tokenizer, and health all work; only synthesis is missing.

- App: `app.py` — lifespan downloads the model snapshot, builds four `onnxruntime` sessions, and loads `AutoTokenizer`. Endpoints: `POST /generate` (501 until implemented) and `GET /health`.
- `Dockerfile` — `python:3.11-slim`, thread env vars pinned to 2 (2-vCPU target), installs `ffmpeg`+`libsndfile1`, `uvicorn app:app --port 8000 --workers 1`, `EXPOSE 8000`, `/health` healthcheck with `--start-period 120s`.
- `voices/` — voice-reference WAVs. Tracked via `voices/.gitkeep`; `.gitignore` excludes `voices/*.wav` so reference audio stays local and never lands in the public repo.

## Commands

```bash
# Container run (the real dev loop)
docker build -t chatterbox-nano-api .
docker run --rm -p 8000:8000 \
  -e DEFAULT_VOICE=/app/voices/ref.wav \
  -e MODEL_DIR=/app/model \
  -v "$(pwd)/voices:/app/voices" \
  -v chatterbox-model:/app/model \
  chatterbox-nano-api

# Bare metal — MUST override MODEL_DIR, it defaults to the container path
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
MODEL_DIR="$PWD/model" uvicorn app:app --port 8000 --workers 1

# Verify. /health is the only runtime gate; /generate will 501.
curl -s localhost:8000/health
```

There is no lint, typecheck, test, or formatter. `python3 -c "import ast; ast.parse(open('app.py').read())"` is the only syntax gate. First boot downloads the ONNX graphs (the bulk of the ~547 MiB repo) before the server binds.

## Implementing `generate_speech`

Port from `run_onnx.py` in `owensong/chatterbox-nano-ONNX`. Constraints already encoded in the code:

- Bind tensors **by name**. `_bind()` is defined but unused — it is the intended entry point and raises loudly on graph drift. `load_sessions()` logs every session's input/output names at startup; read that log before guessing names.
- The model is autoregressive over speech tokens with a 12-layer KV cache. Special IDs: start 6561, stop 6562, decoder silence padding 4299. `max_new_tokens` and `repetition_penalty` are accepted by the request model and threaded through to `generate_speech` — wire them to the loop, don't drop them.
- **The 24 kHz reference-audio check was lost in the rewrite.** `sf.read` is no longer called anywhere; `SAMPLE_RATE` is only used for writing output. Re-add `sf.read` + mono downmix + `if sr != SAMPLE_RATE: raise` before the encoder, or the speech encoder silently gets mismatched-rate conditioning.
- `tokenizer` is a module global loaded in lifespan. Use it; don't re-instantiate per request.

## Gotchas that will bite you

- **`MODEL_DIR` defaults to the absolute container path `/app/model`.** Not CWD-relative. On bare metal that is a permission error for a non-root user — always set `MODEL_DIR` explicitly outside Docker.
- **The tokenizer does not live in `MODEL_DIR`.** `AutoTokenizer.from_pretrained(MODEL_REPO)` uses the default HF cache (`~/.cache/huggingface`), so it re-downloads on every cold start even when `/app/model` is a persistent volume. Point `HF_HOME` at the same volume if you want start-up to be network-free.
- **`ALLOW_PATTERNS` excludes `run_onnx.py`** — the reference implementation you need to port the loop from. Fetch it explicitly (`hf download owensong/chatterbox-nano-ONNX run_onnx.py`) or read it on the model card; don't assume it is sitting in `MODEL_DIR`.
- `ALLOW_PATTERNS` does keep `onnx/*.onnx_data`, which the quantized graphs require beside their `.onnx`. Keep those sidecars; dropping them breaks session load.
- `MODEL_REVISION` defaults to `main`, not a commit hash. Pin it before relying on reproducible output — a moving tag plus `ORT_ENABLE_ALL` can change results between deploys.
- **Startup fails fast if `DEFAULT_VOICE` points at a missing file** (lifespan raises). On a volume-backed deploy this reads as a crashloop until the WAV is uploaded. If `DEFAULT_VOICE` is unset you instead get a warning and every request must carry `voice_reference`.
- No `HF_TOKEN` needed — the model repo is public. Rate limits can still make a cold start slow.

## Deploy

Target: **Coolify on a 2 vCPU / 12 GB RAM VPS**, deployed from the Dockerfile. The 2-vCPU sizing is why `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` are all `2` and workers is `1` — keep them consistent if you change vCPUs.

- Origin: public repo `nishkmg/chatterbox-nano-api`, branch `main`. Clone: `git@github.com:nishkmg/chatterbox-nano-api.git`.
- Coolify service config: port `8000`, health check path `/health`, no build command (Dockerfile handles it).
- Mount two volumes: `/app/voices` (reference WAVs) and `/app/model` (ONNX snapshot). Without the model volume the ~500 MiB download repeats on every redeploy.
- Set `DEFAULT_VOICE=/app/voices/<file>.wav`, `MODEL_DIR=/app/model`, and `HF_HOME=/app/model/.hf` as service env vars.
- The Dockerfile's `HEALTHCHECK --start-period=120s` can be shorter than a cold model download. If Coolify reports the deploy unhealthy, it is the first-boot download, not a real failure — pre-warm the model volume to avoid it.

## Local filesystem

`/Volumes/CrucialSSD` is **exFAT** (`noowners`, no POSIX perms). Files list as `rwx` regardless of real mode; `core.filemode=false` keeps git recording `100644`. Do not `chmod`/`chown`. Every file gets an AppleDouble `._<name>` shadow — excluded by `.gitignore`, but filter them out of searches (`find … ! -name '._*'`) so you don't match duplicates.
