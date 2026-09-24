# AGENTS.md — chatterbox-nano-api

Single-file FastAPI server (`app.py`, ~240 lines) wrapping the four ONNX graphs of Chatterbox-Nano for text-to-speech with voice cloning. CPU only, shipped as a Docker image.

- App: `app.py` — lifespan downloads the model snapshot, builds four `onnxruntime` sessions, loads `GPT2TokenizerFast`. Endpoints: `POST /generate` (TTS → WAV stream) and `GET /health`.
- `tools/verify_pipeline.py` — offline checks for the generation math (KV cache, sampling, penalty). No model download needed. This is the only real test gate; run it after touching `generate_speech`.
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

# Verify. No model download required.
python3 tools/verify_pipeline.py

# Runtime smoke test once /health reports all four sessions
curl -s localhost:8000/health
curl -s -X POST localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"text":"Hello.","voice_reference":"'"$PWD"'/voices/ref.wav","seed":1337}' \
  -o out.wav
```

There is no lint, typecheck, formatter, or CI. `tools/verify_pipeline.py` plus `GET /health` are the only gates. First boot downloads the ONNX graphs (the bulk of the ~547 MiB repo) before the server binds.

## The generation loop

Ported from `run_onnx.py` in `owensong/chatterbox-nano-ONNX` (now fetched into `MODEL_DIR` and gitignored). If you change it, re-run `tools/verify_pipeline.py` — it pins the parts that are easy to break:

- **Cache output offset is `outputs[index + 1]`.** LM outputs are `[logits, k0, v0, k1, v1, …]`, so the first cache tensor is `outputs[1]`, not `outputs[0]`. Off-by-one here silently feeds every layer the wrong KV tensor and yields garbage audio with no error.
- **The first pass feeds `audio_features + text_embeds` concatenated** (the reference conditioning prefix); every later pass feeds exactly **one** token. `attention_mask` and `position_ids` must grow in lockstep with the cache.
- **`_empty_cache` expects exactly 24 inputs** (12 layers × K,V) named `past_key_values.*`, each seeded as `(batch, heads, 0, head_dim)` — zero-length sequence. It raises on any other count so graph drift is loud.
- Special IDs: start 6561, stop 6562, decoder silence padding 4299. On STOP, the token is stripped and 3 silence tokens are appended to the decoder input.
- Tensors bind **by name** through `_bind()`, which raises if a name is missing. `load_sessions()` logs every session's input/output names at startup — read that log before guessing.

## Gotchas that will bite you

- **`MODEL_DIR` defaults to the absolute container path `/app/model`.** Not CWD-relative. On bare metal that is a permission error for a non-root user — always set `MODEL_DIR` explicitly outside Docker.
- **Generation is serialized by a lock and run via `run_in_threadpool`.** ONNX Runtime sessions aren't safe for concurrent `run()`, and a 256-token autoregressive loop is slow enough to starve the event loop if called inline. Keep both. Throughput is one request at a time by design; raise `max_new_tokens` before adding workers.
- **The voice reference must be 24 kHz.** `generate_speech` reads it with `sf.read`, downmixes stereo, and raises on any other rate. The encoder has no resampling stage — resample upstream rather than removing the check.
- **`MODEL_REVISION` defaults to `main`, not a commit hash.** Pin it before relying on reproducible output; a moving tag plus `ORT_ENABLE_ALL` can change results between deploys.
- **Startup fails fast if `DEFAULT_VOICE` points at a missing file** (lifespan raises). On a volume-backed deploy this reads as a crashloop until the WAV is uploaded. If unset you get a warning and every request must carry `voice_reference`.
- Sampling is stochastic by default (`temperature=0.8`). Pass `seed` for reproducible output, or `temperature=0` for greedy. `temperature`, `top_k`, `top_p`, and `seed` are request fields.
- No `HF_TOKEN` needed — the model repo is public. Rate limits can still make a cold start slow.

## Deploy

Target: **Coolify on a 2 vCPU / 12 GB RAM VPS**, deployed from the Dockerfile. The 2-vCPU sizing is why `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/`OPENBLAS_NUM_THREADS` are all `2` and workers is `1` — keep them consistent if you change vCPUs.

- Origin: public repo `nishkmg/chatterbox-nano-api`, branch `main`. Clone: `git@github.com:nishkmg/chatterbox-nano-api.git`.
- Coolify service config: port `8000`, health check path `/health`, no build command (Dockerfile handles it).
- Mount two volumes: `/app/voices` (reference WAVs) and `/app/model` (ONNX snapshot + `.hf` cache). Without the model volume the ~500 MiB download repeats on every redeploy.
- Set `DEFAULT_VOICE=/app/voices/<file>.wav`, `MODEL_DIR=/app/model`, `HF_HOME=/app/model/.hf`, and a pinned `MODEL_REVISION` as service env vars.
- The Dockerfile's `HEALTHCHECK --start-period=120s` can be shorter than a cold model download. If Coolify reports the deploy unhealthy, it is the first-boot download, not a real failure — pre-warm the model volume to avoid it.

## Local filesystem

`/Volumes/CrucialSSD` is **exFAT** (`noowners`, no POSIX perms). Files list as `rwx` regardless of real mode; `core.filemode=false` keeps git recording `100644`. Do not `chmod`/`chown`. Every file gets an AppleDouble `._<name>` shadow — excluded by `.gitignore`, but filter them out of searches (`find … ! -name '._*'`) so you don't match duplicates.
