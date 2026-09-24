# AGENTS.md — chatterbox-nano-api

Single-file FastAPI server (`app.py`, ~176 lines) that wraps the four ONNX graphs of Chatterbox-Nano for text-to-speech with voice cloning. Runs on CPU only. Shipped as a Docker image; no package manager, no build step, no tests, no lint, no CI.

- App: `app.py` — FastAPI app, lifespan loads the ONNX sessions on startup. Endpoints: `POST /generate` (TTS → WAV stream) and `GET /health`.
- `Dockerfile` — `python:3.11-slim`, pins thread env vars to 2 (2-vCPU target), installs `ffmpeg`+`libsndfile1`, runs `uvicorn app:app --port 8000 --workers 1`, `EXPOSE 8000`, `/health` healthcheck (start-period 120s).
- `requirements.txt` — runtime deps only.
- `voices/` — intended home for voice-reference WAVs. Currently empty and not tracked by git (empty dirs aren't).

## Commands

```bash
# Container run (the real dev loop). Model downloads on first start (~547 MiB).
docker build -t chatterbox-nano-api .
docker run --rm -p 8000:8000 \
  -e DEFAULT_VOICE=/app/voices/ref.wav \
  -v "$(pwd)/voices:/app/voices" \
  chatterbox-nano-api

# Bare-metal run (must match the Dockerfile's working dir; see model-dir gotcha)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 8000 --workers 1

# Smoke test once /health reports the sessions
curl -s localhost:8000/health
curl -s -X POST localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"text":"Hello.","voice_reference":"/app/voices/ref.wav"}' \
  -o out.wav
```

There is no lint, typecheck, test, or formatter. `python3 -c "import ast; ast.parse(open('app.py').read())"` is the only syntax gate; `GET /health` returning the four session names is the only runtime gate.

## Gotchas that will bite you

- **`transformers` is missing from `requirements.txt`.** `generate_speech` does `from transformers import AutoTokenizer`; on a clean install every `/generate` fails with `Tokenizer not available...`. Add `transformers` before trusting the endpoint. The model card's own reference requirements also pin `transformers`.
- **The LM step is a single forward pass, not the real pipeline.** The model is autoregressive over speech tokens with a 12-layer KV cache and start/stop IDs (6561/6562, silence 4299). `app.py` calls `language_model.run(None, lm_inputs)` once and feeds its output straight to the decoder. Code comments admit it's "simplified". Don't assume output is valid speech; fixing it means implementing the token loop.
- **Input tensor names are assumed by position** (`get_inputs()[0]`, `[1]`, …). If the model repo or graph order changes, this silently feeds the wrong tensors. Query the session's `get_inputs()` names/types instead.
- **Model download is CWD-relative.** `snapshot_download(local_dir="model")` writes to `./model`, so `/app/model` in the container but wherever you launched from locally. Also downloads the **whole** repo (~547 MiB) including `.onnx_data` sidecars required by the large graphs — don't trim or ignore those.
- **No token, no cache reuse across dirs.** `HF_TOKEN` only matters for private/gated repos; snapshot cache is per-`local_dir`. First start is slow; `/health` is 503/"Model not loaded yet" until lifespan finishes.
- **Voice reference must be 24 kHz** (`SAMPLE_RATE`) or `generate_speech` raises; stereo is downmixed to mono. Point `DEFAULT_VOICE` (or `voice_reference`) at a real file — the `reference.wav` default does not exist in the repo, so an unconfigured `/generate` returns 400.

## Deploy

- Origin is a public GitHub repo, `nishkmg/chatterbox-nano-api` (this working dir). Clone URL: `git@github.com:nishkmg/chatterbox-nano-api.git`.
- The volume `/Volumes/CrucialSSD` is **exFAT** (`noowners`, no POSIX perms). Files list as `rwx` locally regardless of real mode; `git init` should set `core.filemode=false`. Do not `chmod`/`chown`; verify committed modes with `git ls-files -s` (want `100644`, not `100755`).
- exFAT also drops an AppleDouble `._<name>` shadow next to every file (you'll see `._app.py`, `._Dockerfile`, `._AGENTS.md`). `.gitignore` excludes them; filter with `find … ! -name '._*'` when searching so you don't match duplicates.
