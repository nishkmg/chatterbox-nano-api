import os
import io
import numpy as np
import soundfile as sf
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager

MODEL_REPO = os.getenv("MODEL_REPO", "owensong/chatterbox-nano-ONNX")
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")   # pin a commit hash for reproducibility
MODEL_DIR = os.path.abspath(os.getenv("MODEL_DIR", "/app/model"))
SAMPLE_RATE = 24000
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

# Only fetch what we need. .onnx_data sidecars ARE required for quantized graphs.
ALLOW_PATTERNS = [
    "onnx/*.onnx",
    "onnx/*.onnx_data",
    "tokenizer*",
    "vocab*",
    "merges*",
    "special_tokens*",
    "*.json",
]

MODEL_FILES = {
    "embed_tokens":        "embed_tokens_fp16.onnx",
    "speech_encoder":      "speech_encoder_q4f16.onnx",
    "language_model":      "language_model_q4f16.onnx",
    "conditional_decoder": "conditional_decoder_q4.onnx",
}

sessions: dict[str, ort.InferenceSession] = {}
tokenizer = None

def load_sessions():
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR,
        allow_patterns=ALLOW_PATTERNS,
    )
    onnx_dir = os.path.join(MODEL_DIR, "onnx")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.getenv("OMP_NUM_THREADS", "2"))
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    loaded = {}
    for name, filename in MODEL_FILES.items():
        path = os.path.join(onnx_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing ONNX graph: {path}")
        sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        loaded[name] = sess
        # Log signatures so misroutes are visible at startup, not at inference time.
        print(f"[{name}] inputs : {[i.name for i in sess.get_inputs()]}")
        print(f"[{name}] outputs: {[o.name for o in sess.get_outputs()]}")
    return loaded

@asynccontextmanager
async def lifespan(app: FastAPI):
    global sessions, tokenizer
    sessions = load_sessions()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)

    # Fail fast on missing voice reference.
    default_voice = os.getenv("DEFAULT_VOICE")
    if default_voice and not os.path.exists(default_voice):
        raise RuntimeError(f"DEFAULT_VOICE set to {default_voice} but file not found")
    if not default_voice:
        print("WARNING: DEFAULT_VOICE not set; every /generate must supply voice_reference")

    yield
    sessions.clear()

app = FastAPI(title="Chatterbox-Nano ONNX TTS", lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str
    voice_reference: str | None = None
    max_new_tokens: int = MAX_NEW_TOKENS
    repetition_penalty: float = 1.2

def _bind(sess: ort.InferenceSession, **kwargs):
    """Bind tensors by name; raise if the graph doesn't expose the expected inputs."""
    expected = {i.name for i in sess.get_inputs()}
    missing = set(kwargs) - expected
    if missing:
        raise RuntimeError(
            f"ONNX graph inputs {expected} do not match provided keys {set(kwargs)}. "
            f"Model repo layout has likely changed."
        )
    return sess.run(None, kwargs)

def generate_speech(text: str, voice_reference: str, max_new_tokens: int, repetition_penalty: float) -> np.ndarray:
    """
    Full autoregressive pipeline. NOT YET IMPLEMENTED.

    The single-forward-pass placeholder that was here before produced garbage.
    A correct implementation must:
      1. embed text tokens  -> embed_tokens
      2. encode voice ref   -> speech_encoder
      3. loop the LM token-by-token with KV cache, sampling with repetition_penalty
      4. decode final token sequence -> conditional_decoder -> waveform

    Port the loop from the model repo's `run_onnx.py` (it is the reference
    implementation for these four graphs). The signatures logged at startup
    will tell you the exact input/output names to bind.
    """
    raise NotImplementedError(
        "Autoregressive LM loop not yet ported. See run_onnx.py in "
        f"{MODEL_REPO} for the reference implementation."
    )

@app.post("/generate")
async def generate(request: TTSRequest):
    ref = request.voice_reference or os.getenv("DEFAULT_VOICE")
    if not ref:
        raise HTTPException(400, "No voice_reference supplied and DEFAULT_VOICE is unset")
    if not os.path.exists(ref):
        raise HTTPException(400, f"voice_reference not found: {ref}")

    try:
        waveform = generate_speech(request.text, ref, request.max_new_tokens, request.repetition_penalty)
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

    buf = io.BytesIO()
    sf.write(buf, waveform, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav",
                             headers={"Content-Disposition": "attachment; filename=output.wav"})

@app.get("/health")
async def health():
    return {"status": "ok", "sessions": list(sessions.keys()), "tokenizer": tokenizer is not None}