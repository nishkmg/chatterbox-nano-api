import os
import io
import threading
import numpy as np
import soundfile as sf
import onnxruntime as ort
from huggingface_hub import snapshot_download
from transformers import GPT2TokenizerFast
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from contextlib import asynccontextmanager

MODEL_REPO = os.getenv("MODEL_REPO", "owensong/chatterbox-nano-ONNX")
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")   # pin a commit hash for reproducibility
# Default matches the container path; on bare metal export MODEL_DIR to a writable dir.
MODEL_DIR = os.path.abspath(os.getenv("MODEL_DIR", "/app/model"))
# Keep the HF cache on the same volume as MODEL_DIR, else the tokenizer re-downloads on
# every cold start even when the ONNX graphs are already persisted.
os.environ.setdefault("HF_HOME", os.path.join(MODEL_DIR, ".hf"))

SAMPLE_RATE = 24000
START_SPEECH_TOKEN = 6561
STOP_SPEECH_TOKEN = 6562
SILENCE_TOKEN = 4299
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
# ONNX Runtime sessions are not safe for concurrent run() on shared state; serialize
# generation so concurrent requests can't corrupt each other's KV cache.
generation_lock = threading.Lock()

# Only fetch what we need. .onnx_data sidecars ARE required for quantized graphs.
# run_onnx.py is the reference implementation for the autoregressive loop.
ALLOW_PATTERNS = [
    "onnx/*.onnx",
    "onnx/*.onnx_data",
    "tokenizer*",
    "vocab*",
    "merges*",
    "special_tokens*",
    "*.json",
    "run_onnx.py",
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
    # ORT_ENABLE_ALL runs the NCHWc layout transform. On CPU, it rewrites this
    # model's FP16 AveragePool-19 nodes to com.ms.internal.nhwc, which has no
    # compatible CPU kernel and prevents session initialization.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    print(f"ONNX Runtime {ort.__version__}; graph optimization: extended")

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
    # Load from MODEL_DIR, not the hub: the snapshot above already fetched these files.
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR, local_files_only=True)

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
    temperature: float = 0.8
    top_k: int = 1000
    top_p: float = 0.95
    seed: int | None = None

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

def _repetition_penalty(logits: np.ndarray, generated: np.ndarray, penalty: float) -> np.ndarray:
    if penalty == 1.0:
        return logits
    result = logits.copy()
    token_ids = np.unique(generated)
    values = result[:, token_ids]
    result[:, token_ids] = np.where(values < 0, values * penalty, values / penalty)
    return result

def _sample_token(logits, rng, temperature, top_k, top_p) -> np.ndarray:
    if temperature <= 0:
        return np.argmax(logits, axis=-1, keepdims=True).astype(np.int64)

    scores = logits.astype(np.float64) / temperature
    if 0 < top_k < scores.shape[-1]:
        cutoff = np.partition(scores, -top_k, axis=-1)[:, -top_k][:, None]
        scores = np.where(scores < cutoff, -np.inf, scores)

    order = np.argsort(scores, axis=-1)[:, ::-1]
    ordered = np.take_along_axis(scores, order, axis=-1)
    ordered -= np.max(ordered, axis=-1, keepdims=True)
    probs = np.exp(ordered)
    probs /= probs.sum(axis=-1, keepdims=True)

    if 0 < top_p < 1:
        cumulative = np.cumsum(probs, axis=-1)
        probs = np.where(cumulative - probs >= top_p, 0.0, probs)
        probs /= probs.sum(axis=-1, keepdims=True)

    sampled = [rng.choice(order.shape[1], p=probs[row]) for row in range(order.shape[0])]
    return np.take_along_axis(order, np.asarray(sampled)[:, None], axis=-1).astype(np.int64)

def _empty_cache(sess: ort.InferenceSession, batch_size: int) -> dict[str, np.ndarray]:
    """Zero-length KV cache: 12 layers x (K, V) = 24 inputs, named past_key_values.*"""
    cache: dict[str, np.ndarray] = {}
    for value in sess.get_inputs():
        if not value.name.startswith("past_key_values."):
            continue
        dtype = np.float16 if value.type == "tensor(float16)" else np.float32
        heads = value.shape[1] if isinstance(value.shape[1], int) else 12
        head_dim = value.shape[3] if isinstance(value.shape[3], int) else 64
        cache[value.name] = np.zeros((batch_size, heads, 0, head_dim), dtype=dtype)
    if len(cache) != 24:
        raise RuntimeError(f"expected 24 KV-cache inputs, found {len(cache)}")
    return cache

def generate_speech(request: TTSRequest) -> np.ndarray:
    """Ported from run_onnx.py in owensong/chatterbox-nano-ONNX."""
    with generation_lock:
        rng = np.random.default_rng(request.seed)

        # 1. Reference audio must be 24 kHz; the encoder has no resampling stage.
        audio, sr = sf.read(request.voice_reference, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            raise ValueError(f"voice reference must be {SAMPLE_RATE} Hz, got {sr}")
        audio_values = audio[np.newaxis, :].astype(np.float32)

        # 2. Encode the reference into conditioning + prompt speech tokens.
        audio_features, audio_tokens, speaker_embeddings, speaker_features = _bind(
            sessions["speech_encoder"], audio_values=audio_values
        )

        # 3. Autoregressive LM loop with KV cache.
        input_ids = tokenizer(request.text, return_tensors="np")["input_ids"].astype(np.int64)
        generated = np.full((input_ids.shape[0], 1), START_SPEECH_TOKEN, dtype=np.int64)

        lm = sessions["language_model"]
        cache = None
        attention_mask = None
        position_ids = None
        reached_eos = False

        for step in range(request.max_new_tokens):
            embeds = _bind(sessions["embed_tokens"], input_ids=input_ids)[0]
            if step == 0:
                # Prefix the prompt conditioning so the model attends to the reference.
                embeds = np.concatenate((audio_features, embeds), axis=1)
                batch_size, sequence_length, _ = embeds.shape
                cache = _empty_cache(lm, batch_size)
                attention_mask = np.ones((batch_size, sequence_length), dtype=np.int64)
                position_ids = np.arange(sequence_length, dtype=np.int64)[None, :].repeat(batch_size, axis=0)

            outputs = _bind(
                lm,
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **cache,
            )
            logits = _repetition_penalty(outputs[0][:, -1, :], generated, request.repetition_penalty)
            input_ids = _sample_token(logits, rng, request.temperature, request.top_k, request.top_p)
            generated = np.concatenate((generated, input_ids), axis=-1)

            if np.all(input_ids == STOP_SPEECH_TOKEN):
                reached_eos = True
                break

            attention_mask = np.concatenate(
                (attention_mask, np.ones((attention_mask.shape[0], 1), dtype=np.int64)), axis=1
            )
            position_ids = position_ids[:, -1:] + 1
            for index, name in enumerate(cache):
                cache[name] = outputs[index + 1]

        # 4. Decode prompt + generated tokens to a waveform.
        generated_audio = generated[:, 1:-1] if reached_eos else generated[:, 1:]
        silence = np.full((generated_audio.shape[0], 3), SILENCE_TOKEN, dtype=np.int64)
        speech_tokens = np.concatenate((audio_tokens, generated_audio, silence), axis=1)
        waveform = _bind(
            sessions["conditional_decoder"],
            speech_tokens=speech_tokens,
            speaker_embeddings=speaker_embeddings,
            speaker_features=speaker_features,
        )[0]

    return waveform.squeeze().astype(np.float32)

@app.post("/generate")
async def generate(request: TTSRequest):
    ref = request.voice_reference or os.getenv("DEFAULT_VOICE")
    if not ref:
        raise HTTPException(400, "No voice_reference supplied and DEFAULT_VOICE is unset")
    if not os.path.exists(ref):
        raise HTTPException(400, f"voice_reference not found: {ref}")

    request.voice_reference = ref
    try:
        # CPU inference is blocking; run it off the event loop.
        waveform = await run_in_threadpool(generate_speech, request)
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

