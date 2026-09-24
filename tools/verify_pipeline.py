"""Offline checks for the ported generation math. Stubs ONNX/transformers so no model
download is needed. Run: python3 tools/verify_pipeline.py"""
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


class FakeMeta:
    def __init__(self, name, type_, shape):
        self.name, self.type, self.shape = name, type_, shape


class FakeSession:
    """Mimics an ORT session: exposes inputs/outputs and records feed shape history."""

    def __init__(self, inputs, outputs_fn):
        self._inputs = inputs
        self._outputs_fn = outputs_fn
        self.calls = []

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return []

    def run(self, _outputs, feed):
        self.calls.append({k: np.shape(v) for k, v in feed.items()})
        return self._outputs_fn(feed)


def load_app():
    """Import app.py with the heavy/unavailable deps stubbed out."""
    for name in ("onnxruntime", "soundfile", "huggingface_hub", "fastapi",
                 "fastapi.responses", "fastapi.concurrency", "pydantic",
                 "contextlib", "transformers"):
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        if name == "onnxruntime":
            class _GO:
                ORT_ENABLE_ALL = "all"
            mod.InferenceSession = object
            mod.SessionOptions = object
            mod.GraphOptimizationLevel = _GO
        elif name == "fastapi":
            class _App:
                def __init__(self, *a, **k):
                    pass
                def post(self, *a, **k):
                    return lambda fn: fn
                def get(self, *a, **k):
                    return lambda fn: fn
            mod.FastAPI = _App
            mod.HTTPException = type("HTTPException", (Exception,), {})
        elif name == "fastapi.responses":
            mod.StreamingResponse = object
        elif name == "fastapi.concurrency":
            async def run_in_threadpool(fn, *a):
                return fn(*a)
            mod.run_in_threadpool = run_in_threadpool
        elif name == "pydantic":
            class _BM:
                def __init__(self, **kw):
                    for k, v in kw.items():
                        setattr(self, k, v)
            mod.BaseModel = _BM
        elif name == "contextlib":
            def asynccontextmanager(fn):
                return fn
            mod.asynccontextmanager = asynccontextmanager
        elif name == "soundfile":
            def read(*a, **k):
                raise AssertionError("stub")
            mod.read = read
        elif name == "huggingface_hub":
            mod.snapshot_download = lambda *a, **k: None
        elif name == "transformers":
            class _Tok:
                @staticmethod
                def from_pretrained(*a, **k):
                    return None
            mod.GPT2TokenizerFast = _Tok
        sys.modules[name] = mod

    sys.path.insert(0, str(ROOT))
    import app  # noqa: E402
    return app


app = load_app()
failures = []


def check(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}{(' -> ' + detail) if detail else ''}")
    if not cond:
        failures.append(label)


# --- 1. _empty_cache: 24 KV inputs, correct dtypes and zero-length shape ----------
def kv_inputs(dtype="tensor(float16)"):
    ins = [FakeMeta("inputs_embeds", "tensor(float)", [1, None, 1024]),
           FakeMeta("attention_mask", "tensor(int64)", [1, None]),
           FakeMeta("position_ids", "tensor(int64)", [1, None])]
    for i in range(12):
        ins.append(FakeMeta(f"past_key_values.{i}.key", dtype, [1, 12, 0, 64]))
        ins.append(FakeMeta(f"past_key_values.{i}.value", "tensor(float)", [1, 12, 0, 64]))
    return ins


cache = app._empty_cache(FakeSession(kv_inputs(), None), batch_size=1)
check("_empty_cache builds 24 entries", len(cache) == 24, str(len(cache)))
check("_empty_cache zero-length seq dim", all(v.shape[2] == 0 for v in cache.values()))
check("_empty_cache fp16 key dtype", cache["past_key_values.0.key"].dtype == np.float16)
check("_empty_cache fp32 value dtype", cache["past_key_values.0.value"].dtype == np.float32)

# --- 2. Cache update offset: outputs[1:] maps to cache inputs, grows per step -----
# This is the subtle part: outputs are [logits, k0, v0, k1, v1, ...]; dict iteration
# is insertion-ordered, so cache[names[i]] must be outputs[i+1] or every layer after
# the first silently receives the wrong tensor.
order = list(cache.keys())
out = [np.zeros((1, 5, 100), dtype=np.float32)]
for i, name in enumerate(order):
    layer = int(name.split(".")[1])
    shape = (1, 12, 1, 64)
    dt = cache[name].dtype
    out.append(np.full(shape, layer, dtype=dt))
rebuilt = {name: out[i + 1] for i, name in enumerate(order)}
check("cache output offset +1 is correct",
      all(rebuilt[n][0, 0, 0, 0] == int(n.split(".")[1]) for n in order))
check("wrong offset would corrupt layer 0+",
      out[1][0, 0, 0, 0] == 0 and out[3][0, 0, 0, 0] == 1)

# --- 3. _repetition_penalty ------------------------------------------------------
logits = np.array([[1.0, -1.0, 2.0, -2.0]], dtype=np.float32)
gen = np.array([[2, 2, 3]], dtype=np.int64)  # token 2 seen, token 1 never generated
penalized = app._repetition_penalty(logits, gen, 1.2)
check("positive logit divided by penalty", np.isclose(penalized[0, 2], 2.0 / 1.2))
check("unseen token untouched", np.isclose(penalized[0, 1], -1.0))
check("penalty 1.0 is identity", np.array_equal(app._repetition_penalty(logits, gen, 1.0), logits))
check("input logits not mutated in place", np.isclose(logits[0, 2], 2.0))

# --- 4. _sample_token ------------------------------------------------------------
rng = np.random.default_rng(1337)
greedy = app._sample_token(logits, rng, temperature=0.0, top_k=0, top_p=0.0)
check("greedy picks argmax", int(greedy[0, 0]) == 2, str(int(greedy[0, 0])))

onehot = np.array([[0.0, 0.0, 10.0, 0.0]], dtype=np.float32)
s = app._sample_token(onehot, rng, 0.8, 0, 0.95)
check("near-one-hot samples the peak", int(s[0, 0]) == 2, str(int(s[0, 0])))

k1 = app._sample_token(logits, rng, 0.8, top_k=1, top_p=0.0)
check("top_k=1 forces argmax", int(k1[0, 0]) == 2, str(int(k1[0, 0])))

inrange = True
for _ in range(50):
    tok = int(app._sample_token(logits, rng, 1.0, 0, 0.95)[0, 0])
    inrange &= 0 <= tok < logits.shape[-1]
check("sampled ids stay in vocab range", inrange)

# --- 5. End-to-end loop with stubbed graphs -------------------------------------
VOCAB, HID, PROMPT = 7000, 8, 3  # vocab must exceed STOP_SPEECH_TOKEN (6562)
lm_inputs = kv_inputs()
state = {"step": 0}


def lm_outputs(feed):
    state["step"] += 1
    past = feed["past_key_values.0.key"].shape[2]
    seq = feed["inputs_embeds"].shape[1]
    if past == 0:
        # fake tokenizer yields 3 text tokens, prefixed by PROMPT conditioning tokens
        assert seq == PROMPT + 3, f"first pass should embed prompt+text, got {seq}"
    else:
        assert seq == 1, f"decode steps must feed 1 token, got {seq}"
    assert feed["attention_mask"].shape[1] == past + seq, "mask must cover cache+new"
    assert feed["position_ids"].shape[1] == seq, "one position id per new token"
    out = [np.zeros((1, seq, VOCAB), dtype=np.float32)]
    for i, name in enumerate(lm_inputs):
        if name.name.startswith("past_key_values."):
            out.append(np.zeros((1, 12, past + seq, 64),
                                dtype=np.float16 if name.type == "tensor(float16)" else np.float32))
    if state["step"] >= 4:
        out[0][:, -1, app.STOP_SPEECH_TOKEN] = 50.0  # force STOP after 3 generated tokens
    else:
        out[0][:, -1, 2] = 50.0
    return out


class Tok:
    def __call__(self, text, return_tensors=None):
        return {"input_ids": np.array([[1, 4, 5]], dtype=np.int64)}


class Req:
    text = "hi"
    voice_reference = "ref.wav"
    max_new_tokens = 8
    repetition_penalty = 1.2
    temperature = 0.0
    top_k = 0
    top_p = 0.0
    seed = 1337


lm = FakeSession(lm_inputs, lm_outputs)
app.sessions["language_model"] = lm
app.tokenizer = Tok()
app._bind = lambda sess, **kw: sess.run(None, kw)


def enc_outputs(feed):
    # audio_features, audio_tokens, speaker_embeddings, speaker_features
    return [np.zeros((1, PROMPT, HID), dtype=np.float32),
            np.full((1, PROMPT), 7, dtype=np.int64),
            np.zeros((1, 16), dtype=np.float32),
            np.zeros((1, 5, 16), dtype=np.float32)]


app.sessions["speech_encoder"] = FakeSession([], enc_outputs)
app.sessions["embed_tokens"] = FakeSession(
    [], lambda feed: [np.zeros((1, feed["input_ids"].shape[1], HID), dtype=np.float32)])

captured = {}


def dec_outputs(feed):
    captured["speech_tokens"] = feed["speech_tokens"]
    captured["keys"] = sorted(feed)
    return [np.zeros((1, 2400), dtype=np.float32)]


app.sessions["conditional_decoder"] = FakeSession([], dec_outputs)
app.generate_speech = app.generate_speech  # keep real impl
app.sf = types.SimpleNamespace(
    read=lambda *a, **k: (np.zeros(4800, dtype=np.float32), 24000),
    write=lambda *a, **k: None,
)

wave = app.generate_speech(Req())
check("loop ran and produced a waveform", wave.shape == (2400,), str(wave.shape))
check("decoder received speaker_embeddings", "speaker_embeddings" in captured.get("keys", []))
check("decoder received speaker_features", "speaker_features" in captured.get("keys", []))
# START + 3 generated + 3 silence padding, EOS stripped
check("speech_tokens = prompt + generated + 3 silence",
      captured["speech_tokens"].shape[1] == 3 + 3 + 3, str(captured["speech_tokens"].shape))
check("stopped early on STOP token", state["step"] == 4, f"ran {state['step']} passes")
check("padding is SILENCE_TOKEN",
      bool(np.all(captured["speech_tokens"][0, -3:] == app.SILENCE_TOKEN)))

print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED: {failures}"))
sys.exit(1 if failures else 0)
