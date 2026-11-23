"""Hugging Face lookups for the "add model" dialog: list a repo's GGUF files, read a file's header
(a ranged request) for context length and KV-cache cost, and build a registry entry from them."""
import json
import re
import struct
import urllib.error
import urllib.parse
import urllib.request

HF = "https://huggingface.co"
HEADER_BYTES = 1 << 20  # arch keys come before the (huge) tokenizer arrays, so the first MiB is plenty
FALLBACK_CTX = 32768
SAMPLING = {"temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05}

_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
_SHARD_RE = re.compile(r"-\d{5}-of-\d{5}")
_QUANT_RE = re.compile(r"[-_.]((?:I?Q|PQ|PTQ|TQ)\d\w*|BF16|F16|F32)$", re.I)


class HFError(Exception):
    pass


class HTTPStatus(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


def parse_repo_id(text: str) -> str:
    """'owner/name' or any huggingface.co URL for it -> 'owner/name'."""
    t = re.sub(r"^https?://", "", text.strip())
    is_url = t.startswith(("huggingface.co/", "www.huggingface.co/"))
    if is_url:
        t = t.split("/", 1)[1]
        t = "/".join(t.split("/")[:2])
    if not _REPO_RE.match(t):
        raise HFError("Enter a Hugging Face model id like owner/name (or paste its URL)")
    return t


def http_get(url: str, headers: dict | None = None, limit: int | None = None) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "openllm", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read(limit) if limit else resp.read()
    except urllib.error.HTTPError as e:
        raise HTTPStatus(e.code) from e


# ----------------------------------------------------------------- GGUF header
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class _Short(Exception):
    """The buffer ended mid-value."""


class _Reader:
    def __init__(self, buf: bytes) -> None:
        self.buf, self.pos = buf, 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.buf):
            raise _Short
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def unpack(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        return self.take(self.unpack("<Q")).decode("utf-8", "replace")

    def value(self, vtype: int):
        if vtype in _SCALAR:
            return self.unpack(_SCALAR[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            etype, n = self.unpack("<I"), self.unpack("<Q")
            return [self.value(etype) for _ in range(n)]
        raise HFError(f"unsupported GGUF value type {vtype}")


def parse_gguf_meta(buf: bytes) -> dict:
    """Key/value metadata from the start of a GGUF file. Stops at the first tokenizer.* key and
    returns whatever was read if the buffer ends early."""
    if buf[:4] != b"GGUF":
        raise HFError("not a GGUF file")
    r = _Reader(buf)
    r.pos = 4
    meta: dict = {}
    try:
        if r.unpack("<I") < 2:
            raise HFError("GGUF version 1 is not supported")
        r.unpack("<Q")  # tensor count
        for _ in range(r.unpack("<Q")):
            key = r.string()
            if key.startswith("tokenizer."):
                break
            meta[key] = r.value(r.unpack("<I"))
    except _Short:
        pass
    return meta


def kv_profile(meta: dict) -> dict | None:
    """{'max_ctx', 'f16'} where f16 is MiB of KV cache per 1K tokens at f16; None if the keys are missing."""
    arch = meta.get("general.architecture")
    if not arch:
        return None

    def get(key):
        return meta.get(f"{arch}.{key}")

    layers, heads, emb = get("block_count"), get("attention.head_count"), get("embedding_length")
    if not (isinstance(layers, int) and isinstance(emb, int) and heads):
        return None
    kv = get("attention.head_count_kv")
    kv = heads if kv is None else kv
    every = get("full_attention_interval")  # hybrid models: only every Nth layer keeps a KV cache
    attn_layers = layers // every if isinstance(every, int) and every > 0 else layers
    kv_total = sum(kv) if isinstance(kv, list) else kv * attn_layers
    head_dim = emb // (max(heads) if isinstance(heads, list) else heads)
    key_len = get("attention.key_length") or head_dim
    val_len = get("attention.value_length") or key_len
    ctx = get("context_length")
    prof = {"max_ctx": ctx if isinstance(ctx, int) and ctx > 0 else FALLBACK_CTX,
            "f16": max(1, round((key_len + val_len) * kv_total * 2 / 1024))}
    sampling = {k: round(float(v), 3) if k != "top_k" else int(v)
                for k in ("temp", "top_p", "top_k", "min_p")
                if isinstance(v := meta.get(f"general.sampling.{k}"), (int, float)) and not isinstance(v, bool)}
    if sampling:
        prof["sampling"] = sampling
    return prof


# ----------------------------------------------------------------- registry entry
def _base_name(repo: str) -> str:
    return re.sub(r"[-_.]?gguf$", "", repo.split("/", 1)[1], flags=re.I)


def _quant(file: str) -> str | None:
    m = _QUANT_RE.search(file.removesuffix(".gguf"))
    return m.group(1) if m else None


def quant_of(file: str) -> str | None:
    """'Model-Q4_K_M.gguf' -> 'Q4_K_M' (None if the name carries no quant tag)."""
    return _quant(file)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def make_model_id(repo: str, file: str, taken) -> str:
    quant = _quant(file)
    base = _slug(_base_name(repo) + (f"-{quant}" if quant else "")) or "model"
    mid, n = base, 2
    while mid in taken:
        mid, n = f"{base}-{n}", n + 1
    return mid


def build_entry(repo: str, file: str, size_bytes: int, profile: dict | None, runtime: str) -> dict:
    gib = size_bytes / 2**30
    if profile:
        max_ctx, f16 = profile["max_ctx"], profile["f16"]
        note = "Added from Hugging Face. Assumes a non-thinking model with generic sampling; edit user_models.json to change."
    else:
        max_ctx, f16 = FALLBACK_CTX, max(32, round(gib * 16))
        note = ("Added from Hugging Face. Context and KV-cache memory are estimated (the file's metadata "
                "could not be read), so the memory gauge is conservative. Assumes a non-thinking model.")
    quant = _quant(file)
    return {
        "name": _base_name(repo) + (f" ({quant})" if quant else ""),
        "runtime": runtime,
        "repo": repo,
        "file": file,
        "size_gb": round(size_bytes / 1e9, 2),
        "max_ctx": max_ctx,
        "sampling": {**SAMPLING, **((profile or {}).get("sampling") or {})},
        "reasoning_efforts": ["off"],
        "weights_gib": round(gib, 2),
        "kv_mib_per_1k": {"f16": f16, "q8_0": round(f16 * 8.5 / 16), "q4_0": round(f16 * 4.5 / 16)},
        "notes": note,
    }


# ----------------------------------------------------------------- client
class HFClient:
    """Network calls behind one seam: pass a fake `get(url, headers=None, limit=None) -> bytes` in tests."""

    def __init__(self, get=http_get) -> None:
        self._get = get

    def search(self, query: str, limit: int = 15) -> list[dict]:
        """Public GGUF repos matching a free-text query, most downloaded first."""
        qs = urllib.parse.urlencode({"search": query, "filter": "gguf", "sort": "downloads",
                                     "direction": "-1", "limit": limit})
        try:
            data = json.loads(self._get(f"{HF}/api/models?{qs}"))
        except HTTPStatus as e:
            raise HFError(f"Hugging Face returned {e}") from e
        except (OSError, ValueError) as e:
            raise HFError(f"could not reach Hugging Face: {e}") from e
        out = [{"repo": d["id"], "downloads": int(d.get("downloads") or 0)}
               for d in data if isinstance(d, dict) and _REPO_RE.match(d.get("id", ""))]
        if not out:
            raise HFError(f"no GGUF models found for {query!r}")
        return out

    def list_files(self, repo: str) -> list[dict]:
        try:
            data = json.loads(self._get(f"{HF}/api/models/{repo}?blobs=true"))
        except HTTPStatus as e:
            if e.code in (401, 403, 404):  # HF answers 401 for repos that don't exist
                raise HFError(f"{repo} not found, or it is private or gated (only public repos are supported)") from e
            raise HFError(f"Hugging Face returned {e}") from e
        except (OSError, ValueError) as e:
            raise HFError(f"could not reach Hugging Face: {e}") from e
        files = [{"file": s["rfilename"], "size": int(s.get("size") or 0)}
                 for s in data.get("siblings", [])
                 if s["rfilename"].endswith(".gguf") and "/" not in s["rfilename"]
                 and "mmproj" not in s["rfilename"].lower() and not _SHARD_RE.search(s["rfilename"])]
        if not files:
            raise HFError(f"no usable GGUF files at the top level of {repo} (split and sub-folder files are skipped)")
        return sorted(files, key=lambda f: f["size"])

    def read_profile(self, repo: str, file: str) -> dict | None:
        try:
            buf = self._get(f"{HF}/{repo}/resolve/main/{file}",
                            headers={"Range": f"bytes=0-{HEADER_BYTES - 1}"}, limit=HEADER_BYTES)
            return kv_profile(parse_gguf_meta(buf))
        except (HFError, HTTPStatus, OSError):
            return None
