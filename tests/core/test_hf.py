import json
import struct

import pytest

from openllm_app.core import hf


# ----------------------------------------------------------------- GGUF header builder
def _s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, vtype: int, payload: bytes) -> bytes:
    return _s(key) + struct.pack("<I", vtype) + payload


def u32(key, v):
    return _kv(key, 4, struct.pack("<I", v))


def string(key, v):
    return _kv(key, 8, _s(v))


def u32_array(key, vals):
    return _kv(key, 9, struct.pack("<IQ", 4, len(vals)) + b"".join(struct.pack("<I", v) for v in vals))


def gguf(*kvs: bytes, version=3) -> bytes:
    return b"GGUF" + struct.pack("<IQQ", version, 0, len(kvs)) + b"".join(kvs)


LLAMA8B = [
    string("general.architecture", "llama"),
    u32("llama.block_count", 32),
    u32("llama.context_length", 131072),
    u32("llama.embedding_length", 4096),
    u32("llama.attention.head_count", 32),
    u32("llama.attention.head_count_kv", 8),
]


# ----------------------------------------------------------------- repo ids
@pytest.mark.parametrize("text", [
    "owner/name",
    "  owner/name  ",
    "https://huggingface.co/owner/name",
    "huggingface.co/owner/name/tree/main",
    "https://huggingface.co/owner/name/blob/main/file.gguf",
])
def test_parse_repo_id_accepts_ids_and_urls(text):
    assert hf.parse_repo_id(text) == "owner/name"


@pytest.mark.parametrize("text", ["", "justaname", "a/b/c", "../etc/passwd", "ow ner/name", "owner/"])
def test_parse_repo_id_rejects_junk(text):
    with pytest.raises(hf.HFError):
        hf.parse_repo_id(text)


# ----------------------------------------------------------------- header parsing
def test_parse_gguf_meta_reads_scalars_and_strings():
    meta = hf.parse_gguf_meta(gguf(*LLAMA8B))
    assert meta["general.architecture"] == "llama"
    assert meta["llama.block_count"] == 32


def test_parse_gguf_meta_stops_at_tokenizer_keys():
    # a truncated tokenizer array after the arch keys must not matter
    buf = gguf(*LLAMA8B, string("tokenizer.ggml.model", "gpt2"))
    meta = hf.parse_gguf_meta(buf[:-3])
    assert meta["llama.attention.head_count_kv"] == 8


def test_parse_gguf_meta_returns_what_it_got_when_buffer_is_cut():
    buf = gguf(*LLAMA8B)
    meta = hf.parse_gguf_meta(buf[:-5])  # cuts into the last key
    assert meta["llama.block_count"] == 32
    assert "llama.attention.head_count_kv" not in meta


def test_parse_gguf_meta_rejects_non_gguf():
    with pytest.raises(hf.HFError):
        hf.parse_gguf_meta(b"<html>not a model</html>")


def test_parse_gguf_meta_rejects_old_version():
    with pytest.raises(hf.HFError):
        hf.parse_gguf_meta(gguf(*LLAMA8B, version=1))


# ----------------------------------------------------------------- KV profile
def test_kv_profile_dense_llama_matches_known_figure():
    # Llama 3.1 8B: 32 layers x 8 kv heads x 128 dim x (K+V) x 2 bytes = 128 MiB per 1K tokens at f16
    prof = hf.kv_profile(hf.parse_gguf_meta(gguf(*LLAMA8B)))
    assert prof == {"max_ctx": 131072, "f16": 128}


def test_kv_profile_per_layer_kv_heads_array_counts_only_attention_layers():
    meta = hf.parse_gguf_meta(gguf(
        string("general.architecture", "qwen35"),
        u32("qwen35.block_count", 4),
        u32("qwen35.context_length", 262144),
        u32("qwen35.embedding_length", 4096),
        u32("qwen35.attention.head_count", 32),
        u32_array("qwen35.attention.head_count_kv", [0, 0, 0, 4]),  # one attention layer
        u32("qwen35.attention.key_length", 256),
    ))
    # (256 + 256) * 4 kv heads * 2 bytes = 4096 B/token = 4 MiB per 1K
    assert hf.kv_profile(meta) == {"max_ctx": 262144, "f16": 4}


def test_kv_profile_hybrid_model_only_counts_every_nth_layer():
    # Bonsai 2 27B (qwen35): 64 layers, full attention every 4th, 4 kv heads, 256 dim -> 64 MiB per 1K
    meta = hf.parse_gguf_meta(gguf(
        string("general.architecture", "qwen35"),
        u32("qwen35.block_count", 64),
        u32("qwen35.context_length", 262144),
        u32("qwen35.embedding_length", 5120),
        u32("qwen35.attention.head_count", 24),
        u32("qwen35.attention.head_count_kv", 4),
        u32("qwen35.attention.key_length", 256),
        u32("qwen35.attention.value_length", 256),
        u32("qwen35.full_attention_interval", 4),
    ))
    assert hf.kv_profile(meta) == {"max_ctx": 262144, "f16": 64}


def test_kv_profile_carries_the_models_recommended_sampling():
    f32 = lambda key, v: _kv(key, 6, struct.pack("<f", v))
    meta = hf.parse_gguf_meta(gguf(*LLAMA8B, f32("general.sampling.temp", 1.0), f32("general.sampling.top_p", 0.95),
                                   u32("general.sampling.top_k", 20)))
    assert hf.kv_profile(meta)["sampling"] == {"temp": 1.0, "top_p": 0.95, "top_k": 20}


def test_build_entry_prefers_the_models_own_sampling():
    prof = {"max_ctx": 8192, "f16": 64, "sampling": {"temp": 1.0, "top_p": 0.95, "top_k": 20}}
    e = hf.build_entry("o/m-GGUF", "m-Q4_K_M.gguf", 4_000_000_000, prof, "prismml")
    assert e["sampling"] == {"temp": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05}


def test_kv_profile_none_without_arch_keys():
    assert hf.kv_profile({}) is None
    assert hf.kv_profile({"general.architecture": "llama"}) is None


# ----------------------------------------------------------------- entry building
def test_build_entry_with_header_info():
    e = hf.build_entry("RavichandranJ/Dolphin3-Cyber-8B-GGUF", "Dolphin3.0-Llama3.1-8B-abliterated.Q6_K.gguf",
                       6596020288, hf.kv_profile(hf.parse_gguf_meta(gguf(*LLAMA8B))), "prismml")
    assert e["name"] == "Dolphin3-Cyber-8B (Q6_K)"
    assert (e["runtime"], e["repo"]) == ("prismml", "RavichandranJ/Dolphin3-Cyber-8B-GGUF")
    assert e["file"] == "Dolphin3.0-Llama3.1-8B-abliterated.Q6_K.gguf"
    assert e["size_gb"] == 6.6 and e["weights_gib"] == 6.14
    assert e["max_ctx"] == 131072
    assert e["kv_mib_per_1k"] == {"f16": 128, "q8_0": 68, "q4_0": 36}
    assert e["reasoning_efforts"] == ["off"]
    assert e["sampling"] == {"temp": 0.7, "top_p": 0.9, "top_k": 40, "min_p": 0.05}
    assert "header" not in e["notes"].lower()


def test_build_entry_falls_back_conservatively_and_says_so():
    e = hf.build_entry("o/m-GGUF", "m-Q4_K_M.gguf", 4_000_000_000, None, "prismml")
    assert e["max_ctx"] == 32768
    assert e["kv_mib_per_1k"]["f16"] >= 32
    assert e["kv_mib_per_1k"]["q8_0"] < e["kv_mib_per_1k"]["f16"]
    assert "estimated" in e["notes"].lower()


def test_make_model_id_slugs_and_dedupes():
    assert hf.make_model_id("o/My_Model-8B-GGUF", "My_Model-8B.Q6_K.gguf", set()) == "my-model-8b-q6-k"
    assert hf.make_model_id("o/My_Model-8B-GGUF", "My_Model-8B.Q6_K.gguf", {"my-model-8b-q6-k"}) == "my-model-8b-q6-k-2"


# ----------------------------------------------------------------- client (network faked)
def _api(siblings):
    return json.dumps({"siblings": [{"rfilename": n, "size": s} for n, s in siblings]}).encode()


def test_list_files_keeps_usable_ggufs_smallest_first():
    calls = []

    def fake_get(url, headers=None, limit=None):
        calls.append(url)
        return _api([
            ("README.md", 10), ("mmproj-BF16.gguf", 900), ("sub/dir/x.gguf", 5),
            ("m-00001-of-00002.gguf", 7), ("m-Q8_0.gguf", 8500), ("m-Q6_K.gguf", 6500),
        ])

    files = hf.HFClient(fake_get).list_files("o/m")
    assert files == [{"file": "m-Q6_K.gguf", "size": 6500}, {"file": "m-Q8_0.gguf", "size": 8500}]
    assert calls == ["https://huggingface.co/api/models/o/m?blobs=true"]


def test_list_files_explains_when_nothing_usable():
    client = hf.HFClient(lambda *a, **k: _api([("README.md", 1)]))
    with pytest.raises(hf.HFError, match="GGUF"):
        client.list_files("o/m")


def test_http_errors_become_readable_messages():
    def gated(url, headers=None, limit=None):
        raise hf.HTTPStatus(401)

    def missing(url, headers=None, limit=None):
        raise hf.HTTPStatus(404)

    # Hugging Face answers 401 for repos that don't exist, so the message has to cover both
    with pytest.raises(hf.HFError, match="not found.*private or gated"):
        hf.HFClient(gated).list_files("o/m")
    with pytest.raises(hf.HFError, match="not found"):
        hf.HFClient(missing).list_files("o/m")


def test_read_profile_uses_a_range_request_and_parses():
    seen = {}

    def fake_get(url, headers=None, limit=None):
        seen.update(url=url, headers=headers, limit=limit)
        return gguf(*LLAMA8B)

    prof = hf.HFClient(fake_get).read_profile("o/m", "m-Q6_K.gguf")
    assert prof == {"max_ctx": 131072, "f16": 128}
    assert seen["url"] == "https://huggingface.co/o/m/resolve/main/m-Q6_K.gguf"
    assert seen["headers"]["Range"].startswith("bytes=0-")


def test_read_profile_returns_none_when_header_is_unusable():
    assert hf.HFClient(lambda *a, **k: b"<html>").read_profile("o/m", "f.gguf") is None

    def boom(*a, **k):
        raise OSError("offline")

    assert hf.HFClient(boom).read_profile("o/m", "f.gguf") is None


def test_search_returns_repos_by_downloads_and_ignores_junk():
    urls = []

    def fake_get(url, headers=None, limit=None):
        urls.append(url)
        return json.dumps([{"id": "a/b-GGUF", "downloads": 9}, {"id": "bad id"}, {"id": "c/d", "downloads": None}]).encode()

    out = hf.HFClient(fake_get).search("qwen 3")
    assert out == [{"repo": "a/b-GGUF", "downloads": 9}, {"repo": "c/d", "downloads": 0}]
    assert "search=qwen+3" in urls[0] and "filter=gguf" in urls[0]


def test_search_with_no_results_explains():
    with pytest.raises(hf.HFError, match="no GGUF models"):
        hf.HFClient(lambda *a, **k: b"[]").search("zzz")
