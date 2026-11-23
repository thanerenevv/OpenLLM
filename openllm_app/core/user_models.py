"""user_models.json: models added from the UI, kept apart from the curated models.json."""
import json
import re
from pathlib import Path

FILENAME = "user_models.json"

_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")
_REQUIRED = ("name", "runtime", "repo", "file", "max_ctx", "sampling", "reasoning_efforts",
             "weights_gib", "kv_mib_per_1k")


def _valid(m) -> bool:
    """Reject entries that would crash the app or write outside models/ (the file is hand-editable)."""
    if not isinstance(m, dict) or any(k not in m for k in _REQUIRED):
        return False
    f, efforts, kv = m["file"], m["reasoning_efforts"], m["kv_mib_per_1k"]
    runtimes = m.get("runtimes", [])
    return (isinstance(runtimes, list) and all(isinstance(r, str) for r in runtimes)
            and isinstance(f, str) and f.endswith(".gguf") and "/" not in f and not f.startswith(".")
            and isinstance(m["repo"], str) and bool(_REPO_RE.match(m["repo"]))
            and isinstance(efforts, list) and bool(efforts)
            and isinstance(kv, dict) and all(k in kv for k in ("f16", "q8_0", "q4_0"))
            and isinstance(m["sampling"], dict) and isinstance(m["max_ctx"], int))


def load(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {mid: m for mid, m in data.items() if _valid(m)}


def add(path: Path, model_id: str, model: dict) -> None:
    models = load(path)
    models[model_id] = model
    path.write_text(json.dumps(models, indent=2) + "\n")


# ------------------------------------------------------------------ per-model edits
OVERRIDES_FILENAME = "model_overrides.json"
SAMPLING_KEYS = ("temp", "top_p", "top_k", "min_p")


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def clean_overrides(fields) -> dict:
    """Keep only well-formed editable fields (the file is hand-editable, the dialog is not trusted either)."""
    if not isinstance(fields, dict):
        return {}
    out: dict = {}
    if isinstance(fields.get("name"), str) and fields["name"].strip():
        out["name"] = fields["name"].strip()
    if isinstance(fields.get("repo"), str) and _REPO_RE.match(fields["repo"]):
        out["repo"] = fields["repo"]
    f = fields.get("file")
    if isinstance(f, str) and f.endswith(".gguf") and "/" not in f and not f.startswith("."):
        out["file"] = f
    mc = fields.get("max_ctx")
    if isinstance(mc, int) and not isinstance(mc, bool) and mc >= 1024:
        out["max_ctx"] = mc
    smp = fields.get("sampling")
    if isinstance(smp, dict):
        good = {k: smp[k] for k in SAMPLING_KEYS if k in smp and _num(smp[k])}
        if good:
            out["sampling"] = good
    eff = fields.get("reasoning_efforts")
    if isinstance(eff, list) and eff and all(isinstance(e, str) and e.strip() for e in eff):
        out["reasoning_efforts"] = [e.strip() for e in eff]
    return out


def load_overrides(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {mid: c for mid, f in data.items() if (c := clean_overrides(f))}


def set_override(path: Path, model_id: str, fields: dict) -> None:
    data = load_overrides(path)
    data[model_id] = clean_overrides(fields)
    path.write_text(json.dumps(data, indent=2) + "\n")


def clear_override(path: Path, model_id: str) -> None:
    data = load_overrides(path)
    if data.pop(model_id, None) is not None:
        path.write_text(json.dumps(data, indent=2) + "\n")


def apply_overrides(models: dict, overrides: dict) -> None:
    for mid, o in overrides.items():
        if mid in models:
            m = models[mid]
            m.update({k: v for k, v in o.items() if k != "sampling"})
            if "sampling" in o:
                m["sampling"] = {**m.get("sampling", {}), **o["sampling"]}
