"""models.json: runtimes and models, plus where their files live under the project root."""
import json
from pathlib import Path

from . import user_models


class RegistryError(Exception):
    pass


def load_registry(path: Path) -> dict:
    try:
        reg = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise RegistryError(f"cannot read {path}: {e}") from e
    if not isinstance(reg, dict) or "models" not in reg or "runtimes" not in reg:
        raise RegistryError(f"{path} must contain 'runtimes' and 'models'")
    for mid, m in user_models.load(path.with_name(user_models.FILENAME)).items():
        if m["runtime"] in reg["runtimes"]:
            reg["models"].setdefault(mid, m)  # the curated list wins on an id clash
    user_models.apply_overrides(reg["models"], user_models.load_overrides(path.with_name(user_models.OVERRIDES_FILENAME)))
    for m in reg["models"].values():
        m["runtimes"] = allowed_runtimes(reg, m)
    return reg


def allowed_runtimes(reg: dict, m: dict) -> list[str]:
    """Runtimes the user may pick for model m: its own list (default: all), known ones only, default first."""
    wanted = m.get("runtimes")
    wanted = list(reg["runtimes"]) if not isinstance(wanted, list) else wanted
    out = [r for r in wanted if r in reg["runtimes"]]
    if m["runtime"] not in out:
        out.insert(0, m["runtime"])
    return out


def runtime_dir(root: Path, rt: dict) -> Path:
    return root / rt["dir"].format(tag=rt["tag"])


def runtime_bin(root: Path, rt: dict, name: str) -> Path:
    return runtime_dir(root, rt) / name


def model_path(root: Path, m: dict) -> Path:
    return root / "models" / m["file"]
