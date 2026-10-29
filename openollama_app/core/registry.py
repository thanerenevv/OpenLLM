"""models.json: runtimes and models, plus where their files live under the project root."""
import json
from pathlib import Path


class RegistryError(Exception):
    pass


def load_registry(path: Path) -> dict:
    try:
        reg = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise RegistryError(f"cannot read {path}: {e}") from e
    if not isinstance(reg, dict) or "models" not in reg or "runtimes" not in reg:
        raise RegistryError(f"{path} must contain 'runtimes' and 'models'")
    return reg


def runtime_dir(root: Path, rt: dict) -> Path:
    return root / rt["dir"].format(tag=rt["tag"])


def runtime_bin(root: Path, rt: dict, name: str) -> Path:
    return runtime_dir(root, rt) / name


def model_path(root: Path, m: dict) -> Path:
    return root / "models" / m["file"]
