"""Memory fit estimate for a model + context + KV precision on this Mac."""
import subprocess


def total_ram_gb() -> float:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"])) / 2**30
    except (OSError, ValueError, subprocess.CalledProcessError):
        return 0.0


def gpu_budget_gb(ram_gb: float) -> float:
    """Metal allows ~75% of RAM on the GPU; keep 1 GiB spare as llama.cpp does."""
    return ram_gb * 0.75 - 1.0


def estimate_gb(m: dict, ctx: int, kv: str) -> float:
    """GiB on the GPU: weights + KV (measured per 1K tokens) + ~0.55 for recurrent state and compute buffers."""
    return m["weights_gib"] + m["kv_mib_per_1k"][kv] * (ctx / 1024) / 1024 + 0.55


def clamp_ctx(value: int, max_ctx: int) -> int:
    return max(1024, min(value, max_ctx))


def parse_tokens(text: str) -> int:
    """'48k' -> 49152, '49152' -> 49152, '1,024' -> 1024. Raises ValueError on anything else or <= 0."""
    t = text.strip().lower().replace(",", "")
    mult = 1024 if t.endswith("k") else 1
    try:
        n = int(float(t.rstrip("k")) * mult)
    except OverflowError as e:
        raise ValueError(f"not a token count: {text!r}") from e
    if n <= 0:
        raise ValueError(f"token count must be positive: {text!r}")
    return n
