"""Reusable widgets."""
from textual.widgets import Static

BAR_WIDTH = 24


class MemoryGauge(Static):
    """Bar showing estimated GPU memory against this Mac's budget; turns red (class 'over') if it won't fit."""

    DEFAULT_CSS = """
    MemoryGauge { height: 1; margin-top: 1; color: $success; }
    MemoryGauge.over { color: $error; text-style: bold; }
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.used = 0.0
        self.budget = 0.0

    def update_estimate(self, used_gb: float, budget_gb: float) -> None:
        self.used, self.budget = used_gb, budget_gb
        self.set_class(used_gb > budget_gb, "over")
        self.refresh()

    def render(self) -> str:
        frac = 0.0 if self.budget <= 0 else min(self.used / self.budget, 1.0)
        filled = round(frac * BAR_WIDTH)
        bar = "█" * filled + "░" * (BAR_WIDTH - filled)
        note = "  !! likely too big for this Mac" if self.has_class("over") else ""
        return f"Memory  {bar}  {self.used:.1f} / {self.budget:.1f} GB{note}"
