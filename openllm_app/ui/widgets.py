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


# ------------------------------------------------------------------ chat / log
from collections import deque  # noqa: E402

from textual.containers import VerticalScroll  # noqa: E402
from textual.widgets import Collapsible, RichLog  # noqa: E402


class Line(Static):
    """One transcript line. Model and user text is never parsed as markup."""

    def __init__(self, text: str, **kw):
        super().__init__(text, markup=False, **kw)
        self.text = text

    def append(self, more: str) -> None:
        self.text += more
        self.update(self.text)


class ThinkingBlock(Collapsible):
    def __init__(self):
        self._body = Line("")
        super().__init__(self._body, title="thinking…", collapsed=False)

    @property
    def text(self) -> str:
        return self._body.text

    def append(self, more: str) -> None:
        self._body.append(more)

    def finish(self) -> None:
        self.title = f"thinking ({len(self.text.split())} words)"
        self.collapsed = True


class ChatPane(VerticalScroll):
    DEFAULT_CSS = """
    ChatPane { height: 1fr; padding: 0 1; }
    ChatPane .user { color: $accent; margin-top: 1; }
    ChatPane .notice { color: $warning; }
    ChatPane .error { color: $error; }
    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self._think: ThinkingBlock | None = None
        self._reply: Line | None = None

    def _add(self, widget):
        self.mount(widget)
        self.scroll_end(animate=False)
        return widget

    def add_user(self, text: str) -> None:
        self._add(Line(f"you ▸ {text}", classes="user"))

    def add_notice(self, text: str, error: bool = False) -> None:
        self._add(Line(text, classes="error" if error else "notice"))

    def append_reasoning(self, text: str) -> None:
        if self._think is None:
            self._think = self._add(ThinkingBlock())
        self._think.append(text)

    def append_content(self, text: str) -> None:
        if self._think is not None:
            self._think.finish()
        if self._reply is None:
            self._reply = self._add(Line("model ▸ "))
        self._reply.append(text)
        self.scroll_end(animate=False)

    def end_reply(self) -> None:
        if self._think is not None:
            self._think.finish()
        self._think = self._reply = None

    def transcript_text(self) -> str:
        parts = []
        for w in self.children:
            if isinstance(w, ThinkingBlock):
                parts.append(f"[thinking] {w.text}")
            elif isinstance(w, Line):
                parts.append(w.text)
        return "\n".join(parts)


class LogPane(RichLog):
    def __init__(self, **kw):
        super().__init__(max_lines=5000, wrap=True, highlight=False, markup=False, **kw)
        self.texts: deque[str] = deque(maxlen=5000)

    def add(self, line: str) -> None:
        self.texts.append(line)
        self.write(line)


class SnippetBox(Static):
    can_focus = True
    DEFAULT_CSS = """
    SnippetBox { border: round $primary; padding: 0 1; margin-bottom: 1; }
    SnippetBox:focus { border: round $accent; }
    """

    def __init__(self, snippet: str, **kw):
        super().__init__(snippet, markup=False, **kw)
        self.snippet = snippet


# ------------------------------------------------------------------ selector
from rich.text import Text  # noqa: E402
from textual.binding import Binding  # noqa: E402
from textual.message import Message  # noqa: E402
from textual.widget import Widget  # noqa: E402


class Cycler(Widget, can_focus=True):
    """A titled `< value >` selector: Left/Right (or a click on either half) steps through the options."""

    DEFAULT_CSS = """
    Cycler { width: 100%; height: 4; padding: 0 1; border: round ansi_bright_black; background: transparent; }
    Cycler:focus { border: round $accent; }
    Cycler.single { opacity: 70%; }
    """
    BINDINGS = [Binding("left,h", "step(-1)", "Previous", show=False),
                Binding("right,l", "step(1)", "Next", show=False)]

    class Changed(Message):
        def __init__(self, cycler: "Cycler", value: str) -> None:
            super().__init__()
            self.cycler, self.value = cycler, value

    def __init__(self, title: str, options: list[tuple[str, str]], value: str, **kw) -> None:
        super().__init__(**kw)
        self.title = title
        self.options: list[tuple[str, str]] = []
        self._value = value
        self.set_options(options, value)

    @property
    def value(self) -> str:
        return self._value

    @property
    def label(self) -> str:
        return dict(self.options).get(self._value, self._value)

    def set_options(self, options: list[tuple[str, str]], value: str) -> None:
        self.options = list(options)
        self._value = value if value in dict(self.options) else (self.options[0][0] if self.options else value)
        self.set_class(len(self.options) <= 1, "single")
        self.refresh()

    def select(self, value: str) -> None:
        """Choose an option programmatically (posts Changed like a keypress does)."""
        if value in dict(self.options) and value != self._value:
            self._value = value
            self.refresh()
            self.post_message(self.Changed(self, value))

    def action_step(self, delta: int) -> None:
        keys = [k for k, _ in self.options]
        if len(keys) > 1:
            self.select(keys[(keys.index(self._value) + delta) % len(keys)])

    def on_click(self, event) -> None:
        self.focus()
        self.action_step(-1 if event.x < self.size.width // 2 else 1)

    def render(self) -> Text:
        arrows = len(self.options) > 1
        out = Text(self.title + "\n", style="bold")
        out.append("< " if arrows else "  ", style="dim")
        out.append(self.label)
        out.append(" >" if arrows else "")
        return out

