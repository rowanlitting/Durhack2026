import random
import re
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path

from rich.style import Style
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

HERE = Path(__file__).parent
Segment = tuple[str, Style | None]


# ---------- HTML parsing ----------

def parse_style(css: str) -> Style | None:
    fg = None
    for decl in css.split(";"):
        if ":" not in decl:
            continue
        key, val = (s.strip().lower() for s in decl.split(":", 1))
        if key == "color":
            fg = val
    return Style(color=fg) if fg else None

class PreParser(HTMLParser):
    """Turns the inside of one <pre> into lines of (text, style) segments."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[list[Segment]] = [[]]
        self._stack: list[Style | None] = []

    def handle_starttag(self, tag, attrs) -> None:
        if tag == "span":
            self._stack.append(parse_style(dict(attrs).get("style") or ""))

    def handle_endtag(self, tag):
        if tag == "span" and self._stack:
            self._stack.pop()

    def handle_data(self, data):
        style = self._stack[-1] if self._stack else None
        for i, part in enumerate(data.split("\n")):
            if i > 0:
                self.lines.append([])
            if part:
                self.lines[-1].append((part, style))


@lru_cache(maxsize=None)
def load_variants(path: Path) -> tuple:
    """Load and cache every <pre> block in a file (cached so both O's share one parse)."""
    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"Missing font file: {path.name} (expected next to the script)")

    variants = []
    for block in re.findall(r"<pre[^>]*>(.*?)</pre>", source, flags=re.S | re.I):
        parser = PreParser()
        parser.feed(block.lstrip("\n"))
        lines = parser.lines
        while lines and not lines[-1]:
            lines.pop()
        if lines:
            variants.append(lines)
    if not variants:
        raise SystemExit(f"No <pre> blocks found in {path.name}")
    return tuple(variants)


# ---------- Rendering ----------

def line_width(line: list[Segment]) -> int:
    return sum(len(text) for text, _ in line)


def to_text(lines: list[list[Segment]], width: int, height: int) -> Text:
    """Build a fixed-size Rich Text so the widget never resizes."""
    out = Text(no_wrap=True)
    for i in range(height):
        line = lines[i] if i < len(lines) else []
        for text, style in line:
            out.append(text, style=style)
        out.append(" " * (width - line_width(line)))
        if i < height - 1:
            out.append("\n")
    return out


# ---------- Widgets ----------

class GlitchLetter(Static):
    DEFAULT_CSS = """
    GlitchLetter {
        width: auto;
        height: auto;
        margin: 1 0;
    }
    """

    def __init__(
        self,
        variants,
        min_delay: float = 0.08,
        max_delay: float = 1.6,
        wobble: bool = True,
        wobble_min: float = 0.15,
        wobble_max: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        width = max(line_width(l) for v in variants for l in v)
        height = max(len(v) for v in variants)
        self._rendered = [to_text(v, width, height) for v in variants]
        self._current = random.randrange(len(self._rendered))
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.wobble = wobble
        self.wobble_min = wobble_min
        self.wobble_max = wobble_max

    def on_mount(self) -> None:
        self.update(self._rendered[self._current])
        if len(self._rendered) > 1:
            self._schedule()
        if self.wobble:
            self._schedule_wobble()

    # --- font switching ---
    def _schedule(self) -> None:
        self.set_timer(random.uniform(self.min_delay, self.max_delay), self._tick)

    def _tick(self) -> None:
        self._current = random.choice(
            [i for i in range(len(self._rendered)) if i != self._current]
        )
        self.update(self._rendered[self._current])
        self._schedule()

    # --- wobble ---
    def _schedule_wobble(self) -> None:
        self.set_timer(
            random.uniform(self.wobble_min, self.wobble_max), self._wobble_tick
        )

    def _wobble_tick(self) -> None:
        # Mostly rest at the middle, occasionally bob up or down a row.
        top = random.choices([0, 1, 2], weights=[1, 4, 1])[0]
        self.styles.margin = (top, 0, 2 - top, 0)
        self._schedule_wobble()

class Title(Horizontal):
    """A row of GlitchLetters. Letter X loads x_fonts.html."""

    DEFAULT_CSS = """
    Title {
        width: auto;
        height: auto;
        padding: 0 3;
    }
    """

    def __init__(self, text: str, fonts_dir: Path = HERE, **kwargs) -> None:
        super().__init__(**kwargs)
        self.text = text
        self.fonts_dir = fonts_dir

    def compose(self) -> ComposeResult:
        for ch in self.text:
            yield GlitchLetter(load_variants(self.fonts_dir / f"{ch.lower()}_fonts.html"))


class MenuApp(App):
    CSS = """
    Screen {
        align: center top;
    }
    Title {
        margin-top: 4;
    }
    """

    def compose(self) -> ComposeResult:
        yield Title("KOOS")


if __name__ == "__main__":
    MenuApp().run()
