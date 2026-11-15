import random
import time

from rich.segment import Segment
from rich.style import Style
from textual.app import App, ComposeResult
from textual.widgets import Button
from textual.strip import Strip

from canvas import AutoSprayCanvas, SHADES
from title import HERE, load_variants, line_width

from windows import PianoScreen, PaintScreen

SUBTITLE_STYLE = Style(color="#c8c8e0", bold=True)

class Letter:
    """One glitching letter: pre-rendered variants plus timing state (no widget)."""

    def __init__(self, variants, min_delay=0.08, max_delay=1.6,
                 wobble_min=0.15, wobble_max=0.5):
        # Per variant: rows of (dx, char, style) for non-space cells only
        self.variants = [self._to_rows(v) for v in variants]
        self.width = max(line_width(l) for v in variants for l in v)
        self.height = max(len(v) for v in variants)
        self.current = random.randrange(len(self.variants))
        self.wob = 1  # 0, 1 or 2 -> bob up / rest / down
        self.min_delay, self.max_delay = min_delay, max_delay
        self.wobble_min, self.wobble_max = wobble_min, wobble_max
        now = time.monotonic()
        self.next_switch = now + random.uniform(min_delay, max_delay)
        self.next_wobble = now + random.uniform(wobble_min, wobble_max)

    @staticmethod
    def _to_rows(lines):
        rows = []
        for line in lines:
            row, x = [], 0
            for text, style in line:
                fg = Style(color=style.color) if style and style.color else None
                for ch in text:
                    if ch != " ":
                        row.append((x, ch, fg))
                    x += 1
            rows.append(row)
        return rows

    def update(self, now: float) -> bool:
        changed = False
        if len(self.variants) > 1 and now >= self.next_switch:
            self.current = random.choice(
                [i for i in range(len(self.variants)) if i != self.current]
            )
            self.next_switch = now + random.uniform(self.min_delay, self.max_delay)
            changed = True
        if now >= self.next_wobble:
            self.wob = random.choices([0, 1, 2], weights=[1, 4, 1])[0]
            self.next_wobble = now + random.uniform(self.wobble_min, self.wobble_max)
            changed = True
        return changed


class TitleSprayCanvas(AutoSprayCanvas):
    """Spray background with the glitch title composited on top, cell by cell."""

    title_top = 5  # row where the letters rest

    def __init__(self, text: str, subtitle: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.letters = [
            Letter(load_variants(HERE / f"{ch.lower()}_fonts.html")) for ch in text
        ]
        self.subtitle = subtitle
        self.title_height = max(l.height for l in self.letters)

    def on_mount(self) -> None:
        self.set_interval(1 / 30, self.tick_title)

    def tick_title(self) -> None:
        if self.paused:
            return

        now = time.monotonic()
        changed = False
        for letter in self.letters:
            changed |= letter.update(now)
        if changed:
            self.refresh()

    def render_line(self, y: int) -> Strip:
        w = self.size.width

        # 1. the paint, exactly as before (unstyled blanks = terminal background)
        segs = []
        for x in range(w):
            cell = self.pixels.get((x, y))
            if cell:
                color, level = cell
                segs.append(Segment(SHADES[level], Style(color=color)))
            else:
                segs.append(Segment(" "))

        # 2. the title on top, only where a glyph has a visible character
        total = sum(l.width for l in self.letters)
        ox = (w - total) // 2
        for letter in self.letters:
            rows = letter.variants[letter.current]
            r = y - (self.title_top + letter.wob - 1)
            if 0 <= r < len(rows):
                for dx, ch, style in rows[r]:
                    x = ox + dx
                    if 0 <= x < w:
                        segs[x] = Segment(ch, style)
            ox += letter.width

        # 3. a line of text below the letters, non-space cells only
        sub_y = self.title_top + self.title_height + 2
        if self.subtitle and y == sub_y:
            sx = (w - len(self.subtitle)) // 2
            for i, ch in enumerate(self.subtitle):
                x = sx + i
                if ch != " " and 0 <= x < w:
                    segs[x] = Segment(ch, SUBTITLE_STYLE)

        return Strip(segs)


class DrawApp(App):
    CSS = """
    Screen {
        layers: back front;
        align: center middle;
    }
    TitleSprayCanvas {
        layer: back;
        width: 100%;
        height: 100%;
    }
    .menu-btn {
        layer: front;
        width: 30;
        height: 3;
        margin-bottom: 1;
        border: round #8a8ab0;
        background: $surface;
        text-style: bold;
    }
    .menu-btn:hover, .menu-btn:focus {
        border: round #e0e0ff;
    }
    """

    def compose(self) -> ComposeResult:
        yield TitleSprayCanvas("KOOS", subtitle="designed to compose maths, art, and, music")
        yield Button("Piano", id="open-piano", classes="menu-btn")
        yield Button("Canvas", id="open-canvas", classes="menu-btn")

    def on_mount(self) -> None:
        self.install_screen(PianoScreen(), name="piano")
        self.install_screen(PaintScreen(), name="paint")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "open-piano":
            self.open_window("piano")
        elif event.button.id == "open-canvas":
            self.open_window("paint")

    def open_window(self, name: str) -> None:
        canvas = self.query_one(TitleSprayCanvas)
        canvas.paused = True
        # The callback runs when the window calls self.dismiss()
        self.push_screen(name, lambda _: setattr(canvas, "paused", False))

if __name__ == "__main__":
    DrawApp().run()
