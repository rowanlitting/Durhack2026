import numpy as np
import sounddevice as sd
from rich.text import Text
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Static

from canvas import PaintCanvas

from textual.timer import Timer

SAMPLE_RATE = 44100
KEYS = {"a": 261.63, "w": 277.18, "s": 293.66, "e": 311.13,
        "d": 329.63, "f": 349.23, "t": 369.99, "g": 392.00,
        "y": 415.30, "h": 440.00, "u": 466.16, "j": 493.88,
        "k": 523.25}


def make_tone(freq, duration=0.6):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    wave = np.sin(2 * np.pi * freq * t)
    envelope = np.exp(-4 * t)
    return (wave * envelope * 0.3).astype(np.float32)


TONES = {k: make_tone(f) for k, f in KEYS.items()}

# --- Piano layout ---------------------------------------------------------
WHITE = ["a", "s", "d", "f", "g", "h", "j", "k"]
BLACK = {1: "w", 2: "e", 4: "t", 5: "y", 6: "u"}  # boundary index -> key
KEY_W = 6
W = KEY_W * len(WHITE) + 1     # 49
H = 8

WHITE_PRESSED = "on #f5c542"
BLACK_IDLE = "on #444444"
BLACK_PRESSED = "on #ff8c42"


def render_piano(pressed: set) -> Text:
    grid = [[(" ", "") for _ in range(W)] for _ in range(H)]

    def put(r, c, ch, style=""):
        grid[r][c] = (ch, style)

    for c in range(W):
        put(0, c, "─")
        put(H - 1, c, "─")
    for i in range(len(WHITE) + 1):
        c = i * KEY_W
        for r in range(1, H - 1):
            put(r, c, "│")
        put(0, c, "┬")
        put(H - 1, c, "┴")
    put(0, 0, "┌"); put(0, W - 1, "┐")
    put(H - 1, 0, "└"); put(H - 1, W - 1, "┘")

    for i, key in enumerate(WHITE):
        x = i * KEY_W
        if key in pressed:
            for r in range(1, H - 1):
                for c in range(x + 1, x + KEY_W):
                    put(r, c, " ", WHITE_PRESSED)
        put(H - 2, x + KEY_W // 2, key.upper(),
            "bold black on #f5c542" if key in pressed else "bold")

    for b, key in BLACK.items():
        cx = b * KEY_W
        fill = BLACK_PRESSED if key in pressed else BLACK_IDLE
        put(1, cx - 2, "┌"); put(1, cx + 2, "┐")
        put(4, cx - 2, "└"); put(4, cx + 2, "┘")
        for c in range(cx - 1, cx + 2):
            put(1, c, "─"); put(4, c, "─")
        put(4, cx, "┬")
        for r in (2, 3):
            put(r, cx - 2, "│"); put(r, cx + 2, "│")
            for c in range(cx - 1, cx + 2):
                put(r, c, " ", fill)
        put(3, cx, key.upper(), f"bold white {fill}")

    text = Text()
    for r, row in enumerate(grid):
        for ch, style in row:
            text.append(ch, style or None)
        if r < H - 1:
            text.append("\n")
    return text


NOTE_NAMES = {"a": "C", "w": "C#", "s": "D", "e": "D#", "d": "E",
              "f": "F", "t": "F#", "g": "G", "y": "G#", "h": "A",
              "u": "A#", "j": "B", "k": "C"}
STEPS = 16


def render_pattern(pattern: list, cursor: int, playhead=None) -> Text:
    text = Text()
    marker = playhead if playhead is not None else cursor
    for i, key in enumerate(pattern):
        group_color = "#f5c542" if (i // 4) % 2 == 0 else "#7fd1ff"
        if key is None:
            label, style = " · ", "dim"
        else:
            label, style = NOTE_NAMES[key].center(3), f"bold {group_color}"
        if i == marker:
            style = "reverse " + style if key is not None else "reverse dim"
        text.append(label, style)
    return text


# ---------- Piano window ----------

class PianoScreen(ModalScreen):
    DEFAULT_CSS = """
    PianoScreen {
        align: center middle;
        background: $background 40%;
    }
    PianoScreen #window {
        width: auto;
        height: auto;
        padding: 1 3;
        border: round #8a8ab0;
        background: $surface;
    }
    PianoScreen #help { width: 49; text-align: center; margin-bottom: 1; }
    PianoScreen #pattern { width: 49; height: 1; content-align: center middle; margin-bottom: 1; }
    PianoScreen #piano { width: 49; height: 8; }
    PianoScreen #status { width: 49; text-align: center; margin-top: 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.pressed = set()
        self.pattern = [None] * STEPS # we can obtain this by calling pattern = self.app.get_screen("piano").pattern
        self.cursor_pos = 0
        self.bpm = 120
        self.playing = False
        self.step_index = 0
        self.playhead = None
        self.play_timer: Timer | None = None

    def compose(self):
        with Vertical(id="window"):
            yield Static("Space play · ↑/↓ tempo · ⌫ clear · Esc close", id="help")
            yield Static(render_pattern(self.pattern, self.cursor_pos), id="pattern")
            yield Static(render_piano(self.pressed), id="piano")
            yield Static(self.status_text(), id="status")

    def on_mount(self) -> None:
        self.query_one("#window").border_title = " Piano "

    def on_screen_suspend(self) -> None:
        self.stop_playback()
        sd.stop()
        self.pressed.clear()
        self.refresh_piano()

    # --- helpers ---
    def status_text(self):
        state = "▶ playing" if self.playing else "■ stopped"
        return f"{state}  ·  {self.bpm} BPM"

    def refresh_piano(self):
        self.query_one("#piano", Static).update(render_piano(self.pressed))

    def refresh_pattern(self):
        self.query_one("#pattern", Static).update(
            render_pattern(self.pattern, self.cursor_pos, self.playhead))

    def refresh_status(self):
        self.query_one("#status", Static).update(self.status_text())

    @property
    def step_seconds(self):
        return 60 / self.bpm

    def play_note(self, key):
        sd.play(TONES[key], SAMPLE_RATE)
        self.pressed.add(key)
        self.refresh_piano()
        hold = min(0.15, self.step_seconds * 0.9)
        self.set_timer(hold, lambda k=key: self.release_key(k))

    def release_key(self, key):
        self.pressed.discard(key)
        self.refresh_piano()

    # --- playback ---
    def start_playback(self):
        if all(k is None for k in self.pattern):
            return
        self.playing = True
        self.step_index = 0
        self.tick_step()
        self.play_timer = self.set_interval(self.step_seconds, self.tick_step)
        self.refresh_status()

    def stop_playback(self):
        self.playing = False
        self.playhead = None
        if self.play_timer:
            self.play_timer.stop()
            self.play_timer = None
        self.refresh_pattern()
        self.refresh_status()

    def tick_step(self):
        self.playhead = self.step_index
        key = self.pattern[self.step_index]
        if key is not None:
            self.play_note(key)
        self.refresh_pattern()
        self.step_index = (self.step_index + 1) % STEPS

    def change_tempo(self, delta):
        self.bpm = max(40, min(240, self.bpm + delta))
        if self.playing:
            self.play_timer.stop()
            self.play_timer = self.set_interval(self.step_seconds, self.tick_step)
        self.refresh_status()

    # --- input ---
    def on_key(self, event):
        if event.key == "escape":
            self.dismiss()
        elif event.key == "space":
            if self.playing:
                self.stop_playback()
            else:
                self.start_playback()
        elif event.key == "up":
            self.change_tempo(+5)
        elif event.key == "down":
            self.change_tempo(-5)
        elif event.key == "backspace":
            self.pattern = [None] * STEPS
            self.cursor_pos = 0
            if self.playing:
                self.stop_playback()
            else:
                self.refresh_pattern()
        elif event.key in TONES:
            self.play_note(event.key)
            if not self.playing:
                self.pattern[self.cursor_pos] = event.key
                self.cursor_pos = (self.cursor_pos + 1) % STEPS
                self.refresh_pattern()


# ---------- Paint window ----------

class PaintScreen(ModalScreen):
    BINDINGS = [("escape", "close", "Close"), ("c", "clear", "Clear")]

    DEFAULT_CSS = """
    PaintScreen {
        align: center middle;
        background: transparent;
    }
    PaintScreen #window {
        width: 80%;
        height: 75%;
        border: round #8a8ab0;
        background: $surface;
    }
    PaintScreen PaintCanvas { width: 1fr; height: 1fr; }
    PaintScreen #hint { height: 1; text-align: center; color: $text-muted; }
    """

    def compose(self):
        with Vertical(id="window"):
            yield PaintCanvas()
            yield Static("Drag to spray · C clear · Esc close", id="hint")

    def on_mount(self) -> None:
        self.query_one("#window").border_title = " Canvas "

    def action_close(self) -> None:
        self.dismiss()

    def action_clear(self) -> None:
        self.query_one(PaintCanvas).clear()
