import numpy as np
import sounddevice as sd
from rich.text import Text
from textual.app import App
from textual.widgets import Static

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
KEY_W = 6                      # columns per white key
W = KEY_W * len(WHITE) + 1     # 49
H = 8

WHITE_PRESSED = "on #f5c542"
BLACK_IDLE = "on #444444"
BLACK_PRESSED = "on #ff8c42"


def render_piano(pressed: set) -> Text:
    grid = [[(" ", "") for _ in range(W)] for _ in range(H)]

    def put(r, c, ch, style=""):
        grid[r][c] = (ch, style)

    # White keys: outer frame
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

    # White key fills + labels
    for i, key in enumerate(WHITE):
        x = i * KEY_W
        if key in pressed:
            for r in range(1, H - 1):
                for c in range(x + 1, x + KEY_W):
                    put(r, c, " ", WHITE_PRESSED)
        put(H - 2, x + KEY_W // 2, key.upper(),
            "bold black on #f5c542" if key in pressed else "bold")

    # Black keys drawn over the top
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
    # While playing, highlight the playhead; otherwise highlight the write cursor
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


class Piano(App):
    CSS = """
    Screen {
        align: center middle;
    }
    #help {
        width: 49;
        text-align: center;
        margin-bottom: 1;
    }
    #pattern {
        width: 49;
        height: 1;
        content-align: center middle;
        margin-bottom: 1;
    }
    #piano {
        width: 49;
        height: 8;
    }
    #status {
        width: 49;
        text-align: center;
        margin-top: 1;
    }
    """

    def __init__(self):
        super().__init__()
        self.pressed = set()
        self.pattern = [None] * STEPS
        self.cursor = 0

        self.bpm = 120
        self.playing = False
        self.step = 0
        self.playhead = None
        self.play_timer = None

    def compose(self):
        yield Static("Space play/stop  ·  ↑/↓ tempo  ·  Backspace clear",
                     id="help")
        yield Static(render_pattern(self.pattern, self.cursor), id="pattern")
        yield Static(render_piano(self.pressed), id="piano")
        yield Static(self.status_text(), id="status")

    # --- helpers ---------------------------------------------------------
    def status_text(self):
        state = "▶ playing" if self.playing else "■ stopped"
        return f"{state}  ·  {self.bpm} BPM"

    def refresh_piano(self):
        self.query_one("#piano", Static).update(render_piano(self.pressed))

    def refresh_pattern(self):
        self.query_one("#pattern", Static).update(
            render_pattern(self.pattern, self.cursor, self.playhead))

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
        self.set_timer(hold, lambda k=key: self.release(k))

    def release(self, key):
        self.pressed.discard(key)
        self.refresh_piano()

    # --- playback --------------------------------------------------------
    def start_playback(self):
        if all(k is None for k in self.pattern):
            return  # nothing to play
        self.playing = True
        self.step = 0
        self.tick()  # play step 0 immediately
        self.play_timer = self.set_interval(self.step_seconds, self.tick)
        self.refresh_status()

    def stop_playback(self):
        self.playing = False
        self.playhead = None
        if self.play_timer:
            self.play_timer.stop()
            self.play_timer = None
        self.refresh_pattern()
        self.refresh_status()

    def tick(self):
        self.playhead = self.step
        key = self.pattern[self.step]
        if key is not None:
            self.play_note(key)
        self.refresh_pattern()
        self.step = (self.step + 1) % STEPS

    def change_tempo(self, delta):
        self.bpm = max(40, min(240, self.bpm + delta))
        if self.playing:
            # Restart the timer at the new speed, keeping our place in the loop
            self.play_timer.stop()
            self.play_timer = self.set_interval(self.step_seconds, self.tick)
        self.refresh_status()

    # --- input -----------------------------------------------------------
    def on_key(self, event):
        if event.key == "space":
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
            self.cursor = 0
            if self.playing:
                self.stop_playback()
            else:
                self.refresh_pattern()
        elif event.key in TONES:
            self.play_note(event.key)
            # Only record when we're not playing back
            if not self.playing:
                self.pattern[self.cursor] = event.key
                self.cursor = (self.cursor + 1) % STEPS
                self.refresh_pattern()


Piano().run()
