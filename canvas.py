import math
import random
from dataclasses import dataclass

from rich.segment import Segment
from rich.style import Style
from textual.events import MouseDown, MouseMove, MouseUp
from textual.strip import Strip
from textual.widget import Widget

from functools import lru_cache
from textual.events import Resize

SHADES = " ·░▒▓█"  # index = density level

@lru_cache(maxsize=None)
def cell_style(color: str) -> Style:
    return Style(color=color)

@dataclass
class Drip:
    x: int
    y: int
    color: str
    remaining: int  # how many more rows it can travel



class Canvas(Widget):
    """Base canvas: stores paint, draws it, runs drips and optional fade.
    Subclasses decide *what* paints onto it."""

    palette = ["#5555FF", "#FF55FF", "#FF5555", "#55FFFF"]
    radius = 4
    dots = 10
    falloff = 2.5
    drip_chance = 0.3
    max_drips = 40
    fade_every: float | None = None   # None = paint never fades
    paused = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # (x, y) -> (color, level)
        self.pixels: dict[tuple[int, int], tuple[str, int]] = {}
        self.drips: list[Drip] = []
        self.color = random.choice(self.palette)
        self._rows: dict[int, Strip] = {}

    def on_mount(self) -> None:
        self.set_interval(1 / 10, self.update_drips)
        if self.fade_every:
            self.set_interval(self.fade_every, self.fade)

    def render_line(self, y: int) -> Strip:
        cached = self._rows.get(y)
        if cached is not None:
            return cached

        width = self.size.width
        segments = []
        blank = 0
        for x in range(width):
            cell = self.pixels.get((x, y))
            if cell:
                if blank:
                    segments.append(Segment(" " * blank))
                    blank = 0
                color, level = cell
                segments.append(Segment(SHADES[level], cell_style(color)))
            else:
                blank += 1
        if blank:
            segments.append(Segment(" " * blank))

        strip = Strip(segments, width)
        self._rows[y] = strip
        return strip

    def on_resize(self, event: Resize) -> None:
        self._rows.clear()

    def clear(self) -> None:
        self.pixels.clear()
        self.drips.clear()
        self._rows.clear()
        self.refresh()

    def spray_at(self, cx: float, cy: float, dots: int, color: str | None = None) -> None:
        color = color or self.color
        for _ in range(dots):
            angle = random.uniform(0, 2 * math.pi)
            dist = self.radius * random.random() ** self.falloff

            x = round(cx + math.cos(angle) * dist * 2)   # *2 for cell aspect ratio
            y = round(cy + math.sin(angle) * dist)

            if 0 <= x < self.size.width and 0 <= y < self.size.height:
                _, level = self.pixels.get((x, y), (color, 0))
                new_level = min(level + 1, len(SHADES) - 1)
                self.pixels[(x, y)] = (color, new_level)
                self._rows.pop(y, None)

                if (new_level == len(SHADES) - 1 and level < new_level
                        and len(self.drips) < self.max_drips
                        and random.random() < self.drip_chance):
                    self.drips.append(Drip(x, y, color, random.randint(3, 12)))

    def update_drips(self) -> None:
        if self.paused:
            return

        if not self.drips:
            return

        alive = []
        for drip in self.drips:
            if random.random() < 0.7:
                drip.y += 1
                drip.remaining -= 1
                if drip.y >= self.size.height:
                    continue

                level = 3 if drip.remaining > 3 else max(1, drip.remaining)
                _, old = self.pixels.get((drip.x, drip.y), (drip.color, 0))
                self.pixels[(drip.x, drip.y)] = (drip.color, max(old, level))
                self._rows.pop(drip.y, None)

            if drip.remaining > 0:
                alive.append(drip)

        self.drips = alive
        self._rows.clear()
        self.refresh()

    def fade(self) -> None:
        if self.paused:
            return

        if not self.pixels:
            return
        self.pixels = {
            pos: (color, level - 1)
            for pos, (color, level) in self.pixels.items()
            if level > 1
        }
        self._rows.clear()
        self.refresh()


@dataclass
class Hand:
    color: str
    x: float = 0.0
    y: float = 0.0
    angle: float = 0.0
    turn: float = 0.0          # current turn rate (radians per frame)
    speed: float = 0.15
    pen_down: bool = True


class AutoSprayCanvas(Canvas):
    """Menu background: several slow hands, each spraying its own muted color."""

    palette = ["#3F4A8A", "#7A4A7A", "#8A4A4A", "#4A7A7A"]   # desaturated
    dots = 6              # lighter deposit so paint stays soft

    # Fading: check often, but only a small fraction of cells step down each time
    fade_every = 0.4
    fade_chance = 0.08

    # Steering
    max_turn = 0.06       # limit on how sharply a hand can curve
    turn_jitter = 0.008   # how much the turn rate drifts each frame
    turn_damping = 0.97   # pulls the turn rate back toward a straight line

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hands = [
            Hand(
                color=c,
                angle=random.uniform(0, 2 * math.pi),
                speed=random.uniform(0.10, 0.20),
            )
            for c in self.palette
        ]
        self.hands_ready = False

        # Mouse drawing
        self.spray_timer = None
        self.mouse_pos = (0, 0)
        self.last_pos = (0, 0)
        self.brush_index = 0

    def on_mount(self) -> None:
        # Canvas.on_mount (drips + fade timers) is also called by Textual,
        # so only the movement timer is needed here.
        self.set_interval(1 / 30, self.wander)

    def wander(self) -> None:
        if self.paused:
            return

        w, h = self.size.width, self.size.height
        if w < 2 or h < 2:
            return

        if not self.hands_ready:
            for hand in self.hands:
                hand.x = random.uniform(0, w)
                hand.y = random.uniform(0, h)
            self.hands_ready = True

        for hand in self.hands:
            # Smooth steering: nudge the turn *rate*, not the heading itself
            hand.turn += random.uniform(-self.turn_jitter, self.turn_jitter)
            hand.turn *= self.turn_damping
            hand.turn = max(-self.max_turn, min(self.max_turn, hand.turn))
            hand.angle += hand.turn

            x = hand.x + math.cos(hand.angle) * hand.speed * 2
            y = hand.y + math.sin(hand.angle) * hand.speed

            if not 0 <= x < w:
                hand.angle = math.pi - hand.angle
                hand.turn = -hand.turn
                x = min(max(x, 0), w - 1)
            if not 0 <= y < h:
                hand.angle = -hand.angle
                hand.turn = -hand.turn
                y = min(max(y, 0), h - 1)

            hand.x, hand.y = x, y

            if hand.pen_down:
                if random.random() < 0.004:      # lifts the pen rarely
                    hand.pen_down = False
            elif random.random() < 0.03:
                hand.pen_down = True

            if hand.pen_down:
                self.spray_at(x, y, self.dots, hand.color)

        self.refresh()

    def fade(self) -> None:
        if self.paused:
            return

        if not self.pixels:
            return
        faded = {}
        for pos, (color, level) in self.pixels.items():
            if random.random() < self.fade_chance:
                level -= 1
            if level >= 1:
                faded[pos] = (color, level)
        self.pixels = faded
        self.refresh()


    # ---------- Mouse drawing ----------

    def on_mouse_down(self, event: MouseDown) -> None:
        self.capture_mouse()
        self.mouse_pos = self.last_pos = (event.x, event.y)
        # Each new stroke uses the next palette color
        self.brush_index = (self.brush_index + 1) % len(self.palette)
        self.spray_timer = self.set_interval(1 / 30, self.spray_mouse)

    def on_mouse_move(self, event: MouseMove) -> None:
        self.mouse_pos = (event.x, event.y)
        
    def on_mouse_up(self, event: MouseUp) -> None:
        if self.spray_timer:
            self.spray_timer.stop()
            self.spray_timer = None
        self.release_mouse()

    def spray_mouse(self) -> None:
        x0, y0 = self.last_pos
        x1, y1 = self.mouse_pos
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        dots_per_step = max(self.dots // steps, 1)
        color = self.palette[self.brush_index]

        # Interpolate between frames so fast drags leave a continuous line
        for i in range(steps + 1):
            t = i / steps
            self.spray_at(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, dots_per_step, color)

        self.last_pos = self.mouse_pos
        self.refresh()

class PaintCanvas(Canvas):
    """Your original interactive version: spray with the mouse."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.spray_timer = None
        self.mouse_pos = (0, 0)
        self.last_pos = (0, 0)

    def on_mouse_down(self, event: MouseDown) -> None:
        self.capture_mouse()
        self.mouse_pos = self.last_pos = (event.x, event.y)
        self.spray_timer = self.set_interval(1 / 20, self.spray)

    def on_mouse_move(self, event: MouseMove) -> None:
        self.mouse_pos = (event.x, event.y)

    def on_mouse_up(self, event: MouseUp) -> None:
        if self.spray_timer:
            self.spray_timer.stop()
            self.spray_timer = None
        self.release_mouse()

    def spray(self) -> None:
        x0, y0 = self.last_pos
        x1, y1 = self.mouse_pos
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        dots_per_step = max(self.dots // steps, 1)

        for i in range(steps + 1):
            t = i / steps
            self.spray_at(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, dots_per_step)

        self.last_pos = self.mouse_pos
        self.refresh()
