import random
import math
from textual.app import App, ComposeResult
from textual.widget import Widget
from textual.strip import Strip
from textual.events import MouseDown, MouseMove, MouseUp
from rich.segment import Segment
from rich.style import Style
from dataclasses import dataclass

@dataclass
class Drip:
    x: int
    y: int
    color: str
    remaining: int # how many more rows it can travel

@dataclass
class Point:
    count: int
    prev: tuple[int, int] | None

SHADES = " ·░▒▓█"  # index = density level

#-------- general canvas widget --------

class Canvas(Widget):
    def __init__(self) -> None:
        super().__init__()
        # (x, y) -> (color, level)
        self.pixels: dict[tuple[int, int], tuple[str, int]] = {}

        self.points: dict[tuple[int, int], Point] = {}
        self.last_point: tuple[int, int] | None = None
        self.point_timer = None

        self.color = "red"
        self.spray_timer = None
        self.radius = 4        # spray size
        self.dots = 10         # dots per tick
        
        self.mouse_pos = (0, 0)
        self.last_pos = (0, 0)
        self.falloff = 2.5
        
        self.drips: list[Drip] = []
        self.drip_chance = 0.3
        self.max_drips = 40

    def render_line(self, y: int) -> Strip:
        segments = []
        for x in range(self.size.width):
            cell = self.pixels.get((x, y))
            if cell:
                color, level = cell
                segments.append(Segment(SHADES[level], Style(color=color)))
            else:
                segments.append(Segment(" "))
        return Strip(segments)

    def on_mount(self) -> None:
        self.set_interval(1 / 10, self.update_drips)

    def on_mouse_down(self, event: MouseDown) -> None:
        self.capture_mouse()
        self.mouse_pos = self.last_pos = (event.x, event.y)

        self.last_point = None

        self.spray_timer = self.set_interval(1 / 30, self.spray)
        self.point_timer = self.set_interval(1 / 2, self.point)

    def on_mouse_move(self, event: MouseMove) -> None:
        self.mouse_pos = (event.x, event.y)

    def on_mouse_up(self, event: MouseUp) -> None:
        if self.spray_timer:
            self.spray_timer.stop()
            self.spray_timer = None
        if self.point_timer:
            self.point_timer.stop()
            self.point_timer = None
        self.release_mouse()

    def point(self) -> None:
        pos = self.mouse_pos
        if pos in self.points:
            self.points[pos].count += 1
        else:
            self.points[pos] = Point(count=1, prev=self.last_point)
        self.last_point = pos

    def spray(self) -> None:
        x0, y0 = self.last_pos
        x1, y1 = self.mouse_pos
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        dots_per_step = max(self.dots // steps, 1)

        for i in range(steps + 1):
            t = i / steps
            cx = x0 + (x1 - x0) * t
            cy = y0 + (y1 - y0) * t
            self.spray_at(cx, cy, dots_per_step)

        self.last_pos = self.mouse_pos
        self.refresh()

    def spray_at(self, cx: float, cy: float, dots: int) -> None:
        for _ in range(dots):
            angle = random.uniform(0, 2 * math.pi)
            dist = self.radius * random.random() ** self.falloff

            x = round(cx + math.cos(angle) * dist * 2)   # *2 for cell aspect ratio
            y = round(cy + math.sin(angle) * dist)

            if 0 <= x < self.size.width and 0 <= y < self.size.height:
                _, level = self.pixels.get((x, y), (self.color, 0))
                new_level = min(level + 1, len(SHADES) - 1)
                self.pixels[(x, y)] = (self.color, new_level)

                # just became fully saturated: maybe start a drip
                if (new_level == len(SHADES) - 1 and level < new_level
                        and len(self.drips) < self.max_drips
                        and random.random() < self.drip_chance):
                    self.drips.append(Drip(x, y, self.color, random.randint(3, 12)))

    def update_drips(self) -> None:
        if not self.drips:
            return

        alive = []
        for drip in self.drips:
            # skipping some ticks gives each drip a different speed
            if random.random() < 0.7:
                drip.y += 1
                drip.remaining -= 1
                if drip.y >= self.size.height:
                    continue   # ran off the bottom, discard it

                # trail thins out as the drip runs dry
                level = 3 if drip.remaining > 3 else max(1, drip.remaining)
                _, old = self.pixels.get((drip.x, drip.y), (drip.color, 0))
                self.pixels[(drip.x, drip.y)] = (drip.color, max(old, level))

            if drip.remaining > 0:
                alive.append(drip)

        self.drips = alive
        self.refresh()

class DrawApp(App):
    def compose(self) -> ComposeResult:
        yield Canvas()


if __name__ == "__main__":
    DrawApp().run()
