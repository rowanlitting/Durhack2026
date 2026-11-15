import numpy as np
import jax
from rich.color import Color
from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.app import App, ComposeResult
from textual.strip import Strip
from textual.widget import Widget

# textual-image probes the terminal at import time, so import it at the top
# level, before the app starts. If it's missing, fall back to half-blocks.
try:
    from PIL import Image as PILImage
    from textual_image.widget import Image as AutoImage
    from textual_image.widget import HalfcellImage, UnicodeImage

    # True only if textual-image picked a real graphics protocol (kitty/sixel)
    USE_GRAPHICS = AutoImage not in (HalfcellImage, UnicodeImage)
except ImportError:
    PILImage = None
    AutoImage = None
    USE_GRAPHICS = False

MAX_ITER = 100

# Pixels per terminal cell when using graphics mode (cells are ~1:2, so 8x16)
PX_PER_COL, PX_PER_ROW = 4, 8

# Colour stops from far (dark) to near the set (bright)
_STOPS = np.array([
    [30, 28, 70],     # far away: dark indigo (no longer black)
    [70, 55, 140],    # blue-violet
    [150, 80, 170],   # orchid
    [240, 120, 130],  # coral
    [255, 215, 140],  # warm gold at the edge
], dtype=np.float32)

_INSIDE = np.array([255, 235, 190], dtype=np.float32)  # bounded points glow


def colorize(grid: np.ndarray, max_iter: int) -> np.ndarray:
    """Map the 0..100 field to an (H, W, 3) uint8 RGB array."""
    g = np.asarray(grid, dtype=np.float32)
    inside = g <= 0

    # 1 = just escaped late (close to the set), 0 = escaped immediately
    heat = 1.0 - np.clip(g / max_iter, 0, 1)
    heat = heat ** 1.5            # higher power = tighter glow

    # Piecewise-linear lookup through the colour stops
    pos = heat * (len(_STOPS) - 1)
    lo = np.floor(pos).astype(int).clip(0, len(_STOPS) - 2)
    frac = (pos - lo)[..., None]
    rgb = _STOPS[lo] * (1 - frac) + _STOPS[lo + 1] * frac

    rgb[inside] = _INSIDE
    return rgb.astype(np.uint8)


class FractalView(Widget):
    """Fallback renderer: two vertical pixels per cell using '▀'."""

    def __init__(self, max_iter: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self._rgb = np.zeros((0, 0, 3), dtype=np.uint8)

    def set_grid(self, grid) -> None:
        self._rgb = colorize(np.asarray(grid), self.max_iter)
        self.refresh()

    def render_line(self, y: int) -> Strip:
        h, w, _ = self._rgb.shape

        def pixel(row: int, col: int) -> Color:
            if row < h and col < w:
                r, g, b = self._rgb[row, col]
                return Color.from_rgb(int(r), int(g), int(b))
            return Color.from_rgb(0, 0, 0)

        return Strip([
            Segment("▀", Style(color=pixel(2 * y, x), bgcolor=pixel(2 * y + 1, x)))
            for x in range(self.size.width)
        ])


class FractalApp(App):
    CSS = """
    AutoImage, FractalView { width: 100%; height: 100%; }
    """

    def compose(self) -> ComposeResult:
        if USE_GRAPHICS:
            yield AutoImage(id="fractal")
        else:
            yield FractalView(max_iter=MAX_ITER, id="fractal")

    def on_resize(self, event: events.Resize) -> None:
        rows, cols = event.size.height, event.size.width
        key = jax.random.PRNGKey(0)

        if USE_GRAPHICS:
            view = self.query_one("#fractal", AutoImage)
            grid = jax.random.randint(
                key, (rows * PX_PER_ROW, cols * PX_PER_COL), 0, 101
            )
            rgb = colorize(np.asarray(grid), MAX_ITER)
            view.image = PILImage.fromarray(rgb, "RGB")
        else:
            view = self.query_one("#fractal", FractalView)
            grid = jax.random.randint(key, (2 * rows, cols), 0, 101)
            view.set_grid(grid)


if __name__ == "__main__":
    FractalApp().run()
