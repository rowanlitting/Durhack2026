import numpy as np
import jax
from PIL import Image as PILImage
from rich.color import Color
from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.app import App, ComposeResult
from textual.strip import Strip
from textual.widget import Widget

# textual-image probes the terminal at import time, so import it at the top
# level, before the app starts.
from textual_image.widget import Image as AutoImage
from textual_image.widget import HalfcellImage, UnicodeImage

MAX_ITER = 100

# True if textual-image picked a real graphics protocol (kitty/sixel)
USE_GRAPHICS = AutoImage not in (HalfcellImage, UnicodeImage)

# Pixels per terminal cell when using graphics mode (cells are ~1:2, so 8x16)
PX_PER_COL, PX_PER_ROW = 8, 16


def colorize(grid: np.ndarray, max_iter: int) -> np.ndarray:
    """Map iteration counts to an (H, W, 3) uint8 RGB array."""
    t = grid / max_iter
    r = 0.5 + 0.5 * np.cos(2 * np.pi * (3.0 * t + 0.00))
    g = 0.5 + 0.5 * np.cos(2 * np.pi * (3.0 * t + 0.33))
    b = 0.5 + 0.5 * np.cos(2 * np.pi * (3.0 * t + 0.67))
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    rgb[grid >= max_iter] = 0  # never blew up -> black
    return rgb


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
