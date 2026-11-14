import numpy as np
import jax.numpy as jnp
from textual.widget import Widget
from textual.strip import Strip
from textual.reactive import reactive
from rich.segment import Segment
from rich.style import Style
from rich.color import Color


class FractalView(Widget):
    """Renders a 2D array of escape-iteration counts using half-blocks.

    Each terminal cell shows two vertical pixels: the top one as the
    foreground of '▀' and the bottom one as the background.
    """

    def __init__(self, max_iter: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_iter = max_iter
        self._rgb = np.zeros((0, 0, 3), dtype=np.uint8)  # (pixel_h, pixel_w, 3)

    # ---- data -> colour ------------------------------------------------
    def _colorize(self, grid: np.ndarray) -> np.ndarray:
        """Map iteration counts to an (H, W, 3) uint8 RGB array."""
        t = grid / self.max_iter  # 0..1
        # Smooth cosine palette; tweak the phase offsets to change the look
        r = 0.5 + 0.5 * np.cos(2 * np.pi * (3.0 * t + 0.00))
        g = 0.5 + 0.5 * np.cos(2 * np.pi * (3.0 * t + 0.33))
        b = 0.5 + 0.5 * np.cos(2 * np.pi * (3.0 * t + 0.67))
        rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
        # Points that never blew up (hit max_iter) are in the set -> black
        rgb[grid >= self.max_iter] = 0
        return rgb

    def set_grid(self, grid: jnp.ndarray) -> None:
        """Pass in a (H, W) JAX array of iteration counts."""
        grid_np = np.asarray(grid)  # one device->host transfer
        self._rgb = self._colorize(grid_np)
        self.refresh()

    # ---- rendering -----------------------------------------------------
    def render_line(self, y: int) -> Strip:
        width = self.size.width
        top_row, bot_row = 2 * y, 2 * y + 1
        h, w, _ = self._rgb.shape

        def pixel(row: int, col: int) -> Color:
            if row < h and col < w:
                r, g, b = self._rgb[row, col]
                return Color.from_rgb(int(r), int(g), int(b))
            return Color.from_rgb(0, 0, 0)

        segments = [
            Segment("▀", Style(color=pixel(top_row, x), bgcolor=pixel(bot_row, x)))
            for x in range(width)
        ]
        return Strip(segments)
