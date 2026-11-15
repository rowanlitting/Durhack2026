from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

import rotation_iterator as ri
from canvas import Point


@partial(jax.jit, static_argnames=("chunk",))
def evaluate_chunked(grid, rs, inv_rs, ps, us, er2, n_iters, chunk=32):
    """Evaluate the grid in bands of rows, so empty bands finish early."""
    ny, nx = grid.shape
    pad = (-ny) % chunk
    # Pad with a far-away point that escapes immediately
    g = jnp.pad(grid, ((0, pad), (0, 0)), constant_values=1e6 + 1e6j)
    g = g.reshape(-1, chunk, nx)
    inside, steps = jax.lax.map(
        lambda blk: ri.evaluate_grid(blk, rs, inv_rs, ps, us, er2, n_iters), g
    )
    return inside.reshape(-1, nx)[:ny], steps.reshape(-1, nx)[:ny]


def points_for_disp(points, last_point, nx, ny, escape_radius_squared=5,
                    n_iters=300, y_scale=1.0, view=None, notes=None):
    """
    points:     {(x, y): Point}, where Point.prev is the previous position
    last_point: position of the most recently drawn point
    nx, ny:     image size in pixels
    y_scale:    stretches y to compensate for tall terminal cells
    view:       (xmin, xmax, ymin, ymax) in canvas cell coordinates
    notes:      list of semitones 0..12 (None = rest), repeated along the chain
    """
    if len(points) < 2:
        return None  # need at least one segment

    if view is None:
        view = (0, nx, 0, ny)
    xmin, xmax, ymin, ymax = view

    # Build the chain in float64 canvas units (y stretched by y_scale)
    ps_lst, diffs_lst = [], []
    for _ in range(len(points) - 1):
        x0, y0 = last_point
        x1, y1 = points[last_point].prev
        ps_lst.append(x0 + 1j * (y0 * y_scale))
        diffs_lst.append((x1 - x0) + 1j * ((y1 - y0) * y_scale))
        last_point = points[last_point].prev
    ps = np.array(ps_lst, dtype=np.complex128)
    diffs = np.array(diffs_lst, dtype=np.complex128)

    rs = 0.6 * np.abs(diffs)        # use whatever factor you settled on
    us = diffs / rs

    # Piano: pitch rotates each map's branch direction and scales its radius.
    # Index 0 is the end of the chain (the last point drawn).
    if notes:
        for i in range(len(rs)):
            n = notes[i % len(notes)]
            if n is not None:
                us[i] *= np.exp(2j * np.pi * n / 12)   # pitch -> angle
                rs[i] *= 0.2 + 1.6 * n / 12            # pitch -> size (0.2x..1.8x)

    rs = np.minimum(rs, 10)         # cap in canvas units, before rescaling

    # Rebase: view corner -> origin, one pixel -> one unit.
    # This keeps the grid at plain pixel centres, so deep zooms don't lose
    # precision in float32.
    s = nx / (xmax - xmin)          # pixels per canvas unit
    origin = xmin + 1j * ymin * y_scale
    ps = (ps - origin) * s
    rs = rs * s

    px = np.arange(nx) + 0.5
    py = np.arange(ny) + 0.5
    grid = px[None, :] + 1j * py[:, None]

    inside, escape_steps = evaluate_chunked(
        jnp.asarray(grid, dtype=jnp.complex64),
        jnp.asarray(rs, dtype=jnp.float32),
        jnp.asarray(1 / rs, dtype=jnp.float32),
        jnp.asarray(ps, dtype=jnp.complex64),
        jnp.asarray(us, dtype=jnp.complex64),
        escape_radius_squared, n_iters,
    )

    # Avoid log(0) and keep the colour mapping in 0..100
    steps = jnp.maximum(escape_steps, 1)
    return jnp.clip(100 * (1 - jnp.log(steps) / jnp.log(n_iters)), 0, 100)


if __name__ == "__main__":
    last_point = (6, 1)
    points = {
        (2, 3): Point(count=1, prev=None),
        (4, 5): Point(count=1, prev=(2, 3)),
        (6, 1): Point(count=1, prev=(4, 5)),
    }
    out = points_for_disp(points, last_point, 20, 20, notes=[0, 4, None, 7])
    print(np.asarray(out).shape)


if __name__ == "__main__":
    last_point = (6, 1)
    points =  {(2, 3): Point(count=1, prev=None), (4, 5): Point(count=1, prev=(2, 3)), (6, 1): Point(count=1, prev=(4, 5))}
    print(points_for_disp(points, last_point, 10, 10))


#grid = [[1.0]]

#inside, escape_steps = ri.evaluate_grid(
 #       grid, points.keys, inv_rs_arr, ps_arr, unit_vecs, escape_radius_squared, n_iters
  #  )

#print(inside, escape_steps)
