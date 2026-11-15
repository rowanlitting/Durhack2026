import numpy as np
from canvas import Point
import rotation_iterator as ri
import jax
import jax.numpy as jnp


def points_for_disp(points, last_point, nx, ny, escape_radius_squared=5,
                    n_iters=300, y_scale=1.0, view=None):
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

    rs = 0.6 * np.abs(diffs)          # your current factor
    us = diffs / rs
    rs = np.minimum(rs, 10)           # cap in canvas units, before rescaling

    # Rebase: view corner -> origin, one pixel -> one unit
    s = nx / (xmax - xmin)            # pixels per canvas unit
    origin = xmin + 1j * ymin * y_scale
    ps = (ps - origin) * s
    rs = rs * s

    # The grid is always plain pixel centres, whatever the zoom
    px = np.arange(nx) + 0.5
    py = np.arange(ny) + 0.5
    grid = px[None, :] + 1j * py[:, None]

    inside, escape_steps = ri.evaluate_grid(
        jnp.asarray(grid, dtype=jnp.complex64),
        jnp.asarray(rs, dtype=jnp.float32),
        jnp.asarray(1 / rs, dtype=jnp.float32),
        jnp.asarray(ps, dtype=jnp.complex64),
        jnp.asarray(us, dtype=jnp.complex64),
        escape_radius_squared, n_iters,
    )

    steps = jnp.maximum(escape_steps, 1)
    return jnp.clip(100 * (1 - jnp.log(steps) / jnp.log(n_iters)), 0, 100)

if __name__ == "__main__":
    last_point = (6, 1)
    points =  {(2, 3): Point(count=1, prev=None), (4, 5): Point(count=1, prev=(2, 3)), (6, 1): Point(count=1, prev=(4, 5))}
    print(points_for_disp(points, last_point, 10, 10))


#grid = [[1.0]]

#inside, escape_steps = ri.evaluate_grid(
 #       grid, points.keys, inv_rs_arr, ps_arr, unit_vecs, escape_radius_squared, n_iters
  #  )

#print(inside, escape_steps)
