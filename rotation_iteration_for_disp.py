from canvas import Point
import rotation_iterator as ri
import jax
import jax.numpy as jnp


def points_for_disp(points, last_point, nx, ny, escape_radius_squared=5,
                    n_iters=300, y_scale=1.0, view=None):
    if len(points) < 2:
        return None  # need at least one segment

    # view = (xmin, xmax, ymin, ymax) in canvas cell coordinates
    if view is None:
        view = (0, nx, 0, ny)
    xmin, xmax, ymin, ymax = view

    # nx, ny are pixel counts; sample pixel centres across the view
    x = xmin + (jnp.arange(nx, dtype=jnp.float32) + 0.5) / nx * (xmax - xmin)
    y = (ymin + (jnp.arange(ny, dtype=jnp.float32) + 0.5) / ny * (ymax - ymin)) * y_scale
    grid = x[None, :] + 1j * y[:, None]


    length = len(points) - 1
    ps_lst = []
    diffs_lst = []
    for i in range(length):
        x0, y0 = last_point
        x1, y1 = points[last_point].prev
        # y_scale stretches y so geometry isn't squashed on 1:2 terminal cells
        ps_lst.append(x0 + 1j * (y0 * y_scale))
        diffs_lst.append((x1 - x0) + 1j * ((y1 - y0) * y_scale))
        last_point = points[last_point].prev
    ps = jnp.array(ps_lst, dtype=jnp.complex64)
    diffs = jnp.array(diffs_lst, dtype=jnp.complex64)
    rs = 0.6 * abs(diffs)
    us = diffs / rs
    rs = jnp.minimum(rs, 10)
    inv_rs = 1 / rs

    inside, escape_steps = ri.evaluate_grid(
        grid, rs, inv_rs, ps, us, escape_radius_squared, n_iters
    )

    # Avoid log(0) (-> inf) and clip so the colour mapping stays in 0..100
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
