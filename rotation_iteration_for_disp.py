from canvas import Point
import rotation_iterator as ri
import jax
import jax.numpy as jnp

def points_for_disp(points, last_point, nx, ny, escape_radius_squared = 4, n_iters = 300):
    x = jnp.linspace(0, nx, nx)
    y = jnp.linspace(0, ny, ny)
    grid = x[None, :] + 1j * y[:, None]

    length = len(points) - 1
    ps_lst = []
    diffs_lst = []
    for i in range(length):
        x0, y0 = last_point
        x1, y1 = points[last_point].prev
        ps_lst.append(x0 + 1j*y0)
        diffs_lst.append((x1 - x0) + 1j*(y1-y0))
        last_point = points[last_point].prev
    ps = jnp.array(ps_lst, dtype=jnp.complex64)
    diffs = jnp.array(diffs_lst, dtype=jnp.complex64)
    rs = abs(diffs)
    us = diffs/rs
    rs = jnp.minimum(rs, 10)
    inv_rs = 1/rs

    # Run the JIT-compiled, vectorized escape-time evaluation
    inside, escape_steps = ri.evaluate_grid(
        grid, rs, inv_rs, ps, us, escape_radius_squared, n_iters
    )
    return inside#escape_steps

if __name__ == "__main__":
    last_point = (6, 1)
    points =  {(2, 3): Point(count=1, prev=None), (4, 5): Point(count=1, prev=(2, 3)), (6, 1): Point(count=1, prev=(4, 5))}
    print(points_for_disp(points, last_point, 10, 10))


#grid = [[1.0]]

#inside, escape_steps = ri.evaluate_grid(
 #       grid, points.keys, inv_rs_arr, ps_arr, unit_vecs, escape_radius_squared, n_iters
  #  )

#print(inside, escape_steps)