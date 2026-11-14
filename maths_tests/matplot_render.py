import matplotlib.pyplot as plt
import numpy as np

def mandelbrot(z, c):
    return z^2 + c


class fractal_render:
    def __init__(self, iteration_func, escape_radius, n):
        self.iteration_func = iteration_func # function that we iterate
        self.escape_radius = escape_radius # max distance a point can get from zero before,
        # we conclude it has escaped
        self.n_iterations = n




import jax
import jax.numpy as jnp

"""
def mandelbrot_pixel(c, max_iter=500):
    def cond_fun(state):
        z, n = state
        return (n < max_iter) & (z.real**2 + z.imag**2 <= 4.0)

    def body_fun(state):
        z, n = state
        return (z * z + c, n + 1)

    _, iters = jax.lax.while_loop(cond_fun, body_fun, (0j, 0))
    return iters
"""

def iteration_func(z , p, w, c):
    relative_coords = z - p
    weighted_lengths = jnp.abs(relative_coords)*w
    normed_weighted_lens = weighted_lengths / jnp.sum(weighted_lengths)
    angle_factor = 2*normed_weighted_lens
    angles = jnp.angle(relative_coords)
    moduli = jnp.abs(relative_coords)
    new_angles = angles*angle_factor
    new_moduli = moduli**angle_factor
    new_relative_coords = moduli*jnp.exp(1j*new_angles)
    new_x = jnp.exp(jnp.sum(jnp.log(new_relative_coords + p))) + c
    #new_x = jnp.mean(new_relative_coords + p) + c
    return new_x

p = jnp.array([0.6 + 1.7j, -0.1 + 0.8j, 1.4/(0.6 + 1.7j -0.1 + 0.8j)])

def mandelbrot_pixel(z_0, max_iter=300, escape_radius_squared = 4): # normal 500, 5
    #c = 1 -1.2j
    c = 0j
    def cond_fun(state):
        z, n = state
        return (n < max_iter) & (z.real**2 + z.imag**2 <= escape_radius_squared)

    def body_fun(state):
        z, n = state
        p = jnp.array([0.6 + 1.7j, -0.1 + 0.8j, 1.8/(0.6 + 1.7j -0.1 + 0.8j)])
        w = jnp.array([1.0, 1.0, 1.0])#[2.0, 3.0])
        return (iteration_func(z, p, w ,c), n + 1)

    _, iters = jax.lax.while_loop(cond_fun, body_fun, (z_0, 0))
    return iters

# Automatically vectorize across a 2D grid and JIT-compile
mandelbrot_grid = jax.jit(jax.vmap(jax.vmap(mandelbrot_pixel, in_axes=(0, None)), in_axes=(0, None)))

x = jnp.linspace(-2, 2, 400)
y = jnp.linspace(-2, 2, 400)
C = x[None, :] + 1j * y[:, None]

result = mandelbrot_grid(C, 500)
p_arr = jnp.asarray(p)

plt.imshow(
    result,
    extent=[float(x[0]), float(x[-1]), float(y[0]), float(y[-1])],
    origin="lower",
    aspect="equal",
)
plt.scatter(p_arr.real, p_arr.imag, color="red", s=25, zorder=5)
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")
plt.show()
