import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

# Optional: enable 64-bit precision (complex128) for deep zooms into the fractal boundary
# jax.config.update("jax_enable_x64", True)


def normalize_branch_vectors(
    vectors: list[complex | float | tuple[float, float]] | np.ndarray | jax.Array,
) -> jax.Array:
    """
    Normalize a sequence of non-zero direction vectors into complex unit vectors:
        v_k = w_k / |w_k|

    Accepts either:
      - A 1D sequence of complex (or real) numbers, e.g. [1 + 0j, 3 + 4j, -2 - 2j]
      - A 2D sequence of (x, y) pairs, e.g. [(1, 0), (3, 4), (-2, -2)]

    Normalizing once before iterating keeps |v_k| = 1 so rotating in the hot loop
    never scales the magnitude of `u` and requires zero trig/sqrt calls per step.
    """
    arr = jnp.asarray(vectors)

    # Support 2D (N, 2) real coordinate pairs by converting to x + i*y
    if arr.ndim == 2 and arr.shape[1] == 2 and not jnp.iscomplexobj(arr):
        complex_dtype = jnp.result_type(arr.dtype, 1j)
        vecs = arr[:, 0].astype(complex_dtype) + 1j * arr[:, 1].astype(complex_dtype)
    elif arr.ndim == 1:
        complex_dtype = jnp.result_type(arr.dtype, 1j)
        vecs = arr.astype(complex_dtype)
    else:
        raise ValueError(
            "`branch_vectors` must be a 1D list of complex/real numbers or an (N, 2) array of (x, y) pairs."
        )

    # Compute Euclidean magnitudes and guard against zero-length vectors
    magnitudes = jnp.abs(vecs)
    if bool(jnp.any(magnitudes == 0)):
        raise ValueError("All branch direction vectors must be non-zero.")

    return vecs / magnitudes


@jax.jit
def evaluate_point(
    z: jax.Array,
    rs: jax.Array,
    inv_rs: jax.Array,
    ps: jax.Array,
    unit_vecs: jax.Array,
    escape_radius_squared: float,
    n_iters: int,
) -> tuple[jax.Array, jax.Array]:
    """
    Iterate the composed sequence of local doubling maps on a single point `z`.

    For each step k in the composition:
      1. Shift & scale into local coordinates:  u = (z - p_k) / r_k
      2. Align zero-angle ray v_k to +Re axis:  u_aligned = u * conj(v_k)
      3. Apply doubling map in aligned frame:   u_aligned_sq = u_aligned^2
      4. Rotate back to local u-coordinates:    u_next = v_k * u_aligned_sq
      5. Map back to global z-coordinates:      z_next = r_k * u_next + p_k

    Returns:
      (inside, final_step):
        - inside: True if |u|^2 never exceeded `escape_radius_squared` over all steps.
        - final_step: Total sub-steps completed before escaping (or n_iters * len(ps)).
    """
    n_steps = rs.shape[0]
    total_steps = n_iters * n_steps

    # Ensure carry dtype matches the promoted complex type across all inputs
    z_dtype = jnp.result_type(z, rs, ps, unit_vecs)
    z_init = jnp.asarray(z, dtype=z_dtype)

    def cond_fn(state):
        # Continue looping only while under the step budget and still inside escape radius
        step, _, inside = state
        return (step < total_steps) & inside

    def body_fn(state):
        step, z_curr, _ = state

        # Select parameters for the current step in the composed cycle
        idx = step % n_steps
        r = rs[idx]
        inv_r = inv_rs[idx]
        p = ps[idx]
        v = unit_vecs[idx]  # Pre-normalized unit vector w_idx / |w_idx|

        # 1. Transform z into local u-coordinates centered at p and scaled by r
        u = (z_curr - p) * inv_r

        # 2. Rotate u so the branch direction v sits at angle 0 (+real axis)
        u_aligned = u * jnp.conj(v)

        # 3. Square in the aligned frame (doubles the angle measured from v)
        u_aligned_sq = u_aligned * u_aligned

        # 4. Rotate back to the unaligned local u-frame (|v| = 1 preserves magnitude)
        u_next = v * u_aligned_sq

        # 5. Transform back to global z-coordinates
        z_next = r * u_next + p

        # Escape test on |u_next|^2 == |u_aligned_sq|^2 (avoids sqrt from jnp.abs)
        scaled_mag_sq = (
            u_aligned_sq.real * u_aligned_sq.real
            + u_aligned_sq.imag * u_aligned_sq.imag
        )

        mag_sq = jnp.abs(z_init - z_next)**2

        still_inside = jnp.logical_and(scaled_mag_sq <= escape_radius_squared, mag_sq <= 10*r**2)

        return (step + 1, z_next, still_inside)

    final_step, _, inside = jax.lax.while_loop(
        cond_fn, body_fn, (jnp.int32(0), z_init, jnp.bool_(True))
    )
    return inside, final_step


# Vectorize `evaluate_point` across both rows and columns of a 2D complex grid
evaluate_grid = jax.jit(
    jax.vmap(
        jax.vmap(evaluate_point, in_axes=(0, None, None, None, None, None, None)),
        in_axes=(0, None, None, None, None, None, None),
    )
)


def plot_julia_set(
    rs: list[complex | float],
    ps: list[complex],
    branch_vectors: list[complex | float | tuple[float, float]],
    xlim: tuple[float, float] = (-1.5, 1.5),
    ylim: tuple[float, float] = (-1.5, 1.5),
    resolution: int = 800,
    n_iters: int = 100,
    escape_radius_squared: float = 16.0,
    show_labels: bool = True,
    show_branch_rays: bool = True,
    show_radii: bool = True,
) -> None:
    """
    Compute and plot the bounded orbit set (filled Julia-type set) for a composed
    sequence of local doubling maps with custom zero-angle branch vectors and scale radii.
    """
    if not (len(rs) == len(ps) == len(branch_vectors)):
        raise ValueError(
            "`rs`, `ps`, and `branch_vectors` must all have the same length."
        )

    # Convert inputs to JAX arrays and normalize any non-unit branch vectors
    rs_arr = jnp.asarray(rs)
    ps_arr = jnp.asarray(ps)
    inv_rs_arr = 1.0 / rs_arr  # Precompute reciprocals to avoid division in the loop
    unit_vecs = normalize_branch_vectors(branch_vectors)

    # Construct a 2D grid of starting points z = x + i*y in the complex plane
    x = jnp.linspace(xlim[0], xlim[1], resolution)
    y = jnp.linspace(ylim[0], ylim[1], resolution)
    grid = x[None, :] + 1j * y[:, None]

    # Run the JIT-compiled, vectorized escape-time evaluation
    inside, escape_steps = evaluate_grid(
        grid, rs_arr, inv_rs_arr, ps_arr, unit_vecs, escape_radius_squared, n_iters
    )
    inside_np = np.asarray(inside)
    steps_np = np.asarray(escape_steps, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

    # Layer 1: Exterior points colored by log escape step count
    ax.imshow(
        np.ma.masked_where(inside_np, np.log1p(steps_np)),
        extent=extent,
        origin="lower",
        cmap="magma",
    )

    # Layer 2: Interior bounded points rendered in solid black
    ax.imshow(
        np.ma.masked_where(~inside_np, np.zeros_like(steps_np)),
        extent=extent,
        origin="lower",
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    # Convert centers, scale factors, and unit vectors to NumPy for Matplotlib overlays
    ps_np = np.asarray(ps, dtype=np.complex128)
    rs_np = np.asarray(rs, dtype=np.complex128)
    unit_vecs_np = np.asarray(unit_vecs, dtype=np.complex128)

    # Layer 3: Draw dashed circles of radius |r_k| centered at each p_k.
    # These correspond to the local unit circles |u| = 1 (the invariant radius of u -> u^2).
    if show_radii:
        for idx, (p, r) in enumerate(zip(ps_np, rs_np)):
            circle = mpatches.Circle(
                (p.real, p.imag),
                radius=float(np.abs(r)),
                fill=False,
                edgecolor="cyan",
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                zorder=3,
                # Label only the first circle so the legend stays clean
                label="Scale radius |r|" if idx == 0 else None,
            )
            # Subtle dark outline keeps the dashed ring readable over bright exterior bands
            circle.set_path_effects([pe.withStroke(linewidth=2.4, foreground="black")])
            ax.add_patch(circle)

    # Layer 4: Draw vectors r_k * v_k from each center p_k.
    # Using scale=1.0 in xy-units makes the arrow extend from p_k directly to p_k + r_k * v_k,
    # touching the exact zero-angle point on the dashed circle |z - p_k| = |r_k|.
    if show_branch_rays:
        global_dirs = rs_np * unit_vecs_np
        ax.quiver(
            ps_np.real,
            ps_np.imag,
            global_dirs.real,
            global_dirs.imag,
            color="cyan",
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.004,
            zorder=4,
            label="Branch radius (r·v)",
        )

    # Layer 5: Plot the center points p_k
    ax.scatter(
        ps_np.real,
        ps_np.imag,
        c="cyan",
        edgecolors="black",
        linewidths=1.2,
        s=65,
        zorder=5,
        label="Centers (p)",
    )

    # Layer 6: Optional text labels showing index and radius (e.g., "p0 (|r|=1.00)")
    if show_labels:
        for idx, (p, r) in enumerate(zip(ps_np, rs_np)):
            ax.annotate(
                f"p{idx} (|r|={np.abs(r):.2f})",
                (p.real, p.imag),
                xytext=(6, 6),
                textcoords="offset points",
                color="white",
                fontsize=8.5,
                fontweight="bold",
                zorder=6,
                path_effects=[pe.withStroke(linewidth=2, foreground="black")],
            )

    # Enforce equal aspect ratio so circles render without distortion, and lock bounds
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc="upper right", framealpha=0.85)

    ax.set_title("Bounded Orbit Set with Radii & Branch Vectors (Black = Bounded)")
    ax.set_xlabel("Re(z)")
    ax.set_ylabel("Im(z)")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Centers p_k, scale factors r_k, and arbitrary non-unit branch direction vectors
    ps = [-0.6 + 0.22j, -0.08 - 0.20j]
    rs = [1.0, 0.8]

    # Non-unit complex vectors (automatically normalized to unit vectors internally)
    branch_vectors = [2.5 + 0.0j, -3.0 - 3.0j]

    plot_julia_set(
        rs=rs,
        ps=ps,
        branch_vectors=branch_vectors,
        xlim=(-1.5, 1.5),
        ylim=(-1.5, 1.5),
        resolution=800,
        n_iters=100,
        escape_radius_squared=16.0,
        show_labels=True,
        show_branch_rays=True,
        show_radii=True,
    )
