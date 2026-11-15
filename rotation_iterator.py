import jax
import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

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
      5. Check if the resulting point will be within k's escape radius
         (|u_next|^2 <= escape_radius_squared). If so, apply z = r_k * u_next + p_k;
         otherwise leave z unchanged and move to the next function.

    If a full cycle completes without any function acting on `z`, the point has escaped.

    Returns:
      (inside, final_step):
        - inside: True if at least one function acted in every cycle up to `n_iters`.
        - final_step: Total number of function applications completed before escaping
                      (or n_iters * len(ps) if every function acts in every cycle).
    """
    n_steps = rs.shape[0]

    # Ensure carry dtype matches the promoted complex type across all inputs
    z_dtype = jnp.result_type(z, rs, inv_rs, ps, unit_vecs)
    z_init = jnp.asarray(z, dtype=z_dtype)

    def apply_all_points(
        z_start: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        def step_fn(
            idx: int, carry: tuple[jax.Array, jax.Array, jax.Array]
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            z_curr, any_acted, acted_count = carry

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

            # 5. Act only if the mapped point will be within this function's escape radius
            u_next_mag_sq = u_next.real * u_next.real + u_next.imag * u_next.imag
            can_act = u_next_mag_sq <= escape_radius_squared

            z_cand = r * u_next + p
            z_updated = jnp.where(can_act, z_cand, z_curr)
            any_acted = jnp.logical_or(any_acted, can_act)
            acted_count = acted_count + jnp.where(can_act, jnp.int32(1), jnp.int32(0))

            return (z_updated, any_acted, acted_count)

        return jax.lax.fori_loop(
            0, n_steps, step_fn, (z_start, jnp.bool_(False), jnp.int32(0))
        )

    def cond_fn(state):
        iter_idx, _, inside, _ = state
        return (iter_idx < n_iters) & inside

    def body_fn(state):
        iter_idx, z_curr, _, total_steps = state

        # Apply the sequence of maps across `ps`, skipping functions whose
        # image would fall outside their local escape radius
        z_next, any_acted, acted_in_cycle = apply_all_points(z_curr)

        return (
            iter_idx + 1,
            z_next,
            any_acted,
            total_steps + acted_in_cycle,
        )

    _, _, inside, final_step = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (jnp.int32(0), z_init, jnp.bool_(True), jnp.int32(0)),
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



    escape_steps = jnp.maximum(escape_steps,2)



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

    steps_j = jnp.asarray(steps_np, dtype=jnp.float32)
    inside_j = jnp.asarray(inside_np, dtype=bool)

    norm_steps = jnp.log1p(steps_j) / (jnp.log1p(steps_j).max() + 1e-8)
    gray = jnp.where(inside_j, 1.0, norm_steps * 0.85)[::-1]  # flip to top-down

    # --- JAX Sharpened UTF-8 Block & Edge-Directed Renderer (paste before plt.show()) ---
    steps_j = jnp.asarray(steps_np, dtype=jnp.float32)
    inside_j = jnp.asarray(inside_np, dtype=bool)

    norm_steps = jnp.log1p(steps_j) / (jnp.log1p(steps_j).max() + 1e-8)
    gray = jnp.where(inside_j, 1.0, norm_steps * 0.85)[::-1]  # flip to top-down

    # --- JAX Sharpened UTF-8 Block & Edge-Directed Renderer (paste before plt.show()) ---
    steps_j = jnp.asarray(steps_np, dtype=jnp.float32)
    inside_j = jnp.asarray(inside_np, dtype=bool)

    s_shift = jnp.maximum(0.0, steps_j - steps_j.min())
    norm_steps = jnp.log1p(s_shift) / (jnp.log1p(s_shift).max() + 1e-8)
    gray = jnp.where(inside_j, 1.0, norm_steps * 0.85)[::-1]  # flip to top-down

    # 1. 3x3 Gaussian unsharp mask (masked so empty background stays strictly 0)
    p = jnp.pad(gray, 1, mode="edge")
    blur = (
        p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:]
        + 2 * p[1:-1, :-2] + 4 * p[1:-1, 1:-1] + 2 * p[1:-1, 2:]
        + p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
    ) / 16.0
    sharp = jnp.where(gray > 0.0, jnp.clip(gray + 1.8 * (gray - blur), 0.0, 1.0), 0.0)

    # 2. Downsample to ~1:2 terminal grid + sub-cell halves for crisp bulb edges
    ascii_w = 120
    aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    ascii_h = max(1, int(ascii_w * aspect * 0.5))
    bh, bw = max(2, sharp.shape[0] // ascii_h), max(2, sharp.shape[1] // ascii_w)

    cells = sharp[: ascii_h * bh, : ascii_w * bw].reshape(ascii_h, bh, ascii_w, bw)
    has_col = cells.max(axis=(1, 3)) > 0.05  # raise slightly (e.g. 0.15) to clip faint outer glow to blank
    lum = 0.65 * cells.mean(axis=(1, 3)) + 0.35 * cells.max(axis=(1, 3))

    top, bot = cells[:, : bh // 2].mean(axis=(1, 3)), cells[:, bh // 2 :].mean(axis=(1, 3))
    left, right = cells[:, :, :, : bw // 2].mean(axis=(1, 3)), cells[:, :, :, bw // 2 :].mean(axis=(1, 3))

    # 3. Sobel edge angles (restricted to cells with signal so background stays blank)
    lp = jnp.pad(lum, 1, mode="edge")
    gx = (lp[:-2, 2:] + 2 * lp[1:-1, 2:] + lp[2:, 2:]) - (lp[:-2, :-2] + 2 * lp[1:-1, :-2] + lp[2:, :-2])
    gy = (lp[2:, :-2] + 2 * lp[2:, 1:-1] + lp[2:, 2:]) - (lp[:-2, :-2] + lp[:-2, 1:-1] + lp[:-2, 2:])
    mag = jnp.hypot(gx, gy)
    ang = jnp.degrees(jnp.arctan2(gy, gx)) % 180.0

    # Base tonal ramp: 0:' ' (no col), 1:'·', 2:'•', 3:'░', 4:'▒', 5:'▓', 6:'█'
    idx = jnp.where(has_col, jnp.clip(1 + (lum * 5.99).astype(jnp.int32), 1, 6), 0)
    edges = has_col & (mag > 0.35)

    horiz = edges & (ang >= 67.5) & (ang < 112.5)
    vert = edges & ((ang < 22.5) | (ang >= 157.5))
    diag_up = edges & (ang >= 22.5) & (ang < 67.5)
    diag_dn = edges & (ang >= 112.5) & (ang < 157.5)

    idx = jnp.where(vert, 7, idx)      # '│'
    idx = jnp.where(diag_up, 8, idx)   # '╱'
    idx = jnp.where(horiz, 9, idx)     # '─'
    idx = jnp.where(diag_dn, 10, idx)  # '╲'
    idx = jnp.where(edges & (jnp.abs(gx) > 0.3) & (jnp.abs(gy) > 0.3) & (lum < 0.65), 11, idx)  # '╳'

    idx = jnp.where(horiz & (jnp.abs(top - bot) > 0.35), jnp.where(top > bot, 12, 13), idx)    # '▀' / '▄'
    idx = jnp.where(vert & (jnp.abs(left - right) > 0.35), jnp.where(left > right, 14, 15), idx)  # '▌' / '▐'

    glyphs = np.array(list(" ·•░▒▓█│╱─╲╳▀▄▌▐"))
    grid = glyphs[np.asarray(idx)]
    print("\n".join("".join(row) for row in grid))
    plt.show()



if __name__ == "__main__":
    # Centers p_k, scale factors r_k, and arbitrary non-unit branch direction vectors
    #ps = [-0.6 + 0.22j, -0.08 - 0.20j]
    #rs = [1.0, 0.8]

    """ps = [63.+23.j, 49.+22.j, 33.+21.j, 25.+16.j, 26.+12.j]
    rs = [30.842802,  30.237465,  40.320778,   14.9477267,  7.2993155]
    us = [-0.8312156 -0.05937254j, -0.8317104 -0.0519819j,  -0.7066652 -0.44166577j, 0.20211302-0.80845207j, -0.13699915-0.8219949j]
    nx = 120
    ny = 60"""


    ps = jnp.array([
    127.+32.j, 121.+33.j, 110.+33.j, 102.+33.j, 89.+33.j, 86.+33.j, 74.+32.j,
     69.+32.j,  58.+31.j,  57.+31.j,  50.+32.j, 42.+32.j, 32.+32.j, 20.+30.j,
     18.+27.j,  17.+22.j,  17.+16.j,  16.+11.j, 16.+8.j,
    ])
    key = jax.random.key(0)
    ps = jax.random.permutation(key, ps)

    rs = jnp.array([
        7.2993155, 10.,        9.6,       10.,        14.6000001, 10.,
        12,        10.,        12,       8.485282,  9.6,       10.,
    10.,        4.3266616,  6.1188235, 7.2000003, 6.1188235, 12.000001,
        14.8,
    ])
    rs = rs*0.7

    us = [
        -0.8219949+0.13699915j,  -0.8333333+0.j,          -0.8333333+0.j,
        -0.8333333+0.j,          -0.8333333+0.j,          -0.83045477-0.06920456j,
        -0.8333333+0.j,          -0.829911-0.07544645j,   -0.8333333+0.j,
        -0.82495785+0.11785112j, -0.8333333+0.j,          -0.8333333+0.j,
        -0.8219949-0.13699915j,  -0.46225014-0.69337523j, -0.16343011-0.81715053j,
        0.-0.8333333j,          -0.16343011-0.81715053j,  0.-0.8333333j,
        0.-0.8333333j,
    ]

    nx = 156
    ny = 82


    # Non-unit complex vectors (automatically normalized to unit vectors internally)
    branch_vectors = us

    plot_julia_set(
        rs=rs,
        ps=ps,
        branch_vectors=branch_vectors,
        xlim=(0, nx),
        ylim=(0, ny),
        resolution=800,
        n_iters=100,
        escape_radius_squared=4,
        show_labels=True,
        show_branch_rays=True,
        show_radii=True,
    )
