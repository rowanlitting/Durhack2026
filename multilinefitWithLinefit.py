import math
import random

import matplotlib.pyplot as plt


def fit_line(points):
    """points: list of (x, y) in the order they were spawned."""
    n = len(points)
    if n < 2:
        raise ValueError("need at least 2 points")

    mx = sum(p[0] for p in points) / n
    my = sum(p[1] for p in points) / n

    sxx = sum((x - mx) ** 2 for x, _ in points)
    syy = sum((y - my) ** 2 for _, y in points)
    sxy = sum((x - mx) * (y - my) for x, y in points)

    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
    dx, dy = math.cos(theta), math.sin(theta)

    fx = points[-1][0] - points[0][0]
    fy = points[-1][1] - points[0][1]

    if dx * fx + dy * fy < 0:
        dx, dy = -dx, -dy

    angle = math.degrees(math.atan2(dy, dx)) % 360
    length = math.hypot(fx, fy)

    return {
        "centroid": (mx, my),
        "direction": (dx, dy),
        "vector": (dx * length, dy * length),
        "angle_deg": angle,
        "length": length,
    }


def make_test_points(n=15, angle_deg=35, length=20, noise=1.2, start=(2, 3)):
    """Random points scattered around a line, in spawn order."""
    a = math.radians(angle_deg)
    pts = []
    for i in range(n):
        t = length * i / (n - 1)
        x = start[0] + t * math.cos(a) + random.gauss(0, noise)
        y = start[1] + t * math.sin(a) + random.gauss(0, noise)
        pts.append((x, y))
    return pts



def dist_to_line(p, a, b):
    """Perpendicular distance from point p to the infinite line through a and b."""
    (x, y), (x1, y1), (x2, y2) = p, a, b
    num = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1))
    den = math.hypot(x2 - x1, y2 - y1)
    return num / den if den else math.hypot(x - x1, y - y1)


def ransac_line(points, threshold, iters=200):
    """Return indices of the largest set of points that agree on one line."""
    best = []
    for _ in range(iters):
        i, j = random.sample(range(len(points)), 2)
        inliers = [k for k, p in enumerate(points)
                   if dist_to_line(p, points[i], points[j]) <= threshold]
        if len(inliers) > len(best):
            best = inliers

    # refine: refit on the inliers, then re-collect everything near that fit
    for _ in range(2):
        res = fit_line([points[k] for k in best])
        cx, cy = res["centroid"]
        dx, dy = res["direction"]
        a, b = (cx, cy), (cx + dx, cy + dy)
        best = [k for k, p in enumerate(points) if dist_to_line(p, a, b) <= threshold]
    return best


def find_lines(points, threshold=1.5, min_points=5):
    """
    Repeatedly find the best line, remove its points, and repeat.
    Stops when no remaining line has at least min_points supporting it.
    Returns (list of line results, list of outlier points).
    """
    remaining = list(range(len(points)))      # indices, so spawn order is kept
    lines = []

    while len(remaining) >= min_points:
        subset = [points[k] for k in remaining]
        local = ransac_line(subset, threshold)
        if len(local) < min_points:
            break
        chosen = [remaining[k] for k in local]            # back to original indices
        res = fit_line([points[k] for k in chosen])       # chosen is in spawn order
        res["points"] = [points[k] for k in chosen]
        lines.append(res)
        chosen_set = set(chosen)
        remaining = [k for k in remaining if k not in chosen_set]

    outliers = [points[k] for k in remaining]
    return lines, outliers


def plot(lines, outliers):
    fig, ax = plt.subplots(figsize=(8, 6))
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]

    allpts = outliers + [p for r in lines for p in r["points"]]
    span = max(max(p[0] for p in allpts) - min(p[0] for p in allpts),
               max(p[1] for p in allpts) - min(p[1] for p in allpts))

    for n, r in enumerate(lines):
        c = colours[n % len(colours)]
        xs, ys = zip(*r["points"])
        cx, cy = r["centroid"]
        dx, dy = r["direction"]
        ax.scatter(xs, ys, color=c, zorder=3)
        ax.plot([cx - dx * span, cx + dx * span], [cy - dy * span, cy + dy * span],
                "--", color=c, alpha=0.7,
                label=f"line {n + 1}: {r['angle_deg']:.1f}°, length {r['length']:.1f}")
        ax.annotate("", xy=r["points"][-1], xytext=r["points"][0],
                    arrowprops=dict(arrowstyle="->", color=c, lw=2), zorder=4)

    if outliers:
        ox, oy = zip(*outliers)
        ax.scatter(ox, oy, color="grey", marker="x", s=60, zorder=3, label="outliers")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    random.seed()
    pts = (make_test_points(n=15, angle_deg=35, length=20, start=(0, 0))
           + make_test_points(n=12, angle_deg=-40, length=18, start=(2, 25))
           + [(random.uniform(0, 25), random.uniform(0, 25)) for _ in range(4)])

    lines, outliers = find_lines(pts, threshold=1.8, min_points=6)
    print(f"{len(lines)} lines found, {len(outliers)} outliers")
    for n, r in enumerate(lines):
        print(f"line {n + 1}: {len(r['points'])} points, "
              f"angle {r['angle_deg']:.1f}°, length {r['length']:.2f}, "
              f"vector ({r['vector'][0]:.2f}, {r['vector'][1]:.2f})")
    plot(lines, outliers)
