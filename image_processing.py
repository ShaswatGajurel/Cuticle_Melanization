"""
Image processing backend for cuticle melanization analysis.
All functions operate on numpy arrays and are GUI-independent.
"""

import cv2
import numpy as np 
from skimage.morphology import skeletonize
from scipy.interpolate import splprep, splev
from collections import deque
from pathlib import Path
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


# ── Loading ───────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


# ── Segmentation ──────────────────────────────────────────────────────────────

def detect_mask(image: np.ndarray, threshold: int | None = None) -> np.ndarray:
    """
    Isolate caterpillar from white background.
    threshold=None uses Otsu auto-detection; 1–254 uses a manual level.
    Returns a binary mask (255 = caterpillar).
    """
    gray = to_grayscale(image)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    if threshold is None:
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return clean


def get_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([])
    return max(contours, key=cv2.contourArea)


# ── Centerline ────────────────────────────────────────────────────────────────

def _order_skeleton(skeleton: np.ndarray) -> list:
    """
    Order skeleton pixels from one tip of the caterpillar to the other
    using a two-pass BFS (longest path in the skeleton graph).
    """
    ys, xs = np.where(skeleton)
    if len(xs) < 2:
        return list(zip(ys.tolist(), xs.tolist()))

    point_set = set(zip(ys.tolist(), xs.tolist()))

    adj: dict[tuple, list] = {}
    for y, x in point_set:
        neighbors = [
            (y + dy, x + dx)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if (dy or dx) and (y + dy, x + dx) in point_set
        ]
        adj[(y, x)] = neighbors

    endpoints = [p for p, nb in adj.items() if len(nb) == 1]
    start = endpoints[0] if endpoints else min(point_set, key=lambda p: p[1])

    def bfs_farthest(src):
        dist = {src: 0}
        parent: dict = {src: None}
        q = deque([src])
        farthest = src
        while q:
            cur = q.popleft()
            for nb in adj.get(cur, []):
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    parent[nb] = cur
                    q.append(nb)
                    if dist[nb] > dist[farthest]:
                        farthest = nb
        return farthest, parent

    far1, _ = bfs_farthest(start)
    far2, parent = bfs_farthest(far1)

    path = []
    cur = far2
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path


def compute_centerline(mask: np.ndarray, smooth_factor: float = 1.0) -> np.ndarray:
    """
    Skeletonize the mask and return a smoothed, ordered centerline
    as an (N, 2) array of [row, col] coordinates.
    """
    skeleton = skeletonize((mask > 0).astype(bool))
    ordered = _order_skeleton(skeleton)
    if len(ordered) < 4:
        return np.array([])

    pts = np.array(ordered, dtype=float)

    # Remove consecutive duplicate points before spline fitting
    keep = np.concatenate([[True], np.any(np.diff(pts, axis=0) != 0, axis=1)])
    pts = pts[keep]
    if len(pts) < 4:
        return pts.astype(int)

    try:
        s = len(pts) * smooth_factor * 20
        tck, _ = splprep([pts[:, 1], pts[:, 0]], s=s, k=3)
        n_out = max(100, len(pts) // 2)
        x_s, y_s = splev(np.linspace(0, 1, n_out), tck)
        cl = np.column_stack([
            np.clip(y_s.astype(int), 0, mask.shape[0] - 1),
            np.clip(x_s.astype(int), 0, mask.shape[1] - 1),
        ])
        return cl
    except Exception:
        return pts.astype(int)


# ── Intensity measurements ────────────────────────────────────────────────────

def compute_body_intensity(gray: np.ndarray, mask: np.ndarray) -> float:
    pixels = gray[mask > 0]
    return float(np.mean(pixels)) if len(pixels) else 0.0


def compute_centerline_intensity(
    gray: np.ndarray, centerline: np.ndarray, half_width: int = 3
) -> tuple[float, np.ndarray]:
    """
    Sample mean intensity in a strip of ±half_width pixels around each centerline point.
    Returns (overall_mean, per_point_array).
    """
    if len(centerline) == 0:
        return 0.0, np.array([])

    h, w = gray.shape
    vals = []
    for y, x in centerline:
        strip = gray[
            max(0, y - half_width) : min(h, y + half_width + 1),
            max(0, x - half_width) : min(w, x + half_width + 1),
        ]
        vals.append(float(np.mean(strip)))

    arr = np.array(vals)
    return float(np.mean(arr)), arr


# ── Heatmap rendering ─────────────────────────────────────────────────────────

def create_heatmap(
    gray: np.ndarray, mask: np.ndarray, colormap: str = "inferno"
) -> np.ndarray:
    """
    Apply a colormap to the grayscale image within the mask.
    Darker (more melanized) pixels map to higher colormap values (hotter colors).
    Returns an RGB uint8 image with white background outside the mask.
    """
    masked = gray[mask > 0].astype(float)
    if len(masked) == 0:
        return np.stack([gray, gray, gray], axis=-1)

    vmin, vmax = masked.min(), masked.max()
    norm = np.zeros_like(gray, dtype=float)
    if vmax > vmin:
        # Invert so dark = high value = "hot" color — more melanized = brighter
        norm[mask > 0] = 1.0 - (gray[mask > 0].astype(float) - vmin) / (vmax - vmin)

    try:
        cmap = matplotlib.colormaps[colormap]
    except (AttributeError, KeyError):
        cmap = plt.get_cmap(colormap)

    colored = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    bg = np.full_like(colored, 255)
    m3 = np.stack([mask > 0] * 3, axis=-1)
    return np.where(m3, colored, bg)


def draw_overlay(
    rgb_image: np.ndarray,
    contour: np.ndarray,
    centerline: np.ndarray,
) -> np.ndarray:
    """Draw contour (green) and centerline (yellow) on an RGB image."""
    out = rgb_image.copy()
    if len(contour) > 0:
        cv2.drawContours(out, [contour], -1, (0, 220, 0), 2)
    if len(centerline) > 0:
        pts = centerline.astype(np.int32)
        for i in range(len(pts) - 1):
            cv2.line(out, (pts[i][1], pts[i][0]), (pts[i + 1][1], pts[i + 1][0]),
                     (255, 230, 0), 2)
    return out


# ── Export ────────────────────────────────────────────────────────────────────

def export_single(
    output_dir: str,
    stem: str,
    body_intensity: float,
    cl_intensity: float,
    cl_profile: np.ndarray,
    overlay_rgb: np.ndarray,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{
        "image": stem,
        "whole_body_mean_intensity": round(body_intensity, 4),
        "dorsal_centerline_mean_intensity": round(cl_intensity, 4),
    }]).to_csv(out / f"{stem}_summary.csv", index=False)

    if len(cl_profile):
        pd.DataFrame({
            "position_px": range(len(cl_profile)),
            "mean_intensity": cl_profile,
        }).to_csv(out / f"{stem}_centerline_profile.csv", index=False)

    cv2.imwrite(
        str(out / f"{stem}_heatmap.png"),
        cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
    )
    return out


# ── Batch ─────────────────────────────────────────────────────────────────────

def batch_process(
    image_paths: list,
    threshold: int | None = None,
    colormap: str = "inferno",
    progress_callback=None,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    Process a list of image paths. Calls progress_callback(i, total, name) if provided.
    Returns (summary_DataFrame, average_heatmap_RGB_or_None).
    """
    results = []
    heatmaps = []

    for i, path in enumerate(image_paths):
        name = Path(path).name
        if progress_callback:
            progress_callback(i, len(image_paths), name)
        try:
            img = load_image(str(path))
            gray = to_grayscale(img)
            mask = detect_mask(img, threshold)
            cl = compute_centerline(mask)
            body_int = compute_body_intensity(gray, mask)
            cl_int, cl_profile = compute_centerline_intensity(gray, cl)
            heatmap = create_heatmap(gray, mask, colormap)
            overlay = draw_overlay(heatmap, get_contour(mask), cl)

            export_single(
                str(Path(path).parent / "melanization_results"),
                Path(path).stem, body_int, cl_int, cl_profile, overlay,
            )

            results.append({
                "image": name,
                "whole_body_mean_intensity": round(body_int, 4),
                "dorsal_centerline_mean_intensity": round(cl_int, 4),
            })
            heatmaps.append(heatmap.astype(float))

        except Exception as e:
            results.append({"image": name, "error": str(e)})

    avg = None
    if heatmaps:
        target = heatmaps[0].shape[:2]
        resized = [
            cv2.resize(h.astype(np.uint8), (target[1], target[0]))
            if h.shape[:2] != target else h.astype(np.uint8)
            for h in heatmaps
        ]
        avg = np.mean(resized, axis=0).astype(np.uint8)

    return pd.DataFrame(results), avg
