"""Series-level post-processing for horizontal (and tilted horizontal) sections.

Coordinates are QuickNII ``ABA_Mouse_CCFv3_2017_25um`` voxels: x = mediolateral
(0-455, towards the animal's right), y = anteroposterior (0-527, towards anterior),
z = dorsoventral (0-319, towards superior). In a horizontal series, sections are spaced
along z.

Tilt is described by two slope angles: ``pitch`` (dz/dy = tan(pitch), positive when the
anterior end is higher) and ``roll`` (dz/dx = tan(roll), positive when the animal's
right side is higher).

Sections near the dorsal and ventral ends of a series have few landmarks. Every
series-level step therefore weights sections by how reliable the network is at their
depth (and by how much tissue they contain), then applies the result to all sections,
so end sections inherit tilt, spacing and scale from the well-structured middle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

COLUMNS = ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]
MIDLINE_X = 227.5
CENTRE_Y = 264.0
VOXEL_UM = 25.0
BRAIN_Z_RANGE = (18.0, 312.0)
REFERENCE_TISSUE_AREA = 40000.0  # voxels^2 (25 mm^2); larger sections get full weight
HEMISPHERE_ROLL_WEIGHT = 0.5  # one hemisphere constrains roll less well


# ---------------------------------------------------------------------------
# Single-section geometry
# ---------------------------------------------------------------------------
def _split(a):
    a = np.array(a, dtype=np.float64)  # copy: pandas rows can be read-only views
    return a[0:3], a[3:6], a[6:9]


def plane_normal(a) -> np.ndarray:
    """Unit normal of the section plane, oriented upwards (n_z > 0)."""
    _, U, V = _split(a)
    n = np.cross(U, V)
    n /= np.linalg.norm(n)
    return -n if n[2] < 0 else n


def tilt(a):
    """(pitch, roll) slope angles in degrees."""
    n = plane_normal(a)
    return np.degrees(np.arctan2(-n[1], n[2])), np.degrees(np.arctan2(-n[0], n[2]))


def centre_depth(a) -> float:
    """z where the plane crosses the brain's central dorsoventral axis (x=227.5, y=264)."""
    O, _, _ = _split(a)
    n = plane_normal(a)
    return O[2] - (n[0] * (MIDLINE_X - O[0]) + n[1] * (CENTRE_Y - O[1])) / n[2]


def set_tilt(a, pitch: float, roll: float) -> np.ndarray:
    """Rotate a section to the given tilt about the point where it crosses the central axis.

    The section's centre depth is unchanged and the image frame moves rigidly.
    """
    O, U, V = _split(a)
    n_old = plane_normal(a)
    n_new = np.array([-np.tan(np.radians(roll)), -np.tan(np.radians(pitch)), 1.0])
    n_new /= np.linalg.norm(n_new)
    axis = np.cross(n_old, n_new)
    s = np.linalg.norm(axis)
    if s < 1e-12:
        return np.asarray(a, dtype=np.float64).copy()
    R = Rotation.from_rotvec(axis / s * np.arctan2(s, n_old @ n_new))
    pivot = np.array([MIDLINE_X, CENTRE_Y, centre_depth(a)])
    return np.concatenate([pivot + R.apply(O - pivot), R.apply(U), R.apply(V)])


def set_depth(a, z: float) -> np.ndarray:
    """Move a section along its normal so that its centre depth becomes ``z``."""
    O, U, V = _split(a)
    n = plane_normal(a)
    return np.concatenate([O + n * (z - centre_depth(a)) * n[2], U, V])


def mirror(a) -> np.ndarray:
    """Reflect a section through the atlas midline (x -> 455 - x)."""
    O, U, V = _split(a)
    m = np.array([-1.0, 1.0, 1.0])
    return np.concatenate([[2 * MIDLINE_X - O[0], O[1], O[2]], U * m, V * m])


def is_ventral_view(a) -> bool:
    """True if the anchoring maps the image as seen from below (section mounted ventral side up).

    In a dorsal view U x V points ventrally (towards -z).
    """
    _, U, V = _split(a)
    return np.cross(U, V)[2] > 0


def canonical(a) -> np.ndarray:
    """Dorsal-view equivalent of an anchoring: mirrored through the midline if needed."""
    return mirror(a) if is_ventral_view(a) else np.concatenate(_split(a))


def image_rotation(a) -> float:
    """Direction of the midline (pointing anterior) in the image, in degrees clockwise
    from straight up.

    0 means the midline is vertical with anterior at the top, the orientation the network
    is trained for (+-15 degrees). The midline, where the section crosses the sagittal
    midplane, is used rather than the projected anteroposterior axis: on a plane tilted
    about both axes the two differ (by ~10 degrees at 25/25 degree tilt). Assumes square
    pixels and perpendicular U and V, as QuickNII anchorings are.
    """
    _, U, V = _split(a)
    n = plane_normal(a)
    midline = np.array([0.0, n[2], -n[1]])  # n x (1, 0, 0); points anterior as n_z > 0
    ix = midline @ U / np.linalg.norm(U)
    iy = midline @ V / np.linalg.norm(V)  # image y points down
    return float(np.degrees(np.arctan2(ix, -iy)))


def rotate_image_and_anchoring(image: np.ndarray, a, angle: float, fill=None):
    """Rotate an image counterclockwise by ``angle`` degrees (canvas expanded) and return
    the rotated image with its exactly transformed anchoring.

    ``rotate_image_and_anchoring(img, a, image_rotation(a))`` puts anterior at the top.
    """
    from PIL import Image

    O, U, V = _split(a)
    h, w = image.shape[:2]
    px = np.sqrt(np.linalg.norm(U) / w * np.linalg.norm(V) / h)  # voxels per pixel
    if fill is None:
        border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
        fill = int(np.median(border))
    rotated = np.asarray(Image.fromarray(image).rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=fill))
    h2, w2 = rotated.shape[:2]
    phi = np.radians(angle)
    eu, ev = U / np.linalg.norm(U), V / np.linalg.norm(V)
    U2 = w2 * px * (np.cos(phi) * eu + np.sin(phi) * ev)
    V2 = h2 * px * (-np.sin(phi) * eu + np.cos(phi) * ev)
    c = O + U / 2 + V / 2
    return rotated, np.concatenate([c - U2 / 2 - V2 / 2, U2, V2])


def crop_to_tissue(image: np.ndarray, a, margin: float = 0.08, return_box: bool = False):
    """Crop an image to its tissue bounding box plus ``margin`` (fraction of the box) on
    each side, returning the cropped image and its exactly transformed anchoring.

    Keeps framing close to what the network was trained on (box + 0-25% margins), e.g.
    after :func:`rotate_image_and_anchoring` has expanded the canvas. With ``return_box``
    also returns the crop box (row0, row1, col0, col1).
    """
    mask = tissue_mask(image if image.ndim == 2 else image.mean(axis=2))
    h, w = mask.shape
    if not mask.any():
        out = (image, np.concatenate(_split(a)))
        return out + ((0, h, 0, w),) if return_box else out
    rows, cols = np.nonzero(mask)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    dr, dc = int(round(margin * (r1 - r0))), int(round(margin * (c1 - c0)))
    r0, r1, c0, c1 = max(0, r0 - dr), min(h, r1 + dr), max(0, c0 - dc), min(w, c1 + dc)
    O, U, V = _split(a)
    a2 = np.concatenate([O + c0 / w * U + r0 / h * V, (c1 - c0) / w * U, (r1 - r0) / h * V])
    if return_box:
        return image[r0:r1, c0:c1], a2, (r0, r1, c0, c1)
    return image[r0:r1, c0:c1], a2


# ---------------------------------------------------------------------------
# Fixed physical field of view ("canvas") inputs
# ---------------------------------------------------------------------------
def to_canvas(image: np.ndarray, pixel_size_um: float, fov_um: float, out_px: int = 299, fill=None):
    """Square canvas of side ``fov_um`` centred on the tissue, area-averaged to ``out_px``.

    The network then sees true physical size. Returns (canvas uint8, region) where region
    = (x0, y0, side) is the canvas in original pixel coordinates (it may extend past the
    image, which is padded with the background level).
    """
    from PIL import Image

    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img / (np.percentile(img, 99.5) + 1e-9) * 230, 0, 255).astype(np.uint8)
    h, w = img.shape
    step = max(1, int(np.ceil(max(h, w) / 512)))
    mask = tissue_mask(img[::step, ::step])
    if mask.any():
        rows, cols = np.nonzero(mask)
        cy, cx = (rows.mean() + 0.5) * step, (cols.mean() + 0.5) * step
    else:
        cy, cx = h / 2, w / 2
    side = int(round(fov_um / pixel_size_um))
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    if fill is None:
        fill = int(np.median(np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])))
    canvas = Image.new("L", (side, side), fill)
    canvas.paste(Image.fromarray(img), (-x0, -y0))
    return np.asarray(canvas.resize((out_px, out_px), Image.BOX)), (x0, y0, side)


def canvas_to_image_anchoring(b, region, width: int, height: int) -> np.ndarray:
    """Anchoring of the original (width x height) image from the anchoring of its canvas."""
    x0, y0, side = region
    O, U, V = _split(b)
    s0, s1 = -x0 / side, (width - x0) / side
    t0, t1 = -y0 / side, (height - y0) / side
    return np.concatenate([O + s0 * U + t0 * V, (s1 - s0) * U, (t1 - t0) * V])


def prepare_canvases(paths, pixel_size_um, fov_um: float, folder: str) -> dict:
    """Write canvases of ``paths`` into ``folder`` under the same file names.

    ``pixel_size_um`` is a number or {file name: number}. Returns {file name: (region,
    (width, height))} for :func:`canvas_to_image_anchoring`.
    """
    import os

    from PIL import Image

    out = {}
    for p in paths:
        name = os.path.basename(p)
        px = pixel_size_um[name] if isinstance(pixel_size_um, dict) else pixel_size_um
        with Image.open(p) as im:
            grey = np.asarray(im.convert("L"))
        canvas, region = to_canvas(grey, px, fov_um)
        Image.fromarray(canvas).save(os.path.join(folder, name), format="PNG")
        out[name] = (region, (grey.shape[1], grey.shape[0]))
    return out


def rescale(a, length_u: float, length_v: float) -> np.ndarray:
    """Set |U| and |V| keeping the frame centre and directions."""
    O, U, V = _split(a)
    c = O + U / 2 + V / 2
    U2 = U / np.linalg.norm(U) * length_u
    V2 = V / np.linalg.norm(V) * length_v
    return np.concatenate([c - U2 / 2 - V2 / 2, U2, V2])


# ---------------------------------------------------------------------------
# Per-section information from the images
# ---------------------------------------------------------------------------
def tissue_mask(image: np.ndarray) -> np.ndarray:
    """Foreground mask of a section image by Otsu thresholding.

    The background is taken to be whichever class dominates the image border, so this
    works for fluorescence (bright tissue) and brightfield (dark tissue) images.
    """
    from skimage.filters import threshold_otsu

    img = image.astype(np.float64)
    if img.max() == img.min():
        return np.zeros(img.shape, bool)
    fg = img > threshold_otsu(img)
    border = np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]])
    return ~fg if border.mean() > 0.5 else fg


def describe_section(image: np.ndarray, a, both_threshold: float = 0.25):
    """(hemisphere, tissue_fraction, left_fraction) of one image given its anchoring.

    The anchoring maps tissue pixels to atlas x. ``left_fraction`` is the fraction of the
    tissue left of the midline; hemisphere is "both" when each side holds at least
    ``both_threshold`` of it, otherwise "left" or "right". On synthetic validation data a
    threshold of 0.25 calls 2% of one-hemisphere sections "both"; "both" sections called
    one-sided are mostly strongly rolled or end sections that really cross one side.
    """
    mask = tissue_mask(image)
    frac = float(mask.mean())
    if not mask.any():
        return "both", frac, 0.5
    h, w = mask.shape
    O, U, V = _split(a)
    rows, cols = np.nonzero(mask)
    x = O[0] + (cols + 0.5) / w * U[0] + (rows + 0.5) / h * V[0]
    left = float(np.mean(x < MIDLINE_X))
    if min(left, 1 - left) >= both_threshold:
        return "both", frac, left
    return ("left" if left > 0.5 else "right"), frac, left


def load_grey(path: str, max_size: int = 256) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L")
        im.thumbnail((max_size, max_size))
        return np.asarray(im)


def add_section_descriptions(df: pd.DataFrame, paths) -> pd.DataFrame:
    """Add ``hemisphere``, ``tissue_fraction`` and ``left_fraction`` columns (paths aligned with df rows)."""
    rows = [describe_section(load_grey(path), row[COLUMNS].values) for path, (_, row) in zip(paths, df.iterrows())]
    df = df.copy()
    df["hemisphere"] = [r[0] for r in rows]
    df["tissue_fraction"] = [r[1] for r in rows]
    df["left_fraction"] = [r[2] for r in rows]
    # trained for +-15 degrees; larger values suggest the image should be rotated upright
    df["image_rotation"] = [image_rotation(a) for a in df[COLUMNS].values]
    return df


# ---------------------------------------------------------------------------
# Reliability weights
# ---------------------------------------------------------------------------
def depth_weight(z, curve=None):
    """Reliability of the network as a function of centre depth z.

    ``curve`` is ``{"z": [...], "weight": [...]}`` measured on synthetic validation data
    (``training/evaluate.py``). Without it, a Gaussian over the brain's DV extent is used.
    """
    z = np.asarray(z, dtype=np.float64)
    if curve is not None:
        return np.interp(z, curve["z"], curve["weight"])
    mid = np.mean(BRAIN_Z_RANGE)
    sigma = (BRAIN_Z_RANGE[1] - BRAIN_Z_RANGE[0]) / 5.4
    return np.exp(-0.5 * ((z - mid) / sigma) ** 2)


def section_weights(df: pd.DataFrame, curve=None, z=None):
    """(weights, roll_weights) for every row of a predictions table.

    ``z`` overrides the sections' own predicted depths. Weighting by a section's own
    prediction biases series fits: end sections whose predictions err towards the middle
    of the brain get larger weights. Weights from fitted depths avoid this.
    """
    if z is None:
        z = np.array([centre_depth(a) for a in df[COLUMNS].values])
    w = depth_weight(z, curve)
    if "tissue_fraction" in df:
        area = df["tissue_fraction"].values * np.array(
            [np.linalg.norm(a[3:6]) * np.linalg.norm(a[6:9]) for a in df[COLUMNS].values]
        )
        w = w * np.clip(area / REFERENCE_TISSUE_AREA, 0.05, 1.0)
    if "bad_section" in df:
        w = np.where(df["bad_section"].fillna(False).astype(bool).values, 0.0, w)
    roll_w = w.copy()
    if "hemisphere" in df:
        roll_w = np.where(df["hemisphere"].values != "both", roll_w * HEMISPHERE_ROLL_WEIGHT, roll_w)
    return w, roll_w


# ---------------------------------------------------------------------------
# Robust estimation: catastrophic predictions (e.g. a small fragment of cortex at the end
# of a strongly tilted series) must not pull series-level estimates
# ---------------------------------------------------------------------------
def _weighted_median(values, w):
    order = np.argsort(values)
    cw = np.cumsum(w[order])
    return values[order][np.searchsorted(cw, cw[-1] / 2)]


def tukey_weights(resid, w, min_scale, c=4.685):
    """Tukey biweight weights from residuals; scale is the MAD over weighted sections."""
    good = w > 0
    if good.sum() < 3:
        return np.ones_like(resid)
    r = resid[good]
    scale = max(1.4826 * np.median(np.abs(r - np.median(r))), min_scale)
    u = resid / (c * scale)
    return np.where(np.abs(u) < 1, (1 - u**2) ** 2, 0.0)


def robust_mean(values, w, min_scale, iterations=10):
    """Reliability-weighted mean with Tukey down-weighting of outliers.

    Returns (mean, robustness weights).
    """
    values = np.asarray(values, dtype=np.float64)
    m = _weighted_median(values, w)
    rw = tukey_weights(values - m, w, min_scale)
    for _ in range(iterations):
        ww = w * rw
        if ww.sum() <= 0:
            break
        m = np.average(values, weights=ww)
        rw = tukey_weights(values - m, w, min_scale)
    return m, rw


# ---------------------------------------------------------------------------
# Series-level steps
# ---------------------------------------------------------------------------
def _apply(df, fn):
    df = df.copy()
    df[COLUMNS] = np.array([fn(a) for a in df[COLUMNS].values])
    return df


def series_roll(rolls, weights, min_scale=0.5):
    """Common roll of a series, allowing for sections mounted ventral side up.

    A section mounted ventral side up is predicted as its mirror image, which has the
    same pitch but the opposite roll. When roll is clearly non-zero but its sign differs
    between sections, each section keeps its own sign with the common magnitude.

    Returns (per-section roll, robustness weights, boolean array of sections whose sign
    differs from the majority when mounting is mixed).
    """
    rolls = np.asarray(rolls, dtype=np.float64)
    signed, rw_signed = robust_mean(rolls, weights, min_scale)
    magnitude, rw_abs = robust_mean(np.abs(rolls), weights, min_scale)
    good = weights > 0
    majority = 1.0 if np.sum(weights * np.sign(rolls)) >= 0 else -1.0
    signs = np.where(rolls == 0, majority, np.sign(rolls))
    disagree = np.sum(weights[good] * (signs[good] != majority)) / np.sum(weights[good])
    resid = np.abs(rolls[good]) - magnitude
    noise = max(1.4826 * np.median(np.abs(resid - np.median(resid))), min_scale)
    if disagree > 0.1 and magnitude > 3 * noise:
        # trust a section's own sign only when its roll is consistent with +-magnitude;
        # otherwise (e.g. a featureless end section) assume the majority mounting
        signs = np.where(rw_abs > 0, signs, majority)
        return signs * magnitude, rw_abs, signs != majority
    return np.full(len(rolls), signed), rw_signed, np.zeros(len(rolls), bool)


def propagate_tilt(df: pd.DataFrame, curve=None, verbose=True):
    """Set every section to the robust, reliability-weighted series pitch and roll.

    With section numbers, weights use depths from the fitted depth line. Sections that
    appear to be mounted the other way up (opposite roll, see :func:`series_roll`) keep
    their sign and are flagged in ``flipped_mounting``.
    """
    z = None
    if "nr" in df and df["nr"].nunique() > 2:
        slope, intercept, _ = fit_depth_line(df, curve=curve)
        z = intercept + slope * df["nr"].values.astype(np.float64)
    w, roll_w = section_weights(df, curve, z=z)
    if w.sum() <= 0:
        raise ValueError("no section has a positive weight; check bad_section labels")
    tilts = np.array([tilt(a) for a in df[COLUMNS].values])
    pitch, rw_pitch = robust_mean(tilts[:, 0], w, min_scale=0.5)
    rolls, rw_roll, flipped = series_roll(tilts[:, 1], roll_w if roll_w.sum() > 0 else w)
    if verbose:
        print(f"pitch (AP tilt): {pitch:.2f} deg\nroll (ML tilt): {np.abs(rolls[0]):.2f} deg" + (
            f" in magnitude; {int(flipped.sum())} section(s) appear mounted the other way up"
            if flipped.any() else f" ({rolls[0]:.2f})"))
    out = df.copy()
    out[COLUMNS] = np.array([set_tilt(a, pitch, r) for a, r in zip(df[COLUMNS].values, rolls)])
    # sections whose own tilt was rejected as an outlier deserve a manual check
    out["outlier_tilt"] = ((rw_pitch == 0) | (rw_roll == 0)) & (w > 0)
    # mirror-image predictions: left and right are swapped relative to the rest of the series
    out["flipped_mounting"] = flipped
    return out


def set_series_tilt(df: pd.DataFrame, pitch: float, roll: float):
    return _apply(df, lambda a: set_tilt(a, pitch, roll))


def _good(df):
    if "bad_section" in df:
        return ~df["bad_section"].fillna(False).astype(bool).values
    return np.ones(len(df), bool)


def _with_predicted_depth(df):
    """Keep each section's own predicted depth (for QC) before series steps change it."""
    if "predicted_depth" not in df:
        df = df.copy()
        df["predicted_depth"] = [centre_depth(a) for a in df[COLUMNS].values]
    return df


def enforce_index_order(df: pd.DataFrame, curve=None):
    """Reassign predicted depths so that they are monotonic in section number.

    Only sections consistent with the series' robust depth line are reordered; sections
    far off it (see :func:`fit_depth_line`) take their depth from the line instead, so a
    single catastrophic prediction cannot shift every section between it and its true
    place in the order.
    """
    if "nr" not in df:
        raise ValueError("No section numbers found; run predict() with section_numbers=True")
    df = _with_predicted_depth(df.sort_values("nr").reset_index(drop=True))
    nr = df["nr"].values.astype(np.float64)
    depths = np.array([centre_depth(a) for a in df[COLUMNS].values])
    good = _good(df)
    if good.sum() >= 3 and len(np.unique(nr[good])) >= 3:
        slope, intercept, rw = fit_depth_line(df, curve=curve)
        far = good & (rw == 0)
        depths[far] = intercept + slope * nr[far]  # now roughly in place
        new = np.sort(depths[good])
        depths[good] = new if slope > 0 else new[::-1]
    out = df.copy()
    out[COLUMNS] = np.array([set_depth(a, z) for a, z in zip(df[COLUMNS].values, depths)])
    out["depths"] = depths
    return out


def fit_depth_line(df: pd.DataFrame, section_thickness_um=None, curve=None, iterations=8):
    """Robust, reliability-weighted line of centre depth against section number.

    Returns (slope, intercept, robustness weights); slope is in voxels per section number.
    Reliability weights come from the fitted line rather than from each section's own
    prediction (see :func:`section_weights`); Tukey robustness weights reject sections
    whose predicted depth is far off the line.
    """
    from scipy.stats import theilslopes

    nr = df["nr"].values.astype(np.float64)
    depths = np.array([centre_depth(a) for a in df[COLUMNS].values])
    w, _ = section_weights(df, curve)
    good = w > 0
    if good.sum() < 2 or len(np.unique(nr[good])) < 2:
        raise ValueError(
            "need at least two reliable sections with different numbers; predicted depths "
            "may be outside the brain (are these horizontal sections, anterior at the top?)"
        )
    # robust start: Theil-Sen line through the usable sections
    slope, intercept = theilslopes(depths[good], nr[good])[:2]
    if section_thickness_um is not None:
        slope = np.sign(slope) * abs(section_thickness_um) / VOXEL_UM
        intercept = np.median(depths[good] - slope * nr[good])
    fitted = intercept + slope * nr
    rw = tukey_weights(depths - fitted, w, min_scale=1.0)
    for _ in range(iterations):
        w, _ = section_weights(df, curve, z=fitted)
        ww = w * rw
        if (ww > 0).sum() < 2 or len(np.unique(nr[ww > 0])) < 2:
            break
        if section_thickness_um is None:
            slope, intercept = np.polyfit(nr, depths, 1, w=np.sqrt(ww))
        else:
            intercept = np.average(depths - slope * nr, weights=ww)
        fitted = intercept + slope * nr
        rw = tukey_weights(depths - fitted, w, min_scale=1.0)
    return slope, intercept, rw


def enforce_index_spacing(df: pd.DataFrame, section_thickness_um=None, curve=None, verbose=True):
    """Place every section on a reliability-weighted line of depth against section number.

    End sections get their depth from this line, extended beyond the well-structured
    sections, rather than from their own prediction. ``section_thickness_um`` fixes the
    slope; its sign is taken from the data.
    """
    if "nr" not in df:
        raise ValueError("No section numbers found; run predict() with section_numbers=True")
    df = _with_predicted_depth(df.sort_values("nr").reset_index(drop=True))
    slope, intercept, _ = fit_depth_line(df, section_thickness_um, curve)
    if section_thickness_um is not None:
        if np.sign(section_thickness_um) != np.sign(slope) and verbose:
            print("note: section numbers run the opposite way to the given thickness sign; using the data's direction")
    elif verbose:
        print(f"predicted thickness is {abs(slope) * VOXEL_UM:.1f}µm")
    fitted = intercept + slope * df["nr"].values.astype(np.float64)
    out = df.copy()
    out[COLUMNS] = np.array([set_depth(a, z) for a, z in zip(df[COLUMNS].values, fitted)])
    out["depths"] = fitted
    # QC: how far each section's own prediction was from the series line (voxels), and
    # whether it was rejected as an outlier; such sections deserve a manual check
    out["depth_residual"] = df["predicted_depth"].values - fitted
    w, _ = section_weights(df, curve, z=fitted)
    out["outlier_depth"] = (tukey_weights(out["depth_residual"].values, w, min_scale=1.0) == 0) & _good(df)
    if verbose and out["outlier_depth"].any():
        print(f"{int(out['outlier_depth'].sum())} section(s) far from the series depth line: "
              + ", ".join(out.loc[out["outlier_depth"], "Filenames"].astype(str)))
    return out


def propagate_scale(df: pd.DataFrame, pixel_size_um: float, curve=None, verbose=True):
    """Give every section the series-wide scale (tissue shrinkage) of the reliable sections.

    With a known pixel size, an image's physical size is known up to one shrinkage factor
    per series. The factor is the weighted median over sections; |U| and |V| of every
    section are then set from its pixel width and height.
    """
    w, _ = section_weights(df, curve)
    px = pixel_size_um / VOXEL_UM
    lu = np.array([np.linalg.norm(a[3:6]) for a in df[COLUMNS].values])
    lv = np.array([np.linalg.norm(a[6:9]) for a in df[COLUMNS].values])
    scale = np.sqrt(lu * lv / (df["width"].values * df["height"].values * px * px))
    order = np.argsort(scale)
    cw = np.cumsum(w[order])
    s = scale[order][np.searchsorted(cw, cw[-1] / 2)]
    if verbose:
        print(f"series scale factor: {s:.3f} (1 = no shrinkage)")
    out = df.copy()
    out[COLUMNS] = np.array(
        [
            rescale(a, s * wd * px, s * ht * px)
            for a, wd, ht in zip(df[COLUMNS].values, df["width"].values, df["height"].values)
        ]
    )
    return out


def set_hemisphere(df: pd.DataFrame, side: str, sections=None, all_sections: bool = False):
    """Move one-hemisphere sections to the hemisphere the user knows was imaged.

    The network assumes sections are viewed from the dorsal side; a hemisphere mounted
    ventral side up looks like the other hemisphere. Mirroring the anchoring through the
    midline fixes this. A section is mirrored when most of its tissue lies on the other
    side of the midline.

    By default only sections called one-sided are considered. ``all_sections=True``
    considers every section (use it when the whole series is one hemisphere, so that
    sections wrongly called "both" are moved too). ``sections`` limits the change to rows
    whose filename contains one of the given strings.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    if "hemisphere" not in df:
        raise ValueError("no hemisphere column; run predict() with a horizontal model first")
    hemi = df["hemisphere"].values
    sel = np.ones(len(df), bool) if all_sections else hemi != "both"
    if sections is not None:
        sel &= np.array([any(s in f for s in sections) for f in df["Filenames"]])
    if "left_fraction" in df:
        majority_left = df["left_fraction"].values > 0.5
    else:
        majority_left = hemi == "left"
    flip = sel & (majority_left != (side == "left"))
    df = df.copy()
    if flip.any():
        df.loc[flip, COLUMNS] = np.array([mirror(a) for a in df.loc[flip, COLUMNS].values])
        df.loc[flip & (hemi != "both"), "hemisphere"] = side
        if "left_fraction" in df:
            df.loc[flip, "left_fraction"] = 1 - df.loc[flip, "left_fraction"]
    return df
