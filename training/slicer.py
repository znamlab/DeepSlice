"""Reslice Allen CCFv3 volumes along arbitrary planes, in QuickNII anchoring coordinates.

Coordinate system (QuickNII ``ABA_Mouse_CCFv3_2017_25um``), in 25 um voxels with voxel
centres at integer coordinates:

* x: mediolateral, 0..455, increasing towards the animal's right
* y: anteroposterior, 0..527, increasing towards anterior
* z: dorsoventral, 0..319, increasing towards superior

The mapping from Allen's PIR-ordered arrays was verified by reslicing the Allen Nissl
volume with the ground-truth anchorings in ``examples/example_brain/GLTa/GLTa.json``.

An anchoring is a length-9 vector ``(O, U, V)``: image position ``(s, t)`` in ``[0, 1]^2``
(``s`` across the width, ``t`` down the height, from the top-left corner) maps to the
atlas point ``O + s U + t V``.

Canonical horizontal orientation used for training: a *dorsal view* with anterior at the
top of the image, so ``U`` points towards the animal's right (+x) and ``V`` towards
posterior (-y). Sections mounted ventral side up are mirror images; because the atlas is
(almost) bilaterally symmetric they are labelled as the mirrored position, which keeps
the label unambiguous (see :func:`mirror_anchoring`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates

ATLAS_SHAPE = (456, 528, 320)  # x, y, z
MIDLINE_X = 227.5
BRAIN_CENTRE_Y = 264.0
VOLUME_NAMES = ("ara_nissl_25", "average_template_25")


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------
def allen_to_xyz(volume_pir: np.ndarray) -> np.ndarray:
    """Reorder an Allen CCF array (AP, DV, ML; PIR) into QuickNII (x, y, z) order."""
    return np.ascontiguousarray(volume_pir[::-1, ::-1, :].transpose(2, 0, 1))


def load_volumes(data_dir: str, names=VOLUME_NAMES, mmap: bool = True) -> dict:
    """Load the Allen volumes in QuickNII order, normalised to ~[0, 1].

    The first call converts ``<name>.nrrd`` (downloaded from the Allen Institute) into
    ``<name>_xyz.npy`` next to it; later calls memory-map the ``.npy`` files, which is
    cheap and shares memory between data-loading workers.

    Returns a dict with one float32 array per name and ``"mask"``: a uint8 brain mask
    (``annotation_25 > 0``, dilated by one voxel).
    """
    out = {}
    for name in tuple(names) + ("mask",):
        npy = os.path.join(data_dir, f"{name}_xyz.npy")
        if not os.path.exists(npy):
            import nrrd  # pynrrd; only needed for the one-off conversion

            if name == "mask":
                ann, _ = nrrd.read(os.path.join(data_dir, "annotation_25.nrrd"))
                arr = binary_dilation(allen_to_xyz(ann > 0), iterations=1)
                arr = arr.astype(np.uint8)
            else:
                vol, _ = nrrd.read(os.path.join(data_dir, f"{name}.nrrd"))
                arr = allen_to_xyz(vol).astype(np.float32)
                ann, _ = nrrd.read(os.path.join(data_dir, "annotation_25.nrrd"))
                inside = allen_to_xyz(ann > 0)
                arr /= np.percentile(arr[inside], 99.5)
            np.save(npy, arr)
        out[name] = np.load(npy, mmap_mode="r" if mmap else None)
    return out


# ---------------------------------------------------------------------------
# Anchoring geometry
# ---------------------------------------------------------------------------
def split_anchoring(anchoring):
    a = np.asarray(anchoring, dtype=np.float64)
    return a[0:3], a[3:6], a[6:9]


def join_anchoring(O, U, V) -> np.ndarray:
    return np.concatenate([O, U, V]).astype(np.float64)


def pixel_grid(anchoring, height: int, width: int) -> np.ndarray:
    """Atlas coordinates of pixel centres, shape (3, height, width)."""
    O, U, V = split_anchoring(anchoring)
    s = (np.arange(width) + 0.5) / width
    t = (np.arange(height) + 0.5) / height
    P = O[:, None, None] + U[:, None, None] * s[None, None, :] + V[:, None, None] * t[None, :, None]
    return P


def render(volume, anchoring, height: int, width: int, order: int = 1, cval: float = 0.0):
    """Sample ``volume`` (x, y, z order) on the image grid defined by ``anchoring``."""
    return map_coordinates(volume, pixel_grid(anchoring, height, width), order=order, cval=cval, mode="constant")


def crop_anchoring(anchoring, s0: float, s1: float, t0: float, t1: float) -> np.ndarray:
    """Anchoring of the sub-image covering ``[s0, s1] x [t0, t1]`` (fractions of the frame).

    Values outside [0, 1] pad the frame instead.
    """
    O, U, V = split_anchoring(anchoring)
    return join_anchoring(O + s0 * U + t0 * V, (s1 - s0) * U, (t1 - t0) * V)


def flip_lr_anchoring(anchoring) -> np.ndarray:
    """Exact anchoring of the left-right flipped image."""
    O, U, V = split_anchoring(anchoring)
    return join_anchoring(O + U, -U, V)


def flip_ud_anchoring(anchoring) -> np.ndarray:
    """Exact anchoring of the up-down flipped image."""
    O, U, V = split_anchoring(anchoring)
    return join_anchoring(O + V, U, -V)


def mirror_anchoring(anchoring) -> np.ndarray:
    """Reflect an anchoring through the atlas midline (x -> 455 - x).

    Rendering a left-right mirrored volume with ``anchoring`` is identical to rendering
    the original volume with ``mirror_anchoring(anchoring)``. Combined with
    :func:`flip_lr_anchoring` this gives the canonical label of a flipped image (U keeps
    its sign), which is what a bilaterally symmetric atlas requires.
    """
    O, U, V = split_anchoring(anchoring)
    m = np.array([-1.0, 1.0, 1.0])
    return join_anchoring(np.array([2 * MIDLINE_X - O[0], O[1], O[2]]), U * m, V * m)


def canonical_flip_lr(anchoring) -> np.ndarray:
    """Label of a left-right flipped image under the mirror-symmetric convention."""
    return mirror_anchoring(flip_lr_anchoring(anchoring))


def rotate_anchoring(anchoring, angle_deg: float) -> np.ndarray:
    """Rotate the image frame by ``angle_deg`` about its centre, within the section plane.

    Positive angles rotate the frame from U towards V; the tissue then appears rotated
    the opposite way in the image. Frame size (|U|, |V|) is preserved.
    """
    O, U, V = split_anchoring(anchoring)
    c = O + U / 2 + V / 2
    lu, lv = np.linalg.norm(U), np.linalg.norm(V)
    eu = U / lu
    n = np.cross(U, V)
    n /= np.linalg.norm(n)
    ev = np.cross(n, eu)  # in-plane unit vector orthogonal to eu, on the V side
    a = np.radians(angle_deg)
    eu2 = np.cos(a) * eu + np.sin(a) * ev
    ev2 = -np.sin(a) * eu + np.cos(a) * ev
    U2, V2 = lu * eu2, lv * ev2
    return join_anchoring(c - U2 / 2 - V2 / 2, U2, V2)


def plane_normal(anchoring) -> np.ndarray:
    _, U, V = split_anchoring(anchoring)
    n = np.cross(U, V)
    return n / np.linalg.norm(n)


def horizontal_tilt(anchoring):
    """(pitch, roll) in degrees of a section relative to the horizontal plane.

    pitch > 0: anterior end higher (plane rises towards +y).
    roll > 0: animal's right side higher (plane rises towards +x).
    """
    n = plane_normal(anchoring)
    if n[2] < 0:
        n = -n
    pitch = np.degrees(np.arctan2(-n[1], n[2]))
    roll = np.degrees(np.arctan2(-n[0], n[2]))
    return pitch, roll


# ---------------------------------------------------------------------------
# Synthetic horizontal sections
# ---------------------------------------------------------------------------
@dataclass
class HorizontalSamplerConfig:
    max_tilt_deg: float = 35.0  # tilt magnitude hypot(pitch, roll) ~ U(0, max), direction uniform
    max_inplane_deg: float = 15.0
    depth_range: tuple = (18.0, 312.0)  # z range of plane centres (brain DV extent)
    y_jitter: float = 20.0  # jitter of the pivot point along AP, voxels
    margin_range: tuple = (-0.04, 0.25)  # per-side margin, fraction of tissue extent
    p_hemisphere: float = 0.35
    p_partial: float = 0.15
    hemi_offset_vox: float = 20.0  # +-0.5 mm
    hemi_angle_deg: float = 10.0
    max_partial_cuts: int = 2
    min_tissue_vox2: float = 4000.0  # minimum tissue area in the frame (~2.5 mm^2)
    min_tissue_fraction: float = 0.05
    probe_size: int = 160
    probe_extent: float = 760.0  # side of the square probe frame, voxels
    hemisphere_framing_tight_p: float = 0.5
    # fixed physical field of view: a square "canvas" of this side (voxels) centred on the
    # tissue, instead of framing the tissue; the network then sees true size
    canvas_vox: float = 0.0
    canvas_scale_jitter: tuple = (0.88, 1.12)  # tissue shrinkage / pixel-size error
    canvas_centre_jitter: float = 0.05  # fraction of the canvas side
    # blocks: near-coronal cuts removing the front and/or back of the tissue
    p_block: float = 0.0
    block_keep_ap_vox: tuple = (160.0, 400.0)  # kept AP extent of a two-sided block (4-10 mm)
    block_cut_y: tuple = (120.0, 420.0)  # position of a one-sided cut
    block_angle_deg: float = 15.0


@dataclass
class SectionSample:
    anchoring: np.ndarray
    halfspaces: list = field(default_factory=list)  # list of (normal(3,), offset): keep n.P > d
    hemisphere: str = "both"  # "both", "left", "right"
    pitch: float = 0.0
    roll: float = 0.0
    inplane: float = 0.0
    depth: float = 0.0  # z of the pivot point
    tissue_fraction: float = 1.0


def _plane_axes(pitch_deg, roll_deg, inplane_deg):
    """Unit in-plane axes (eu, ev) for a dorsal-view horizontal plane with the given tilt.

    Tilts are slope angles, matching :func:`horizontal_tilt`: the plane satisfies
    dz/dy = tan(pitch) and dz/dx = tan(roll). Before in-plane rotation, eu is the
    projection of +x (animal's right) onto the plane and ev = eu x n points posterior.
    """
    n = np.array([-np.tan(np.radians(roll_deg)), -np.tan(np.radians(pitch_deg)), 1.0])
    n /= np.linalg.norm(n)
    eu = np.array([1.0, 0.0, 0.0]) - n[0] * n
    eu /= np.linalg.norm(eu)
    ev = np.cross(eu, n)
    a = np.radians(inplane_deg)
    return np.cos(a) * eu + np.sin(a) * ev, -np.sin(a) * eu + np.cos(a) * ev


def apply_halfspaces(P: np.ndarray, halfspaces) -> np.ndarray:
    """Boolean mask of grid points (3, H, W) that satisfy every half-space constraint."""
    keep = np.ones(P.shape[1:], dtype=bool)
    for n, d in halfspaces:
        keep &= np.tensordot(np.asarray(n), P, axes=1) > d
    return keep


def _random_cut_through(rng, centre, eu, ev, extent):
    """Half-space whose boundary is a random chord near the edge of the tissue."""
    theta = rng.uniform(0, 2 * np.pi)
    direction = np.cos(theta) * eu + np.sin(theta) * ev  # in-plane, points to removed side
    dist = rng.uniform(0.15, 0.45) * extent
    # keep points with direction.P < direction.centre + dist  <=>  (-direction).P > -(...)
    return (-direction, -(direction @ centre + dist))


def _block_cuts(rng, cfg):
    """Half-spaces of near-coronal cuts keeping the back, the front, or a slab of tissue."""
    cuts = []
    kind = rng.choice(["keep_back", "keep_front", "slab"])

    def cut(y, keep_anterior):
        ang = np.radians(rng.uniform(-cfg.block_angle_deg, cfg.block_angle_deg))
        n = np.array([np.sin(ang), np.cos(ang), 0.0]) * (1.0 if keep_anterior else -1.0)
        return (n, n @ np.array([MIDLINE_X, y, 0.0]))

    if kind == "keep_back":
        cuts.append(cut(rng.uniform(*cfg.block_cut_y), keep_anterior=False))
    elif kind == "keep_front":
        cuts.append(cut(rng.uniform(*cfg.block_cut_y), keep_anterior=True))
    else:
        keep = rng.uniform(*cfg.block_keep_ap_vox)
        y0 = rng.uniform(20.0, max(21.0, 520.0 - keep))
        cuts += [cut(y0, keep_anterior=True), cut(y0 + keep, keep_anterior=False)]
    return cuts


def sample_horizontal_section(
    rng: np.random.Generator,
    mask: np.ndarray,
    cfg: HorizontalSamplerConfig = None,
    max_tries: int = 50,
    pitch=None,
    roll=None,
    inplane=None,
    depth=None,
    force_hemisphere=None,
) -> SectionSample:
    """Sample a tilted horizontal plane and an image frame around the tissue on it.

    ``pitch``, ``roll``, ``inplane``, ``depth`` and ``force_hemisphere`` ("both", "left",
    "right"; "both" also disables partial cuts) fix those quantities instead of drawing
    them, e.g. to build a series.
    """
    cfg = cfg or HorizontalSamplerConfig()
    fixed = dict(pitch=pitch, roll=roll, inplane=inplane, depth=depth)
    for _ in range(max_tries):
        # 1. plane
        # uniform tilt magnitude covers every tilt level equally (uniform-in-disc would put
        # two thirds of samples above 20 degrees)
        magnitude = rng.uniform(0, cfg.max_tilt_deg)
        direction = rng.uniform(0, 2 * np.pi)
        pitch, roll = magnitude * np.cos(direction), magnitude * np.sin(direction)
        inplane = rng.uniform(-cfg.max_inplane_deg, cfg.max_inplane_deg)
        depth = rng.uniform(*cfg.depth_range)
        pitch = pitch if fixed["pitch"] is None else fixed["pitch"]
        roll = roll if fixed["roll"] is None else fixed["roll"]
        inplane = inplane if fixed["inplane"] is None else fixed["inplane"]
        depth = depth if fixed["depth"] is None else fixed["depth"]
        centre = np.array([MIDLINE_X, BRAIN_CENTRE_Y + rng.uniform(-cfg.y_jitter, cfg.y_jitter), depth])
        eu, ev = _plane_axes(pitch, roll, inplane)

        # 2. tissue cuts (random draws happen in the same order whether or not fixed)
        halfspaces = []
        hemisphere = "both"
        r = rng.uniform()
        if force_hemisphere is not None:
            r = 2.0 if force_hemisphere == "both" else -1.0
        if r < cfg.p_hemisphere:
            hemisphere = "left" if rng.uniform() < 0.5 else "right"
            if force_hemisphere is not None:
                hemisphere = force_hemisphere
            ang = np.radians(rng.uniform(-cfg.hemi_angle_deg, cfg.hemi_angle_deg))
            cut_n = np.array([np.cos(ang), np.sin(ang), 0.0])  # roughly +x, slanted in AP
            cut_p = np.array([MIDLINE_X + rng.uniform(-cfg.hemi_offset_vox, cfg.hemi_offset_vox), BRAIN_CENTRE_Y, 0])
            sign = 1.0 if hemisphere == "right" else -1.0
            halfspaces.append((sign * cut_n, sign * (cut_n @ cut_p)))
        elif r < cfg.p_hemisphere + cfg.p_partial:
            for _ in range(rng.integers(1, cfg.max_partial_cuts + 1)):
                halfspaces.append(_random_cut_through(rng, centre, eu, ev, 400.0))
        if cfg.p_block > 0 and rng.uniform() < cfg.p_block:
            halfspaces.extend(_block_cuts(rng, cfg))

        # 3. probe the tissue footprint on a large square frame centred on the pivot
        E = cfg.probe_extent
        probe = join_anchoring(centre - E / 2 * eu - E / 2 * ev, E * eu, E * ev)
        P = pixel_grid(probe, cfg.probe_size, cfg.probe_size)
        tissue = map_coordinates(mask, P, order=0, cval=0, mode="constant") > 0
        tissue &= apply_halfspaces(P, halfspaces)
        px = E / cfg.probe_size
        area = tissue.sum() * px * px
        if area < cfg.min_tissue_vox2:
            continue
        rows, cols = np.nonzero(tissue)
        s0, s1 = cols.min() / cfg.probe_size, (cols.max() + 1) / cfg.probe_size
        t0, t1 = rows.min() / cfg.probe_size, (rows.max() + 1) / cfg.probe_size

        # 4. frame
        if cfg.canvas_vox > 0:
            # fixed field of view centred on the tissue centroid
            side = cfg.canvas_vox * rng.uniform(*cfg.canvas_scale_jitter) / E
            sc = (cols.mean() + 0.5) / cfg.probe_size + rng.uniform(-1, 1) * cfg.canvas_centre_jitter * side
            tc = (rows.mean() + 0.5) / cfg.probe_size + rng.uniform(-1, 1) * cfg.canvas_centre_jitter * side
            anchoring = crop_anchoring(probe, sc - side / 2, sc + side / 2, tc - side / 2, tc + side / 2)
            tissue_fraction = min(1.0, area / (side * E) ** 2)
            if tissue_fraction < cfg.min_tissue_fraction:
                continue
            return SectionSample(anchoring, halfspaces, hemisphere, pitch, roll, inplane, depth, tissue_fraction)
        # frame = tissue bounding box + random margins on each side
        tight = hemisphere == "both" or rng.uniform() < cfg.hemisphere_framing_tight_p
        if not tight:
            # keep the frame centred on the midline: mirror the bbox across it
            mid_s = 0.5  # the probe is centred on the midline at the pivot
            half = max(abs(s0 - mid_s), abs(s1 - mid_s))
            s0, s1 = mid_s - half, mid_s + half
        ws, ht = s1 - s0, t1 - t0
        m = rng.uniform(*cfg.margin_range, size=4)
        s0, s1 = s0 - m[0] * ws, s1 + m[1] * ws
        t0, t1 = t0 - m[2] * ht, t1 + m[3] * ht
        anchoring = crop_anchoring(probe, s0, s1, t0, t1)

        frame_area = np.linalg.norm(anchoring[3:6]) * np.linalg.norm(anchoring[6:9])
        tissue_fraction = min(1.0, area / frame_area)
        if tissue_fraction < cfg.min_tissue_fraction:
            continue
        return SectionSample(anchoring, halfspaces, hemisphere, pitch, roll, inplane, depth, tissue_fraction)
    raise RuntimeError("could not sample a section with enough tissue")


def render_section(volume, mask, sample: SectionSample, height: int, width: int, order: int = 1):
    """Render a sampled section. Returns (image in [0, ~1], boolean tissue mask)."""
    P = pixel_grid(sample.anchoring, height, width)
    img = map_coordinates(volume, P, order=order, cval=0.0, mode="constant").astype(np.float32)
    tissue = map_coordinates(mask, P, order=0, cval=0, mode="constant") > 0
    tissue &= apply_halfspaces(P, sample.halfspaces)
    img[~tissue] = 0.0
    return img, tissue


def preview(data_dir: str, out_png: str, n: int = 24, seed: int = 0, size: int = 299):
    """Save a grid of random synthetic sections for visual inspection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vols = load_volumes(data_dir)
    rng = np.random.default_rng(seed)
    cols = 6
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(3 * cols, 3.2 * np.ceil(n / cols)))
    for ax in axes.ravel():
        ax.axis("off")
    for i in range(n):
        s = sample_horizontal_section(rng, vols["mask"])
        src = VOLUME_NAMES[i % len(VOLUME_NAMES)]
        img, _ = render_section(vols[src], vols["mask"], s, size, size)
        ax = axes.ravel()[i]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"{s.hemisphere} z={s.depth:.0f} p={s.pitch:.0f} r={s.roll:.0f}", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=60)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Preview synthetic horizontal sections")
    p.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "data"))
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    preview(a.data_dir, a.out, a.n, a.seed)
