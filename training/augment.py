"""Appearance augmentations that make Allen reslices look like nuclear-stain fluorescence.

All operations change appearance only (plus a small elastic warp of a few pixels), so the
anchoring label of the section is unchanged. They operate on the (possibly oversampled)
render from :mod:`training.slicer`, then reduce it to 299x299 with nearest-neighbour
resampling, which is what DeepSlice inference does to real images
(``flow_from_dataframe`` loads with ``interpolation="nearest"``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d, map_coordinates

NETWORK_SIZE = 299


@dataclass
class AppearanceConfig:
    ap_blur_sigma: tuple = (0.0, 1.0)  # px at 299 scale, along image rows (≈ AP); hides Nissl striping
    p_warp: float = 0.5
    warp_amplitude: float = 2.5  # px at 299 scale
    warp_smoothness: float = 25.0  # px at 299 scale
    gamma: tuple = (0.6, 1.6)
    speckle: tuple = (0.0, 0.5)
    p_damage: float = 0.35
    bias_strength: tuple = (0.0, 0.6)
    background: tuple = (0.0, 0.15)  # relative to the tissue's bright level
    p_dust: float = 0.2
    tissue_level: tuple = (110.0, 255.0)  # grey value of the bright end of tissue
    sensor_noise: tuple = (0.0, 8.0)  # grey levels
    blur_sigma: tuple = (0.0, 1.2)  # px at 299 scale
    dropout: tuple = (0.0, 0.01)
    p_invert: float = 0.05  # occasional brightfield-like polarity, for robustness
    # features of real multiplexed (BARseq/ISS) sections; off by default
    p_blobs: float = 0.0  # injection-site-like bright blobs
    blob_radius: tuple = (5.0, 16.0)  # px at 299 scale
    p_rim: float = 0.0  # bright tissue edge
    p_seams: float = 0.0  # tile seams and tile-to-tile intensity steps
    seam_spacing: tuple = (15.0, 45.0)  # px at 299 scale
    p_rolony: float = 0.0  # sparse bright dots


def _smooth_noise(rng, shape, sigma):
    """Unit-variance random field with spatial correlation length ~``sigma`` pixels.

    Generated on a coarse grid (one node per ``sigma`` pixels) and interpolated, which is
    much faster than filtering full-resolution noise with a wide Gaussian.
    """
    step = max(1.0, float(sigma))
    small = tuple(max(2, int(np.ceil(n / step)) + 2) for n in shape)
    f = gaussian_filter(rng.standard_normal(small), 0.7)
    coords = np.meshgrid(*[np.linspace(0.5, s - 1.5, n) for s, n in zip(small, shape)], indexing="ij")
    f = map_coordinates(f, coords, order=1)
    return f / (f.std() + 1e-8)


def elastic_warp(img, tissue, rng, amplitude, smoothness):
    h, w = img.shape
    dy = _smooth_noise(rng, (h, w), smoothness) * amplitude
    dx = _smooth_noise(rng, (h, w), smoothness) * amplitude
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords = [yy + dy, xx + dx]
    img = map_coordinates(img, coords, order=1, mode="constant", cval=0.0)
    tissue = map_coordinates(tissue.astype(np.float32), coords, order=0, mode="constant", cval=0.0) > 0.5
    return img, tissue


def tissue_damage(img, tissue, rng, k):
    """Holes, tears and folds inside the tissue."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    ys, xs = np.nonzero(tissue)
    if len(ys) == 0:
        return img, tissue
    for _ in range(rng.integers(1, 4)):
        kind = rng.choice(["hole", "tear", "fold"])
        i = rng.integers(len(ys))
        cy, cx = ys[i], xs[i]
        if kind == "hole":
            ry, rx = rng.uniform(2, 15, size=2) * k
            th = rng.uniform(0, np.pi)
            u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
            v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
            sel = (u / rx) ** 2 + (v / ry) ** 2 < 1
            img[sel] = 0
            tissue = tissue & ~sel
        else:
            th = rng.uniform(0, np.pi)
            d = np.abs((xx - cx) * np.sin(th) - (yy - cy) * np.cos(th))
            along = np.abs((xx - cx) * np.cos(th) + (yy - cy) * np.sin(th))
            length = rng.uniform(15, 80) * k
            if kind == "tear":
                sel = (d < rng.uniform(0.5, 2.5) * k) & (along < length)
                img[sel] = 0
                tissue = tissue & ~sel
            else:  # fold: doubled tissue looks brighter
                sel = (d < rng.uniform(2, 6) * k) & (along < length) & tissue
                img[sel] *= rng.uniform(1.4, 2.0)
    return img, tissue


def injection_blobs(img, tissue, rng, cfg, k):
    """1-4 bright, textured, roughly round blobs (solid or ring-shaped) inside the tissue."""
    ys, xs = np.nonzero(tissue)
    if len(ys) == 0:
        return img
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w]
    level = np.percentile(img[tissue], 90)
    for _ in range(rng.integers(1, 5)):
        i = rng.integers(len(ys))
        r = rng.uniform(*cfg.blob_radius) * k
        d = np.hypot(yy - ys[i], xx - xs[i]) / r
        if rng.uniform() < 0.5:
            profile = np.exp(-0.5 * (d / 0.6) ** 4)  # solid
        else:
            profile = np.exp(-0.5 * ((d - 0.7) / 0.2) ** 2) + 0.3 * np.exp(-0.5 * (d / 0.5) ** 2)  # ring
        texture = np.clip(1 + 0.5 * gaussian_filter(rng.standard_normal(img.shape), 0.7 * k), 0, None)
        img = img + rng.uniform(1.5, 4.0) * level * profile * texture * tissue
    return img


def tile_seams(img, rng, cfg, k):
    """Imaging tile grid at a random angle: per-tile gain steps and thin dark/bright seams."""
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    th = rng.uniform(0, np.pi / 2)
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)
    spacing = rng.uniform(*cfg.seam_spacing) * k
    iu, iv = np.floor(u / spacing).astype(int), np.floor(v / spacing).astype(int)
    gains = rng.normal(1.0, rng.uniform(0.02, 0.08), size=(iu.max() - iu.min() + 1, iv.max() - iv.min() + 1))
    out = img * gains[iu - iu.min(), iv - iv.min()]
    width = rng.uniform(0.5, 1.5) * k
    seam = (np.mod(u, spacing) < width) | (np.mod(v, spacing) < width)
    return out * np.where(seam, rng.uniform(0.75, 1.25), 1.0)


def dapi_appearance(img, tissue, rng: np.random.Generator, cfg: AppearanceConfig = None, k: int = 1, from_nissl: bool = True) -> np.ndarray:
    """Turn a render in [0, ~1] into a uint8 nuclear-stain-like image.

    ``k`` is the oversampling factor of the render relative to 299 px; pixel-scale
    parameters are multiplied by it. Returns a uint8 image of the *same* size as the
    input; use :func:`to_network_size` afterwards.
    """
    cfg = cfg or AppearanceConfig()
    img = img.astype(np.float32).copy()
    tissue = tissue.copy()

    if from_nissl:
        img = gaussian_filter1d(img, rng.uniform(*cfg.ap_blur_sigma) * k + 1e-3, axis=0)
    if rng.uniform() < cfg.p_warp:
        img, tissue = elastic_warp(img, tissue, rng, cfg.warp_amplitude * k * rng.uniform(), cfg.warp_smoothness * k)
    img = np.clip(img, 0, None) ** rng.uniform(*cfg.gamma)
    s = rng.uniform(*cfg.speckle)
    if s > 0:
        img *= np.clip(1 + s * gaussian_filter(rng.standard_normal(img.shape), 0.5 * k), 0, None)
    if rng.uniform() < cfg.p_damage:
        img, tissue = tissue_damage(img, tissue, rng, k)
    if cfg.p_blobs > 0 and rng.uniform() < cfg.p_blobs:
        img = injection_blobs(img, tissue, rng, cfg, k)
    if cfg.p_rim > 0 and rng.uniform() < cfg.p_rim:
        d = distance_transform_edt(tissue)
        img = img * (1 + rng.uniform(0.3, 1.5) * np.exp(-d / (rng.uniform(1.0, 4.0) * k)) * tissue)
    if cfg.p_rolony > 0 and rng.uniform() < cfg.p_rolony:
        dots = (rng.uniform(size=img.shape) < rng.uniform(0.002, 0.02)) & tissue
        img = img + dots * rng.uniform(1.0, 3.0) * np.percentile(img[tissue], 90) if tissue.any() else img
    bias = rng.uniform(*cfg.bias_strength)
    if bias > 0:
        img *= np.exp(bias * _smooth_noise(rng, img.shape, img.shape[0] / 6))
    if cfg.p_seams > 0 and rng.uniform() < cfg.p_seams:
        img = tile_seams(img, rng, cfg, k)

    # map to grey levels: bright tissue level, background level, then sensor effects
    hi = np.percentile(img[tissue], 99) if tissue.any() else 1.0
    level = rng.uniform(*cfg.tissue_level)
    out = img / (hi + 1e-8) * level
    bg = rng.uniform(*cfg.background) * level
    out = np.where(tissue, out + bg, bg)
    if rng.uniform() < cfg.p_dust:
        n = rng.integers(1, 30)
        ys, xs = rng.integers(0, img.shape[0], n), rng.integers(0, img.shape[1], n)
        out[ys, xs] = rng.uniform(0.5, 1.0, n) * level
        out = np.maximum(out, gaussian_filter(out, 0.7 * k))
    blur = rng.uniform(*cfg.blur_sigma) * k
    if blur > 0.05:
        out = gaussian_filter(out, blur)
    out = out + rng.standard_normal(out.shape) * rng.uniform(*cfg.sensor_noise)
    p = rng.uniform(*cfg.dropout)
    if p > 0:
        out[rng.uniform(size=out.shape) < p] = 0
    out = np.clip(out, 0, 255)
    if rng.uniform() < cfg.p_invert:
        out = 255 - out
    return out.astype(np.uint8)


def to_network_size(img_u8: np.ndarray, method: str = "nearest") -> np.ndarray:
    """Resize to 299x299.

    "nearest" is what ``keras.utils.load_img`` does to arbitrary images at inference;
    "box" (area averaging) matches inputs prepared at 299x299 beforehand, as the
    fixed-field-of-view horizontal model does.
    """
    if img_u8.shape == (NETWORK_SIZE, NETWORK_SIZE):
        return img_u8
    resample = {"nearest": Image.NEAREST, "box": Image.BOX}[method]
    return np.asarray(Image.fromarray(img_u8).resize((NETWORK_SIZE, NETWORK_SIZE), resample))


def network_input(img_u8: np.ndarray) -> np.ndarray:
    """Reproduce DeepSlice's inference preprocessing for a grey uint8 299x299 image.

    Inference: RGB image -> ``rgb2gray`` (weights sum to 1, so a grey image is
    unchanged) -> subtract the per-image mean and divide by the per-image standard
    deviation (``samplewise_std_normalization=True`` makes Keras also switch on
    ``samplewise_center``) -> the single channel is broadcast into all three channels.
    """
    x = img_u8.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    return np.repeat(x[..., None], 3, axis=-1)
