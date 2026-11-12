"""Synthetic horizontal training data, generated on the fly from the Allen volumes."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from . import augment, slicer

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MASK_SIZE = 64  # resolution of the stored tissue masks used by the evaluation metric
MAX_HALFSPACES = 2
HEMI_CODES = {"both": 0, "left": 1, "right": 2}

_VOLUME_CACHE = {}


def get_volumes(data_dir=DEFAULT_DATA_DIR):
    if data_dir not in _VOLUME_CACHE:
        _VOLUME_CACHE[data_dir] = slicer.load_volumes(data_dir)
    return _VOLUME_CACHE[data_dir]


@dataclass
class SampleConfig:
    sampler: slicer.HorizontalSamplerConfig = field(default_factory=slicer.HorizontalSamplerConfig)
    appearance: augment.AppearanceConfig = field(default_factory=augment.AppearanceConfig)
    p_nissl: float = 0.75  # otherwise the STPT average template
    oversample_choices: tuple = (1, 2, 3)
    oversample_p: tuple = (0.4, 0.4, 0.2)
    downsample: str = "nearest"


CANVAS_VOX = 560.0  # 14 mm field of view of the fixed-canvas model


def canvas_config() -> SampleConfig:
    """Fixed 14 mm field of view, any in-plane rotation, blocks and BARseq-like features."""
    return SampleConfig(
        sampler=slicer.HorizontalSamplerConfig(
            max_inplane_deg=180.0, canvas_vox=CANVAS_VOX, p_hemisphere=0.4, p_block=0.35,
            min_tissue_fraction=0.01,
        ),
        appearance=augment.AppearanceConfig(p_blobs=0.3, p_rim=0.4, p_seams=0.3, p_rolony=0.3),
        oversample_choices=(2, 3),
        oversample_p=(0.6, 0.4),
        downsample="box",
    )


CONFIGS = {"default": SampleConfig, "canvas": canvas_config}


def make_sample(rng, vols, cfg: SampleConfig = None):
    """One synthetic section: (uint8 299x299 image, float32 label (9,), metadata dict)."""
    cfg = cfg or SampleConfig()
    s = slicer.sample_horizontal_section(rng, vols["mask"], cfg.sampler)
    from_nissl = rng.uniform() < cfg.p_nissl
    vol = vols["ara_nissl_25" if from_nissl else "average_template_25"]
    k = int(rng.choice(cfg.oversample_choices, p=cfg.oversample_p))
    size = augment.NETWORK_SIZE * k
    img, tissue = slicer.render_section(vol, vols["mask"], s, size, size)
    img = augment.dapi_appearance(img, tissue, rng, cfg.appearance, k=k, from_nissl=from_nissl)
    img = augment.to_network_size(img, cfg.downsample)
    meta = dict(
        hemisphere=HEMI_CODES[s.hemisphere],
        pitch=s.pitch,
        roll=s.roll,
        inplane=s.inplane,
        depth=s.depth,
        tissue_fraction=s.tissue_fraction,
        nissl=from_nissl,
        halfspaces=s.halfspaces,
    )
    return img, s.anchoring.astype(np.float32), meta


def tissue_mask_for(anchoring, halfspaces, mask, size=MASK_SIZE):
    """In-brain pixels of a frame (used to mask the evaluation metric)."""
    P = slicer.pixel_grid(anchoring, size, size)
    from scipy.ndimage import map_coordinates

    m = map_coordinates(mask, P, order=0, cval=0, mode="constant") > 0
    return m & slicer.apply_halfspaces(P, halfspaces)


def _pack_halfspaces(halfspaces):
    arr = np.zeros((MAX_HALFSPACES, 4), np.float32)
    for i, (n, d) in enumerate(halfspaces[:MAX_HALFSPACES]):
        arr[i, :3], arr[i, 3] = n, d
    return arr, len(halfspaces[:MAX_HALFSPACES])


def unpack_halfspaces(arr, n):
    return [(arr[i, :3].astype(np.float64), float(arr[i, 3])) for i in range(int(n))]


def _fixed_chunk(args):
    seed, chunk, n, data_dir, cfg = args
    vols = get_volumes(data_dir)
    rng = np.random.default_rng([seed, chunk])
    images = np.zeros((n, augment.NETWORK_SIZE, augment.NETWORK_SIZE), np.uint8)
    labels = np.zeros((n, 9), np.float32)
    masks = np.zeros((n, MASK_SIZE, MASK_SIZE), bool)
    hs = np.zeros((n, MAX_HALFSPACES, 4), np.float32)
    nhs = np.zeros(n, np.int8)
    meta = {key: np.zeros(n, np.float32) for key in META_KEYS}
    for i in range(n):
        img, lab, m = make_sample(rng, vols, cfg)
        images[i], labels[i] = img, lab
        hs[i], nhs[i] = _pack_halfspaces(m["halfspaces"])
        masks[i] = tissue_mask_for(lab, m["halfspaces"], vols["mask"])
        for key in META_KEYS:
            meta[key][i] = m[key]
    return dict(images=images, labels=labels, masks=masks, halfspaces=hs, n_halfspaces=nhs, **meta)


META_KEYS = ["hemisphere", "pitch", "roll", "inplane", "depth", "tissue_fraction", "nissl"]


def make_fixed_set(path, n, seed, data_dir=DEFAULT_DATA_DIR, cfg: SampleConfig = None, workers=1, chunk=200):
    """Generate a fixed (validation/test) set and save it as a compressed .npz.

    The result depends only on ``seed``, ``n`` and ``chunk``, not on ``workers``.
    """
    jobs = [(seed, c, min(chunk, n - c * chunk), data_dir, cfg) for c in range(int(np.ceil(n / chunk)))]
    if workers > 1:
        from multiprocessing import Pool

        with Pool(workers) as pool:
            parts = pool.map(_fixed_chunk, jobs)
    else:
        parts = [_fixed_chunk(j) for j in jobs]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(path, **out)


def load_fixed_set(path):
    return dict(np.load(path))


def make_shard(out_dir, shard, n, seed, data_dir=DEFAULT_DATA_DIR, cfg: SampleConfig = None, workers=1, chunk=200):
    """Write one training shard: ``shard_NNNN_images.npy`` (uint8, memory-mappable) and
    ``shard_NNNN_meta.npz`` (labels and sample metadata).

    Pre-generating shards on CPU nodes decouples data generation (~150 ms per sample per
    core) from GPU training (~350 samples/s on an A100).
    """
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(seed * 100_000 + shard, c, min(chunk, n - c * chunk), data_dir, cfg) for c in range(int(np.ceil(n / chunk)))]
    if workers > 1:
        from multiprocessing import Pool

        with Pool(workers) as pool:
            parts = pool.map(_fixed_chunk, jobs)
    else:
        parts = [_fixed_chunk(j) for j in jobs]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    base = os.path.join(out_dir, f"shard_{shard:04d}")
    np.save(base + "_images.npy.tmp.npy", out.pop("images"))
    np.savez(base + "_meta.npz", **out)
    os.replace(base + "_images.npy.tmp.npy", base + "_images.npy")  # written last: marks the shard complete


def load_shards(shard_dir):
    """Memory-map all complete shards. Returns (list of image arrays, labels, shard index, row index)."""
    import glob

    images, labels, shard_idx, row_idx = [], [], [], []
    for f in sorted(glob.glob(os.path.join(shard_dir, "shard_*_images.npy"))):
        im = np.load(f, mmap_mode="r")
        lab = np.load(f.replace("_images.npy", "_meta.npz"))["labels"]
        assert len(im) == len(lab), f
        shard_idx.append(np.full(len(im), len(images)))
        row_idx.append(np.arange(len(im)))
        images.append(im)
        labels.append(lab)
    if not images:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    return images, np.concatenate(labels), np.concatenate(shard_idx), np.concatenate(row_idx)


# ---------------------------------------------------------------------------
# Keras datasets (imported lazily so the module works without TensorFlow)
# ---------------------------------------------------------------------------
def _keras():
    import keras

    return keras


def make_train_dataset(batch_size, steps_per_epoch, data_dir=DEFAULT_DATA_DIR, cfg=None, workers=8, use_multiprocessing=True, max_queue_size=32):
    keras = _keras()

    class SyntheticSections(keras.utils.PyDataset):
        def __init__(self):
            super().__init__(workers=workers, use_multiprocessing=use_multiprocessing, max_queue_size=max_queue_size)

        def __len__(self):
            return steps_per_epoch

        def __getitem__(self, idx):
            rng = np.random.default_rng()  # fresh entropy: every batch is new
            vols = get_volumes(data_dir)
            x = np.zeros((batch_size, augment.NETWORK_SIZE, augment.NETWORK_SIZE, 3), np.float32)
            y = np.zeros((batch_size, 9), np.float32)
            for i in range(batch_size):
                img, lab, _ = make_sample(rng, vols, cfg)
                x[i], y[i] = augment.network_input(img), lab
            return x, y

    return SyntheticSections()


def make_fixed_dataset(fixed, batch_size):
    """Keras dataset over a fixed set loaded with :func:`load_fixed_set`."""
    keras = _keras()
    images, labels = fixed["images"], fixed["labels"]

    class FixedSections(keras.utils.PyDataset):
        def __len__(self):
            return int(np.ceil(len(images) / batch_size))

        def __getitem__(self, idx):
            sl = slice(idx * batch_size, (idx + 1) * batch_size)
            x = np.stack([augment.network_input(im) for im in images[sl]])
            return x, labels[sl]

    return FixedSections()


def make_shard_dataset(shard_dir, batch_size, steps_per_epoch, workers=8):
    """Keras dataset of random batches drawn from pre-generated shards."""
    keras = _keras()
    images, labels, shard_idx, row_idx = load_shards(shard_dir)
    print(f"{len(labels)} training samples in {len(images)} shards")

    class ShardSections(keras.utils.PyDataset):
        def __init__(self):
            super().__init__(workers=workers, use_multiprocessing=False, max_queue_size=32)

        def __len__(self):
            return steps_per_epoch

        def __getitem__(self, idx):
            sel = np.sort(np.random.default_rng().choice(len(labels), batch_size, replace=False))
            x = np.stack([augment.network_input(images[shard_idx[i]][row_idx[i]]) for i in sel])
            return x, labels[sel]

    return ShardSections()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate a fixed validation set (--out) or a training shard (--shard-dir)")
    p.add_argument("--out", help="validation/test set .npz")
    p.add_argument("--shard-dir", help="directory for training shards")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--config", choices=sorted(CONFIGS), default="default")
    a = p.parse_args()
    cfg = CONFIGS[a.config]()
    if a.shard_dir:
        make_shard(a.shard_dir, a.shard, a.n, a.seed, a.data_dir, cfg=cfg, workers=a.workers)
    else:
        make_fixed_set(a.out, a.n, a.seed, a.data_dir, cfg=cfg, workers=a.workers)
