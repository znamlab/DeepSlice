"""Evaluate DeepSlice-format weights on a fixed synthetic set.

Main metric (Carey et al. 2023): mean Euclidean distance between corresponding points of
the predicted and true frames, over in-brain pixels, in 25 um voxels.
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import dataset, slicer

CENTRE_XY = (slicer.MIDLINE_X, slicer.BRAIN_CENTRE_Y)


def frame_distance(pred, true, masks):
    """Per-section mean distance (voxels) between predicted and true points over ``masks``."""
    n, size = masks.shape[0], masks.shape[1]
    st = (np.arange(size) + 0.5) / size
    s, t = np.meshgrid(st, st)  # s across width (columns), t down height (rows)
    d = np.asarray(pred, np.float64) - np.asarray(true, np.float64)
    D = d[:, None, None, 0:3] + s[None, ..., None] * d[:, None, None, 3:6] + t[None, ..., None] * d[:, None, None, 6:9]
    dist = np.linalg.norm(D, axis=-1)
    out = np.full(n, np.nan)
    for i in range(n):
        if masks[i].any():
            out[i] = dist[i][masks[i]].mean()
    return out


def centre_depth(anchoring):
    """z where the section plane crosses the brain's central DV axis (x=227.5, y=264)."""
    O, U, V = slicer.split_anchoring(anchoring)
    n = np.cross(U, V)
    return O[2] - (n[0] * (CENTRE_XY[0] - O[0]) + n[1] * (CENTRE_XY[1] - O[1])) / n[2]


def normal_angle(pred, true):
    a = np.array([slicer.plane_normal(p) for p in pred])
    b = np.array([slicer.plane_normal(t) for t in true])
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(a * b, axis=1)), 0, 1)))


def summarise(pred, fixed):
    true = fixed["labels"]
    dist = frame_distance(pred, true, fixed["masks"])
    depth_true = np.array([centre_depth(a) for a in true])
    depth_err = np.abs(np.array([centre_depth(a) for a in pred]) - depth_true)
    ang = normal_angle(pred, true)
    tilt = np.hypot(fixed["pitch"], fixed["roll"])
    res = dict(
        n=int(len(true)),
        distance_median=float(np.nanmedian(dist)),
        distance_mean=float(np.nanmean(dist)),
        normal_angle_median=float(np.median(ang)),
        centre_depth_err_median=float(np.median(depth_err)),
        by_hemisphere={},
        by_tilt={},
        by_depth={},
    )
    for name, code in dataset.HEMI_CODES.items():
        sel = fixed["hemisphere"] == code
        if sel.any():
            res["by_hemisphere"][name] = float(np.nanmedian(dist[sel]))
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40)]:
        sel = (tilt >= lo) & (tilt < hi)
        if sel.any():
            res["by_tilt"][f"{lo}-{hi}"] = float(np.nanmedian(dist[sel]))
    edges = np.arange(0, 340, 20)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (depth_true >= lo) & (depth_true < hi)
        if sel.sum() >= 5:
            res["by_depth"][f"{lo}-{hi}"] = float(np.nanmedian(dist[sel]))
    per_sample = dict(distance=dist, normal_angle=ang, depth_err=depth_err, depth_true=depth_true)
    return res, per_sample


def depth_reliability_curve(depth_true, dist, edges=np.arange(0, 340, 20), min_n=5):
    """Weights 1/error^2 per depth bin (normalised to max 1), for series post-processing."""
    centres, w = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (depth_true >= lo) & (depth_true < hi) & np.isfinite(dist)
        if sel.sum() >= min_n:
            centres.append((lo + hi) / 2)
            w.append(1.0 / np.median(dist[sel]) ** 2)
    w = np.array(w)
    return dict(z=[float(c) for c in centres], weight=[float(x) for x in w / w.max()])


def predict_fixed(weights, fixed, batch_size=32, xception_weights=None):
    from DeepSlice.neural_network import neural_network
    from DeepSlice.metadata import metadata_loader

    if xception_weights is None:
        config, path = metadata_loader.load_config()
        xception_weights = metadata_loader.get_data_path(config["weight_file_paths"]["xception_imagenet"], path)
    model = neural_network.initialise_network(xception_weights, weights, "mouse")
    ds = dataset.make_fixed_dataset(fixed, batch_size)
    return model.predict(ds, verbose=0).astype(np.float64)


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, help="DeepSlice-format .h5")
    p.add_argument("--val-set", required=True)
    p.add_argument("--out", required=True, help="output .json")
    a = p.parse_args()
    fixed = dataset.load_fixed_set(a.val_set)
    pred = predict_fixed(a.weights, fixed)
    res, per = summarise(pred, fixed)
    res["depth_reliability"] = depth_reliability_curve(per["depth_true"], per["distance"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    np.savez(a.out.replace(".json", "_per_sample.npz"), pred=pred, **per)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
