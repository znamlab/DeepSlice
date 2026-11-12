"""Series-level check of the horizontal model with synthetic dorsal-to-ventral series.

Each series has one tilt and regular section spacing, like a real brain. Sections are
rendered as DAPI-like PNGs named ``_sNNN``, run through the full
``DSModel(orientation="horizontal")`` pipeline, and compared with the truth before and
after series post-processing, separately for the middle and the dorsal/ventral ends.

Example::

    python -m training.eval_series --out-dir training/runs/series --n-series 6
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
from PIL import Image

from . import augment, dataset, evaluate, slicer

END_FRACTION = 0.15  # outermost fraction of sections at each end


def make_series(folder, rng, vols, pitch, roll, thickness_um, every, hemisphere, nissl, z_top=305.0, z_bottom=20.0, inplane_sd=5.0, p_face_down=0.0, inplane_max=15.0):
    """Write one series of PNGs; return {filename: (anchoring, mask)} for the sections kept.

    With ``p_face_down`` > 0, that fraction of sections is mounted ventral side up: the
    image is flipped left-right and its true anchoring is the exactly flipped frame.
    """
    cfg = slicer.HorizontalSamplerConfig(y_jitter=0.0, p_partial=0.0)
    vol = vols["ara_nissl_25" if nissl else "average_template_25"]
    step = thickness_um * every / 25.0  # voxels between kept sections
    truth = {}
    for k, z in enumerate(np.arange(z_top, z_bottom, -step)):
        side = hemisphere if hemisphere != "mixed" else rng.choice(["both", "left", "right"], p=[0.5, 0.25, 0.25])
        try:
            s = slicer.sample_horizontal_section(
                rng, vols["mask"], cfg, max_tries=5, pitch=pitch, roll=roll,
                inplane=float(np.clip(rng.normal(0, inplane_sd), -inplane_max, inplane_max)), depth=z, force_hemisphere=side,
            )
        except RuntimeError:
            continue  # too little tissue at this depth
        lu, lv = np.linalg.norm(s.anchoring[3:6]), np.linalg.norm(s.anchoring[6:9])
        h = 600
        w = int(round(h * lu / lv))
        img, tissue = slicer.render_section(vol, vols["mask"], s, h, w)
        img = augment.dapi_appearance(img, tissue, rng, k=2, from_nissl=nissl)
        a = s.anchoring
        if rng.uniform() < p_face_down:
            img, a = img[:, ::-1], slicer.flip_lr_anchoring(a)
        name = f"series_s{1 + k * every:03d}.png"
        Image.fromarray(np.ascontiguousarray(img)).save(os.path.join(folder, name))
        truth[name] = (a, dataset.tissue_mask_for(a, s.halfspaces, vols["mask"]), s.hemisphere)
    return truth


def series_errors(df, truth):
    names = list(df["Filenames"])
    pred = df[["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]].values.astype(np.float64)
    true = np.array([truth[n][0] for n in names])
    masks = np.array([truth[n][1] for n in names])
    return evaluate.frame_distance(pred, true, masks)


def run_series(model, rng, vols, params):
    from DeepSlice.coord_post_processing import horizontal as hz

    with tempfile.TemporaryDirectory() as folder:
        truth = make_series(folder, rng, vols, **params)
        model.predict(folder + "/", section_numbers=True)
        raw = model.predictions.copy()
        if params["hemisphere"] in ("left", "right"):
            # before tilt propagation: mirroring flips the sign of roll
            model.set_hemisphere(params["hemisphere"], all_sections=True)
        model.propagate_angles()
        model.enforce_index_order()
        model.enforce_index_spacing(section_thickness=params["thickness_um"])
        post = model.predictions.copy()
    # raw and post are both sorted by section number, so rows line up
    n = len(raw)
    is_end = np.zeros(n, bool)
    n_end = max(1, int(round(END_FRACTION * n)))
    is_end[:n_end] = is_end[-n_end:] = True
    e_raw, e_post = series_errors(raw, truth), series_errors(post, truth)
    tilt = hz.tilt(post[hz.COLUMNS].values[0])
    return dict(
        params={k: (v if not isinstance(v, np.generic) else v.item()) for k, v in params.items()},
        n_sections=n,
        raw_median=float(np.nanmedian(e_raw)),
        post_median=float(np.nanmedian(e_post)),
        raw_ends_median=float(np.nanmedian(e_raw[is_end])),
        post_ends_median=float(np.nanmedian(e_post[is_end])),
        raw_middle_median=float(np.nanmedian(e_raw[~is_end])),
        post_middle_median=float(np.nanmedian(e_post[~is_end])),
        raw_max=float(np.nanmax(e_raw)),
        post_max=float(np.nanmax(e_post)),
        tilt_error=[float(tilt[0] - params["pitch"]), float(tilt[1] - params["roll"])],
        hemisphere_called=list(raw["hemisphere"]),
        sections=dict(
            nr=[int(x) for x in raw["nr"]],
            true_hemisphere=[truth[n][2] for n in raw["Filenames"]],
            raw_error=[float(x) for x in e_raw],
            post_error=[float(x) for x in e_post],
            raw_depth=[float(hz.centre_depth(a)) for a in raw[hz.COLUMNS].values],
            true_depth=[float(hz.centre_depth(truth[n][0])) for n in raw["Filenames"]],
            raw_tilt=[[float(t) for t in hz.tilt(a)] for a in raw[hz.COLUMNS].values],
            raw_anchoring=raw[hz.COLUMNS].values.astype(float).tolist(),
            post_anchoring=post[hz.COLUMNS].values.astype(float).tolist(),
            true_anchoring=[truth[n][0].tolist() for n in raw["Filenames"]],
            masks=[truth[n][1].astype(int).tolist() for n in raw["Filenames"]],
            tissue_fraction=[float(x) for x in raw["tissue_fraction"]],
            hemisphere=list(raw["hemisphere"]),
            left_fraction=[float(x) for x in raw["left_fraction"]],
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-series", type=int, default=6)
    p.add_argument("--seed", type=int, default=2024)
    a = p.parse_args()

    from DeepSlice import DSModel

    vols = dataset.get_volumes()
    rng = np.random.default_rng(a.seed)
    model = DSModel("mouse", orientation="horizontal")
    results = []
    hemis = ["both", "left", "mixed", "right", "both", "mixed"]
    for i in range(a.n_series):
        mag, direction = rng.uniform(0, 30), rng.uniform(0, 2 * np.pi)
        hemisphere = hemis[i % len(hemis)]
        params = dict(
            pitch=float(mag * np.cos(direction)), roll=float(mag * np.sin(direction)),
            thickness_um=50.0, every=int(rng.choice([2, 4, 6])), hemisphere=hemisphere,
            nissl=bool(i % 3 != 2), p_face_down=0.5 if hemisphere in ("left", "right") else 0.0,
        )
        r = run_series(model, rng, vols, params)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k not in ("hemisphere_called", "sections")}), flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "series_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    keys = ["raw_middle_median", "post_middle_median", "raw_ends_median", "post_ends_median", "raw_max", "post_max"]
    print("mean over series:", {k: round(float(np.mean([r[k] for r in results])), 2) for k in keys})


if __name__ == "__main__":
    main()
