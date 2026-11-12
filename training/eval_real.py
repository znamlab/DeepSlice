"""Evaluate the horizontal model on real sections aligned by hand in QuickNII/VisuAlign.

Runs ``DSModel(orientation="horizontal")`` on a folder of section images, post-processes
the series, and compares both raw and post-processed anchorings with a hand-aligned
QUINT JSON (only the linear anchoring is used; VisuAlign markers are ignored). Error is
the paper's metric: mean distance between corresponding points of the predicted and true
frames over pixels that are tissue in the image and brain in the atlas, in 25 um voxels.

With ``--hemisphere-crops``, every section that shows both hemispheres is also cut at the
midline (located with the hand alignment), and the two one-hemisphere series are
evaluated the same way: their true anchorings follow exactly from the crop.

The network assumes a dorsal view with the midline vertical and anterior at the top
(+-15 degrees). The true rotation of every section is reported; ``--derotate`` rotates
the images upright using the hand alignment first, to test the model independently of
mounting rotation. A hand alignment of a section mounted ventral side up is compared with
its mirror image, which is equivalent for the symmetric atlas; pass ``--hemisphere`` for
one-hemisphere series to test ``set_hemisphere`` and compare with the actual side.

Example::

    python -m training.eval_real --images brains/B1/png --truth brains/B1/aligned.json \\
        --thickness 50 --hemisphere-crops --out-dir training/runs/real_B1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

from DeepSlice.coord_post_processing import horizontal as hz

from . import evaluate, slicer

COLS = ["ox", "oy", "oz", "ux", "uy", "uz", "vx", "vy", "vz"]
MASK_SIZE = 64
CROP_OVERLAP = 0.03  # crops extend this fraction of the width past the midline, like a real cut


def metric_mask(image, anchoring, brain_mask):
    """Pixels (MASK_SIZE^2) that are tissue in the image and brain in the atlas."""
    from DeepSlice.coord_post_processing.horizontal import tissue_mask

    small = np.asarray(Image.fromarray(image).resize((MASK_SIZE, MASK_SIZE), Image.BILINEAR))
    tissue = tissue_mask(small)
    P = slicer.pixel_grid(anchoring, MASK_SIZE, MASK_SIZE)
    from scipy.ndimage import map_coordinates

    brain = map_coordinates(brain_mask, P, order=0, cval=0, mode="constant") > 0
    both = tissue & brain
    return both if both.any() else brain


def load_truth(path):
    from DeepSlice.read_and_write.QuickNII_functions import read_QUINT_JSON

    df, _ = read_QUINT_JSON(path)
    df = df.dropna(subset=COLS)
    df["stem"] = [os.path.splitext(os.path.basename(f))[0] for f in df["Filenames"]]
    return df


def stage_images(images_dir, truth, folder, derotate=False):
    """Copy aligned images into ``folder`` as greyscale PNGs named ``<stem>_sNNN.png``.

    With ``derotate``, each image is first rotated so that its midline is vertical with
    anterior at the top, using the hand alignment (the anchoring is transformed exactly).
    Returns {staged name: (true anchoring, grey image)}.
    """
    files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir)}
    staged = {}
    for _, row in truth.iterrows():
        if row["stem"] not in files:
            continue
        img = np.asarray(Image.open(os.path.join(images_dir, files[row["stem"]])).convert("L"))
        a = row[COLS].values.astype(np.float64)
        if derotate:
            img, a = hz.rotate_image_and_anchoring(img, a, hz.image_rotation(a))
            img, a = hz.crop_to_tissue(img, a)  # the expanded canvas has wide empty margins
        stem = re.sub(r"_s\d+", "", row["stem"])
        name = f"{stem}_s{int(row['nr']):03d}.png"
        Image.fromarray(img).save(os.path.join(folder, name))
        staged[name] = (a, img)
    if not staged:
        raise ValueError("no image in --images matches a filename in --truth")
    return staged


def run_pipeline(model, folder, thickness, hemisphere):
    model.predict(folder + "/", section_numbers=True)
    raw = model.predictions.copy()
    if hemisphere:
        model.set_hemisphere(hemisphere, all_sections=True)
    model.propagate_angles()
    model.enforce_index_order()
    model.enforce_index_spacing(section_thickness=thickness)
    return raw, model.predictions.copy()


def score(raw, post, staged, brain_mask, compare_actual_side):
    rows = []
    post = post.set_index("Filenames")
    for _, r in raw.iterrows():
        true, img = staged[r["Filenames"]]
        target = true if compare_actual_side else hz.canonical(true)
        mask = metric_mask(img, target, brain_mask)[None]
        e_raw = evaluate.frame_distance(r[COLS].values.astype(float)[None], target[None], mask)[0]
        p = post.loc[r["Filenames"]]
        e_post = evaluate.frame_distance(p[COLS].values.astype(float)[None], target[None], mask)[0]
        rows.append(
            dict(
                Filenames=r["Filenames"], nr=int(r["nr"]), raw_error=e_raw, post_error=e_post,
                hemisphere=r.get("hemisphere"), outlier_depth=bool(p.get("outlier_depth", False)),
                depth_residual=float(p.get("depth_residual", np.nan)),
                flipped_mounting=bool(p.get("flipped_mounting", False)),
                ventral_view_true=bool(hz.is_ventral_view(true)),
                rotation_true=hz.image_rotation(true), rotation_pred=float(r.get("image_rotation", np.nan)),
            )
        )
    df = pd.DataFrame(rows).sort_values("nr").reset_index(drop=True)
    n = len(df)
    n_end = max(1, int(round(0.15 * n)))
    df["end_section"] = False
    df.loc[: n_end - 1, "end_section"] = True
    df.loc[n - n_end :, "end_section"] = True
    return df


def summarise(df):
    out = dict(n=len(df))
    for col in ["raw_error", "post_error"]:
        out[f"{col}_median"] = float(df[col].median())
        out[f"{col}_middle_median"] = float(df.loc[~df["end_section"], col].median())
        out[f"{col}_ends_median"] = float(df.loc[df["end_section"], col].median())
        out[f"{col}_max"] = float(df[col].max())
    out["flagged"] = int(df["outlier_depth"].sum())
    rot = df["rotation_true"].abs()
    out["rotation_true_abs_median"] = float(rot.median())
    out["rotation_true_abs_max"] = float(rot.max())
    for t in (15, 30, 45, 90):
        out[f"rotation_true_over_{t}"] = float((rot > t).mean())
    out["ventral_view_true"] = int(df["ventral_view_true"].sum())
    out["flipped_mounting_called"] = int(df["flipped_mounting"].sum())
    return out


def hemisphere_crops(staged, folder_left, folder_right):
    """Cut both-hemisphere sections at the midline; return staged dicts for each side."""
    sides = {"a": {}, "b": {}}
    for name, (true, img) in staged.items():
        O, U, V = slicer.split_anchoring(true)
        # where the midline crosses the middle row of the image
        s_mid = (slicer.MIDLINE_X - O[0] - 0.5 * V[0]) / U[0]
        if not 0.25 < s_mid < 0.75:
            continue
        h, w = img.shape
        for key, (s0, s1), folder in [("a", (0.0, s_mid + CROP_OVERLAP), folder_left), ("b", (s_mid - CROP_OVERLAP, 1.0), folder_right)]:
            c0, c1 = int(round(s0 * w)), int(round(s1 * w))
            crop = img[:, c0:c1]
            a = slicer.crop_anchoring(true, c0 / w, c1 / w, 0.0, 1.0)
            Image.fromarray(np.ascontiguousarray(crop)).save(os.path.join(folder, name))
            sides[key][name] = (a, crop)
    return sides["a"], sides["b"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True, help="folder of section images")
    p.add_argument("--truth", required=True, help="QuickNII/VisuAlign JSON aligned by hand")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--thickness", type=float, default=None, help="section thickness in um")
    p.add_argument("--hemisphere", choices=["left", "right"], default=None, help="whole series is this hemisphere")
    p.add_argument("--hemisphere-crops", action="store_true", help="also test midline crops of whole sections")
    p.add_argument("--weights", default=None, help="DeepSlice-format weights (default: installed horizontal model)")
    p.add_argument("--derotate", action="store_true", help="rotate images upright using the hand alignment first")
    a = p.parse_args()

    from DeepSlice import DSModel

    brain_mask = slicer.load_volumes(os.path.join(os.path.dirname(__file__), "data"), names=())["mask"]
    model = DSModel("mouse", orientation="horizontal")
    if a.weights:
        model.config["weight_file_paths"]["mouse_horizontal"]["primary"] = {"path": os.path.abspath(a.weights), "url": None}
    os.makedirs(a.out_dir, exist_ok=True)
    truth = load_truth(a.truth)
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        sets = {"whole": os.path.join(tmp, "whole")}
        os.makedirs(sets["whole"])
        staged = {"whole": stage_images(a.images, truth, sets["whole"], derotate=a.derotate)}
        if a.hemisphere_crops:
            for k in ("crop_a", "crop_b"):
                sets[k] = os.path.join(tmp, k)
                os.makedirs(sets[k])
            staged["crop_a"], staged["crop_b"] = hemisphere_crops(staged["whole"], sets["crop_a"], sets["crop_b"])
        for key, folder in sets.items():
            if len(staged[key]) < 3:
                continue
            hemi = a.hemisphere if key == "whole" else None
            raw, post = run_pipeline(model, folder, a.thickness, hemi)
            df = score(raw, post, staged[key], brain_mask, compare_actual_side=hemi is not None)
            df.to_csv(os.path.join(a.out_dir, f"{key}_sections.csv"), index=False)
            model.save_predictions(os.path.join(a.out_dir, f"{key}_predictions"))
            results[key] = summarise(df)
            print(key, json.dumps(results[key]), flush=True)
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
