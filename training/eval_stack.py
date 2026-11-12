"""Run the horizontal model on a stack of sections already registered to each other.

When sections have been aligned into a common frame (e.g. BARseq/ISS "volume_stacks"),
their true atlas anchorings in that frame share the same U and V and differ only by a
regular step of the origin with section number. This script:

1. extracts the real sections from the stack (frames that are linear interpolations of
   their neighbours are skipped), cleans the background and numbers them;
2. finds the stack's in-plane orientation from the model itself: the best of four 90
   degree rotations, then refinement with the model's median image-rotation estimate;
3. predicts every section, maps the predictions back to the stack frame and fits one
   rigid stack model to them (robustly, with reliability weights);
4. reports how far each section's own prediction is from the stack fit (an accuracy
   estimate that needs no atlas alignment), the fitted section thickness and tilt, and
   writes QuickNII JSONs and atlas overlays for inspection.

Example: a stack resampled to 10 um in z, where odd frames are the 20 um sections and
even frames are interpolations (section number = (frame + 17) / 2)::

    python -m training.eval_stack \\
        --stack /path/to/volume_stacks/channel_0?.tif \\
        --first-frame 1 --first-section 9 --frames-per-section 2 --thickness 20 --pixel-size 10 \\
        --fov 14000 --name brain1 --out-dir training/runs/brain1
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from DeepSlice.coord_post_processing import horizontal as hz

COLS = hz.COLUMNS


# ---------------------------------------------------------------------------
# Stack loading and cleaning
# ---------------------------------------------------------------------------
def load_stack(paths):
    """Mean of the channel stacks, each normalised by its 99th percentile. (z, y, x)."""
    out = None
    for p in paths:
        im = Image.open(p)
        frames = []
        for i in range(im.n_frames):
            im.seek(i)
            frames.append(np.asarray(im, np.float32))
        v = np.nan_to_num(np.stack(frames))
        v /= np.percentile(v[v > 0], 99)
        out = v if out is None else out + v
    return out / len(paths)


def real_frames(stack, min_pixels=1000):
    """Frames with data that are not linear interpolations of their neighbours."""
    has = np.array([np.count_nonzero(f) > min_pixels for f in stack])
    keep = []
    for k in np.nonzero(has)[0]:
        if 0 < k < len(stack) - 1 and has[k - 1] and has[k + 1]:
            mid = (stack[k - 1] + stack[k + 1]) / 2
            if np.linalg.norm(stack[k] - mid) < 1e-3 * np.linalg.norm(stack[k]):
                continue
        keep.append(int(k))
    return keep


def clean_section(img):
    """uint8 image with only the largest piece of tissue; everything else uniform background.

    Very bright spots (e.g. injection sites) are clipped before thresholding, so that the
    threshold separates tissue from background rather than injection sites from the rest,
    and the intensity scale uses the tissue's 95th percentile so they saturate instead of
    darkening the rest of the section.
    """
    from skimage.filters import threshold_otsu

    smooth = ndimage.gaussian_filter(img, 2)
    vals = smooth[smooth > 0]
    if vals.size == 0:
        return None, None
    tissue = smooth > threshold_otsu(np.minimum(vals, np.percentile(vals, 98)))
    labels, n = ndimage.label(tissue)
    if n == 0:
        return None, None
    sizes = ndimage.sum(tissue, labels, range(1, n + 1))
    tissue = ndimage.binary_fill_holes(labels == 1 + int(np.argmax(sizes)))
    tissue = ndimage.binary_closing(tissue, iterations=3)
    bg = np.median(img[(img > 0) & ~tissue]) if np.any((img > 0) & ~tissue) else 0.0
    out = np.where(tissue, img, bg)
    hi = np.percentile(out[tissue], 95)
    return np.clip(out / hi * 210, 0, 255).astype(np.uint8), tissue


# ---------------------------------------------------------------------------
# Geometry: stack-frame image <-> rotated, cropped network image
# ---------------------------------------------------------------------------
def to_network_image(img, angle):
    """Rotate counterclockwise by ``angle`` (canvas expanded) and crop to the tissue.

    Returns the image and a function mapping stack-frame pixel coordinates (x, y) to
    normalised coordinates (s, t) of the returned image.
    """
    h, w = img.shape
    dummy = np.concatenate([[0.0, 0, 0], [w, 0, 0], [0, h, 0]])  # anchoring = pixel coordinates
    rot, _ = hz.rotate_image_and_anchoring(img, dummy, angle, fill=int(np.median(img[0])))
    crop, _, (r0, r1, c0, c1) = hz.crop_to_tissue(rot, dummy, margin=0.08, return_box=True)
    h2, w2 = rot.shape
    phi = np.radians(angle)

    def stack_to_net(x, y):
        dx, dy = np.asarray(x) - w / 2, np.asarray(y) - h / 2
        x2 = w2 / 2 + dx * np.cos(phi) + dy * np.sin(phi)
        y2 = h2 / 2 - dx * np.sin(phi) + dy * np.cos(phi)
        return (x2 - c0) / (c1 - c0), (y2 - r0) / (r1 - r0)

    return crop, stack_to_net


def to_stack_anchoring(b, stack_to_net, w, h):
    """Anchoring of the full stack-frame image (w x h) from a network-image anchoring b."""
    O, U, V = b[0:3], b[3:6], b[6:9]

    def atlas(x, y):
        s, t = stack_to_net(x, y)
        return O + s * U + t * V

    p00, p10, p01 = atlas(0, 0), atlas(w, 0), atlas(0, h)
    return np.concatenate([p00, p10 - p00, p01 - p00])


def predict_canvas(model, sections, pixel_size_um, folder):
    """Fixed-field-of-view model: predict stack-frame images directly (no rotation search)."""
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))
    for nr, img in sections.items():
        Image.fromarray(img).save(os.path.join(folder, f"stack_s{nr:03d}.png"))
    model.predict(folder + "/", section_numbers=True, pixel_size_um=pixel_size_um)
    df = model.predictions.copy()
    df["net_anchoring"] = list(df[COLS].values.astype(float))
    return df


def predict_sections(model, sections, angle, folder):
    """Predict all sections at a global rotation; returns raw predictions in the stack frame."""
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))
    maps = {}
    for nr, img in sections.items():
        net, m = to_network_image(img, angle)
        name = f"stack_s{nr:03d}.png"
        Image.fromarray(net).save(os.path.join(folder, name))
        maps[name] = m
    model.predict(folder + "/", section_numbers=True)
    df = model.predictions.copy()
    h, w = next(iter(sections.values())).shape
    df["net_anchoring"] = list(df[COLS].values.astype(float))
    df[COLS] = np.array([to_stack_anchoring(a, maps[f], w, h) for a, f in zip(df["net_anchoring"], df["Filenames"])])
    df["width"], df["height"] = w, h
    return df


# ---------------------------------------------------------------------------
# Rigid stack model
# ---------------------------------------------------------------------------
def frame_distance(a, b, w=1.0, h=1.0, n=24):
    """Mean distance between corresponding points of two frames on an n x n grid."""
    s = (np.arange(n) + 0.5) / n
    S, T = np.meshgrid(s, s)
    d = a - b
    D = d[0:3] + S[..., None] * d[3:6] + T[..., None] * d[6:9]
    return float(np.linalg.norm(D, axis=-1).mean())


def fit_rigid_stack(df, curve=None, iterations=10):
    """Fit O = O0 + nr * step, U and V constant, robustly. Returns (fitted df, params)."""
    A = df[COLS].values.astype(float)
    nr = df["nr"].values.astype(float)
    w, _ = hz.section_weights(df, curve)
    rw = np.ones(len(df))
    for _ in range(iterations):
        ww = w * rw
        X = np.stack([np.ones_like(nr), nr], 1) * np.sqrt(ww)[:, None]
        coef = np.linalg.lstsq(X, A[:, 0:3] * np.sqrt(ww)[:, None], rcond=None)[0]  # (2, 3)
        U = np.average(A[:, 3:6], axis=0, weights=ww)
        V = np.average(A[:, 6:9], axis=0, weights=ww)
        fit = np.concatenate([coef[0][None] + nr[:, None] * coef[1][None], np.tile(U, (len(nr), 1)), np.tile(V, (len(nr), 1))], 1)
        resid = np.array([frame_distance(a, f) for a, f in zip(A, fit)])
        rw = hz.tukey_weights(resid, w, min_scale=1.0)
    out = df.copy()
    out[COLS] = fit
    out["stack_residual"] = resid
    out["stack_outlier"] = rw == 0
    normal = hz.plane_normal(fit[0])
    step = coef[1]
    params = dict(
        O0=coef[0].tolist(), step=step.tolist(), U=U.tolist(), V=V.tolist(),
        step_um=float(np.linalg.norm(step) * hz.VOXEL_UM),
        step_along_normal_um=float(abs(step @ normal) * hz.VOXEL_UM),
        shear_deg=float(np.degrees(np.arccos(min(1.0, abs(step @ normal) / np.linalg.norm(step))))),
        pitch_roll=[float(x) for x in hz.tilt(fit[0])],
        pixel_size_um=[float(np.linalg.norm(U) / df["width"].iloc[0] * hz.VOXEL_UM), float(np.linalg.norm(V) / df["height"].iloc[0] * hz.VOXEL_UM)],
    )
    return out, params


def stack_score(df):
    """Median distance of the predictions from the best rigid stack (lower = more consistent)."""
    fitted, _ = fit_rigid_stack(df)
    return float(np.median(fitted["stack_residual"]))


# ---------------------------------------------------------------------------
# Overlays
# ---------------------------------------------------------------------------
def atlas_overlay(sections, fitted, out_png, n_show=8, data_dir=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nrrd

    from training import slicer

    data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
    ann, _ = nrrd.read(os.path.join(data_dir, "annotation_25.nrrd"))
    ann = slicer.allen_to_xyz(ann)
    rows = fitted.sort_values("nr").iloc[np.linspace(0, len(fitted) - 1, n_show).astype(int)]
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(n_show / cols)), cols, figsize=(4.2 * cols, 4.4 * np.ceil(n_show / cols)))
    for ax, (_, r) in zip(axes.ravel(), rows.iterrows()):
        img = sections[int(r["nr"])]
        h, w = img.shape
        lab = slicer.render(ann, r[COLS].values.astype(float), h // 2, w // 2, order=0)
        edge = (np.diff(lab, axis=0, prepend=lab[:1]) != 0) | (np.diff(lab, axis=1, prepend=lab[:, :1]) != 0)
        ax.imshow(img[::2, ::2], cmap="gray")
        ov = np.zeros(edge.shape + (4,))
        ov[edge] = [1, 0.3, 0.1, 0.8]
        ax.imshow(ov)
        ax.set_title(f"s{int(r['nr']):03d}  depth {hz.centre_depth(r[COLS].values.astype(float)):.0f}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=70)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stack", nargs="+", required=True, help="channel TIFF stacks (z = sections), averaged")
    p.add_argument("--first-frame", type=int, required=True)
    p.add_argument("--first-section", type=int, required=True)
    p.add_argument("--frames-per-section", type=int, default=1)
    p.add_argument("--thickness", type=float, required=True, help="section thickness, um")
    p.add_argument("--pixel-size", type=float, default=None, help="stack pixel size, um (for scale checks)")
    p.add_argument("--name", default="stack")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--refine-iterations", type=int, default=3)
    p.add_argument("--weights", default=None, help="DeepSlice-format weights (default: installed horizontal model)")
    p.add_argument("--fov", type=float, default=None, help="field of view (um) of a fixed-canvas model; needs --pixel-size")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    stack = load_stack(sorted(a.stack))
    frames = real_frames(stack)
    sections = {}
    for f in frames:
        nr = a.first_section + (f - a.first_frame) / a.frames_per_section
        if nr != int(nr):
            print(f"frame {f} does not map to a whole section number; skipped")
            continue
        img, _ = clean_section(stack[f])
        if img is not None:
            sections[int(nr)] = img
    print(f"{len(sections)} sections: {min(sections)}-{max(sections)}")
    for nr, img in sections.items():
        Image.fromarray(img).save(os.path.join(a.out_dir, f"{a.name}_s{nr:03d}.png"))

    from DeepSlice import DSModel

    model = DSModel("mouse", orientation="horizontal")
    if a.weights:
        model.config["weight_file_paths"]["mouse_horizontal"]["primary"] = {"path": os.path.abspath(a.weights), "url": None}
    if a.fov:
        model.config["weight_file_paths"]["mouse_horizontal"]["field_of_view_um"] = a.fov
    fov = model.config["weight_file_paths"]["mouse_horizontal"].get("field_of_view_um")
    log = {}
    with tempfile.TemporaryDirectory() as folder:
        if fov:
            if not a.pixel_size:
                raise SystemExit("--pixel-size is required for a fixed field-of-view model")
            raw = predict_canvas(model, sections, a.pixel_size, folder)
            angle, scores = 0.0, {}
        else:
            raw, angle, scores = search_orientation(model, sections, folder, a.refine_iterations)
        log.update(coarse_scores=scores, rotation_deg=angle)

    # 3. rigid stack fit vs the standard series post-processing
    fitted, params = fit_rigid_stack(raw, model._depth_curve())
    model.predictions = raw.drop(columns=["net_anchoring"])
    model.propagate_angles()
    model.enforce_index_order()
    model.enforce_index_spacing(section_thickness=a.thickness)
    post = model.predictions.set_index("nr").loc[raw["nr"]].reset_index()
    _report(a, raw, fitted, params, post, angle, scores, sections)


def search_orientation(model, sections, folder, refine_iterations):
    """Framed (+-15 degree) models: find the stack's in-plane rotation from the model itself."""
    # 1. coarse orientation: which 90 degree rotation gives the most consistent stack?
    scores = {}
    for angle in (0, 90, 180, 270):
        scores[angle] = stack_score(predict_sections(model, sections, angle, folder))
        print(f"rotation {angle:3d}: median distance from rigid stack {scores[angle]:.2f} voxels", flush=True)
    angle = float(min(scores, key=scores.get))
    # 2. refine with the model's own estimate of the remaining rotation
    for _ in range(refine_iterations):
        raw = predict_sections(model, sections, angle, folder)
        residual = float(np.median([hz.image_rotation(b) for b in raw["net_anchoring"]]))
        print(f"rotation {angle:6.1f}: remaining {residual:+.1f} deg, stack consistency {stack_score(raw):.2f} voxels", flush=True)
        if abs(residual) < 1.0:
            break
        angle += residual
    raw = predict_sections(model, sections, angle, folder)
    return raw, angle, scores


def _report(a, raw, fitted, params, post, angle, scores, sections):
    raw_vs_fit = np.array([frame_distance(x, y) for x, y in zip(raw[COLS].values, fitted[COLS].values)])
    post_vs_fit = np.array([frame_distance(x, y) for x, y in zip(post[COLS].values.astype(float), fitted[COLS].values)])
    raw_depth = np.array([hz.centre_depth(x) for x in raw[COLS].values])
    slope = np.polyfit(raw["nr"].values, raw_depth, 1)[0]
    summary = dict(
        n_sections=len(raw),
        rotation_deg=angle,
        coarse_rotation_scores=scores,
        stack_fit=params,
        expected_step_um=a.thickness,
        raw_vs_stack_median=float(np.median(raw_vs_fit)),
        raw_vs_stack_p90=float(np.percentile(raw_vs_fit, 90)),
        post_vs_stack_median=float(np.median(post_vs_fit)),
        raw_depth_slope_um_per_section=float(abs(slope) * hz.VOXEL_UM),
        hemisphere_calls=raw["hemisphere"].value_counts().to_dict(),
        left_fraction_median=float(raw["left_fraction"].median()),
        stack_outliers=fitted.loc[fitted["stack_outlier"], "nr"].tolist(),
        pixel_size_um_given=a.pixel_size,
    )
    print(json.dumps(summary, indent=2))
    with open(os.path.join(a.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    per = pd.DataFrame(dict(nr=raw["nr"], raw_vs_stack=raw_vs_fit, post_vs_stack=post_vs_fit, raw_depth=raw_depth,
                            hemisphere=raw["hemisphere"], tissue_fraction=raw["tissue_fraction"],
                            raw_pitch=[hz.tilt(x)[0] for x in raw[COLS].values], raw_roll=[hz.tilt(x)[1] for x in raw[COLS].values]))
    per.to_csv(os.path.join(a.out_dir, "sections.csv"), index=False)

    # 4. QuickNII JSONs for the extracted stack-frame PNGs
    from DeepSlice.read_and_write.QuickNII_functions import write_QUINT_JSON

    for label, df in [("stack_fit", fitted), ("raw", raw), ("postprocessed", post)]:
        out = df[COLS].copy()
        out["Filenames"] = [f"{a.name}_s{int(n):03d}.png" for n in df["nr"]]
        out["nr"] = df["nr"].values
        out["width"], out["height"] = raw["width"].values, raw["height"].values
        write_QUINT_JSON(out, os.path.join(a.out_dir, f"{a.name}_{label}"), aligner="DeepSlice horizontal", target="ABA_Mouse_CCFv3_2017_25um.cutlas")
    atlas_overlay(sections, fitted, os.path.join(a.out_dir, "overlay_stack_fit.png"))


if __name__ == "__main__":
    main()
