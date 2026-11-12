"""Tests for DeepSlice/coord_post_processing/horizontal.py."""

import numpy as np
import pandas as pd
import pytest

from DeepSlice.coord_post_processing import horizontal as hz

COLS = hz.COLUMNS


def frame(z, pitch, roll, inplane=0.0, half_w=230.0, half_h=280.0, centre_xy=(hz.MIDLINE_X, hz.CENTRE_Y)):
    """Dorsal-view horizontal anchoring centred on the central DV axis at depth z."""
    n = np.array([-np.tan(np.radians(roll)), -np.tan(np.radians(pitch)), 1.0])
    n /= np.linalg.norm(n)
    eu = np.array([1.0, 0, 0]) - n[0] * n
    eu /= np.linalg.norm(eu)
    ev = np.cross(eu, n)
    a = np.radians(inplane)
    eu, ev = np.cos(a) * eu + np.sin(a) * ev, -np.sin(a) * eu + np.cos(a) * ev
    c = np.array([centre_xy[0], centre_xy[1], z])
    return np.concatenate([c - half_w * eu - half_h * ev, 2 * half_w * eu, 2 * half_h * ev])


def test_frame_helper():
    a = frame(150, 12, -8, inplane=5)
    assert np.allclose(hz.tilt(a), (12, -8))
    assert np.isclose(hz.centre_depth(a), 150)


def test_set_tilt_keeps_depth_and_size():
    a = frame(150, 5, -3, inplane=7)
    b = hz.set_tilt(a, 12, -8)
    assert np.allclose(hz.tilt(b), (12, -8))
    assert np.isclose(hz.centre_depth(b), 150)
    assert np.isclose(np.linalg.norm(b[3:6]), np.linalg.norm(a[3:6]))
    assert np.isclose(np.linalg.norm(b[6:9]), np.linalg.norm(a[6:9]))
    assert np.allclose(hz.set_tilt(b, 5, -3), a)


def test_set_depth():
    a = frame(150, 20, 10)
    b = hz.set_depth(a, 201.5)
    assert np.isclose(hz.centre_depth(b), 201.5)
    assert np.allclose(hz.tilt(b), hz.tilt(a))
    assert np.allclose(b[3:], a[3:])


def test_mirror():
    a = frame(150, 12, -8, centre_xy=(150.0, 264.0))
    m = hz.mirror(a)
    assert np.allclose(hz.mirror(m), a)
    p, r = hz.tilt(m)
    assert np.isclose(p, 12) and np.isclose(r, 8)


def test_rescale_keeps_centre():
    a = frame(150, 12, -8)
    b = hz.rescale(a, 300, 400)
    c = lambda x: x[0:3] + x[3:6] / 2 + x[6:9] / 2  # noqa: E731
    assert np.allclose(c(a), c(b))
    assert np.isclose(np.linalg.norm(b[3:6]), 300) and np.isclose(np.linalg.norm(b[6:9]), 400)


def test_matches_training_conventions():
    slicer = pytest.importorskip("training.slicer")
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = frame(rng.uniform(30, 290), *rng.uniform(-30, 30, 2), inplane=rng.uniform(-15, 15))
        assert np.allclose(hz.tilt(a), slicer.horizontal_tilt(a))
        assert np.allclose(hz.mirror(a), slicer.mirror_anchoring(a))


# ---------------------------------------------------------------------------
# series
# ---------------------------------------------------------------------------
TRUE_PITCH, TRUE_ROLL, THICK_UM = 12.0, -8.0, 50.0


def make_series(seed=0, end_fraction=0.2, hemisphere="both"):
    """A dorsal-to-ventral series: accurate predictions in the middle, bad ones at the ends."""
    rng = np.random.default_rng(seed)
    nr = np.arange(0, 136)
    z_true = 300 - nr * THICK_UM / hz.VOXEL_UM  # numbered dorsal -> ventral
    n_end = int(round(end_fraction * len(nr) / 2))
    rows, truth = [], []
    for i, (k, z) in enumerate(zip(nr, z_true)):
        truth.append(frame(z, TRUE_PITCH, TRUE_ROLL))
        end = i < n_end or i >= len(nr) - n_end
        tilt_sd, depth_sd = (12.0, 25.0) if end else (2.0, 3.0)
        p = TRUE_PITCH + rng.normal(0, tilt_sd)
        r = TRUE_ROLL + rng.normal(0, tilt_sd)
        rows.append(frame(z + rng.normal(0, depth_sd), p, r, inplane=rng.normal(0, 3)))
    df = pd.DataFrame(np.array(rows), columns=COLS)
    df["Filenames"] = [f"sec_s{k:03d}.png" for k in nr]
    df["nr"] = nr
    df["width"], df["height"] = 1000, 1200
    df["hemisphere"] = hemisphere
    df["tissue_fraction"] = 0.6
    is_end = np.zeros(len(nr), bool)
    is_end[:n_end] = is_end[-n_end:] = True
    return df, np.array(truth), is_end


def depth_errors(df, truth):
    return np.abs(np.array([hz.centre_depth(a) for a in df[COLS].values]) - np.array([hz.centre_depth(a) for a in truth]))


@pytest.mark.parametrize("hemisphere", ["both", "left"])
def test_end_sections_follow_structured_sections(hemisphere):
    df, truth, is_end = make_series(hemisphere=hemisphere)
    before = depth_errors(df, truth)
    out = hz.propagate_tilt(df, verbose=False)
    out = hz.enforce_index_spacing(out, verbose=False)
    tilts = np.array([hz.tilt(a) for a in out[COLS].values])
    assert np.allclose(tilts, [TRUE_PITCH, TRUE_ROLL], atol=1.5)
    after = depth_errors(out, truth)
    assert np.median(after[is_end]) < 2.0 and after[is_end].max() < 5.0
    assert np.median(after[is_end]) < np.median(before[is_end]) / 4
    # depth now follows the section-number line, not the end sections' own predictions
    d = np.array([hz.centre_depth(a) for a in out[COLS].values])
    assert np.allclose(np.diff(d), np.diff(d)[0])


def test_catastrophic_end_sections_are_rejected_and_flagged():
    """A tiny fragment at the end of a strongly tilted series can be predicted far off."""
    df, truth, is_end = make_series(seed=3)
    for i, dz in [(1, -100.0), (3, -95.0)]:  # dorsal end, predicted ~100 voxels too deep
        df.loc[i, COLS] = frame(hz.centre_depth(truth[i]) + dz, -1.0, 10.0)
    out = hz.propagate_tilt(df, verbose=False)
    out = hz.enforce_index_order(out)  # must not shuffle the outliers into the series
    out = hz.enforce_index_spacing(out, verbose=False)
    tilts = np.array([hz.tilt(a) for a in out[COLS].values])
    assert np.allclose(tilts, [TRUE_PITCH, TRUE_ROLL], atol=1.0)
    errors = depth_errors(out, truth)
    assert errors.max() < 5.0
    flagged = set(np.nonzero(out["outlier_depth"].values)[0])
    assert {1, 3} <= flagged  # other end sections have 25-voxel noise and may be flagged too
    assert not any(not is_end[i] for i in flagged)
    assert out.loc[[1, 3], "depth_residual"].abs().min() > 90


@pytest.mark.parametrize("p_flipped", [0.0, 0.3])
def test_mixed_mounting_keeps_roll_sign(p_flipped):
    """Sections mounted ventral side up are predicted mirrored: same pitch, opposite roll."""
    df, truth, is_end = make_series(seed=5)
    rng = np.random.default_rng(0)
    flip = rng.uniform(size=len(df)) < p_flipped
    if flip.any():
        df.loc[flip, COLS] = np.array([hz.mirror(a) for a in df.loc[flip, COLS].values])
    truth = np.array([hz.mirror(t) if f else t for t, f in zip(truth, flip)])
    out = hz.enforce_index_spacing(hz.propagate_tilt(df, verbose=False), verbose=False)
    tilts = np.array([hz.tilt(a) for a in out[COLS].values])
    expected_roll = np.where(flip, -TRUE_ROLL, TRUE_ROLL)
    assert np.allclose(tilts[:, 0], TRUE_PITCH, atol=1.0)
    # middle sections are accurate, so their own roll sign is reliable; the noisy end
    # sections (12 degree tilt noise) may fall back to the majority sign
    mid = ~is_end
    assert np.allclose(tilts[mid, 1], expected_roll[mid], atol=1.0)
    assert np.array_equal(out["flipped_mounting"].values[mid], flip[mid])
    assert np.mean(np.isclose(tilts[:, 1], expected_roll, atol=1.0)) > 0.9
    assert np.median(depth_errors(out, truth)) < 1.5


def test_small_roll_with_consistent_mounting_uses_signed_mean():
    rolls = np.random.default_rng(1).normal(0.3, 0.8, 100)  # sign noise, not mixed mounting
    per_section, _, flipped = hz.series_roll(rolls, np.ones(100))
    assert not flipped.any() and np.allclose(per_section, per_section[0])
    assert abs(per_section[0] - 0.3) < 0.3


def test_reliability_weights_beat_uniform_weights():
    flat = {"z": [0.0, 320.0], "weight": [1.0, 1.0]}
    errs = {"weighted": [], "uniform": []}
    for seed in range(5):
        df, truth, _ = make_series(seed)
        for name, curve in [("weighted", None), ("uniform", flat)]:
            out = hz.enforce_index_spacing(hz.propagate_tilt(df, curve, verbose=False), curve=curve, verbose=False)
            t = np.array([hz.tilt(a) for a in out[COLS].values])[0]
            errs[name].append(np.hypot(t[0] - TRUE_PITCH, t[1] - TRUE_ROLL))
    assert np.mean(errs["weighted"]) < np.mean(errs["uniform"])


def test_known_thickness_and_bad_sections():
    df, truth, _ = make_series()
    df["bad_section"] = False
    df.loc[60, COLS] = frame(10, 40, 30)  # wildly wrong, but marked bad
    df.loc[60, "bad_section"] = True
    out = hz.enforce_index_spacing(hz.propagate_tilt(df, verbose=False), section_thickness_um=THICK_UM, verbose=False)
    d = np.array([hz.centre_depth(a) for a in out[COLS].values])
    assert np.allclose(-np.diff(d), THICK_UM / hz.VOXEL_UM)
    assert np.median(depth_errors(out, truth)) < 1.5


def test_enforce_index_order_is_monotonic():
    df, _, _ = make_series()
    out = hz.enforce_index_order(df.sample(frac=1, random_state=0))
    d = np.array([hz.centre_depth(a) for a in out[COLS].values])
    assert np.all(np.diff(d) <= 1e-9)


def test_propagate_scale():
    df, _, is_end = make_series()
    px_um, shrink = 4.0, 0.93
    rng = np.random.default_rng(1)
    true_u = df["width"].values * px_um / hz.VOXEL_UM * shrink
    true_v = df["height"].values * px_um / hz.VOXEL_UM * shrink
    noise = np.where(is_end, rng.normal(1, 0.15, len(df)), rng.normal(1, 0.01, len(df)))
    df[COLS] = np.array([hz.rescale(a, u * k, v * k) for a, u, v, k in zip(df[COLS].values, true_u, true_v, noise)])
    out = hz.propagate_scale(df, px_um, verbose=False)
    lu = np.array([np.linalg.norm(a[3:6]) for a in out[COLS].values])
    assert np.allclose(lu / true_u, 1, atol=0.01)


def test_set_hemisphere_all_sections_uses_majority_side():
    df, _, _ = make_series()
    df = df.iloc[:3].copy()
    df["hemisphere"] = ["both", "left", "right"]
    df["left_fraction"] = [0.7, 0.95, 0.02]  # first: one hemisphere wrongly called "both"
    default = hz.set_hemisphere(df, "right")
    assert np.allclose(default.loc[0, COLS].values.astype(float), df.loc[0, COLS].values.astype(float))
    out = hz.set_hemisphere(df, "right", all_sections=True)
    assert np.allclose(out.loc[0, COLS].values.astype(float), hz.mirror(df.loc[0, COLS].values))
    assert np.allclose(out["left_fraction"], [0.3, 0.05, 0.02])
    assert list(out["hemisphere"]) == ["both", "right", "right"]


def test_set_hemisphere_mirrors_only_wrong_side():
    df, _, _ = make_series()
    df = df.iloc[:3].copy()
    df["hemisphere"] = ["both", "left", "right"]
    out = hz.set_hemisphere(df, "right")
    assert list(out["hemisphere"]) == ["both", "right", "right"]
    assert np.allclose(out.loc[1, COLS].values.astype(float), hz.mirror(df.loc[1, COLS].values))
    assert np.allclose(out.loc[[0, 2], COLS].values.astype(float), df.loc[[0, 2], COLS].values.astype(float))


@pytest.mark.parametrize("brightfield", [False, True])
def test_describe_section(brightfield):
    img = np.zeros((100, 200))
    img[20:80, 20:95] = 200  # tissue on the left half of the image
    if brightfield:
        img = 255 - img
    # U along +x: image left maps to x < midline
    a = np.concatenate([[hz.MIDLINE_X - 200, 400, 200], [400, 0, 0], [0, -300, 0]])
    hemi, frac, left = hz.describe_section(img, a)
    assert hemi == "left" and 0.2 < frac < 0.25 and left == 1.0
    img2 = np.zeros((100, 200))
    img2[20:80, 20:180] = 200
    assert hz.describe_section(img2, a)[0] == "both"


def test_image_rotation_and_view():
    # the sampler's in-plane angle is measured from the midline, for any tilt
    a = frame(150, 12, -8)
    assert np.isclose(hz.image_rotation(a), 0.0, atol=1e-6) and not hz.is_ventral_view(a)
    # frame rotated in plane by +30 deg (U towards V): the midline appears 30 deg anticlockwise
    for p, r in [(12, 0), (25, -25), (-30, 10)]:
        assert np.isclose(hz.image_rotation(frame(150, p, r, inplane=30)), -30.0, atol=1e-6)
    flipped = np.concatenate([a[0:3] + a[3:6], -a[3:6], a[6:9]])  # left-right flipped image
    assert hz.is_ventral_view(flipped) and not hz.is_ventral_view(hz.canonical(flipped))
    assert np.allclose(hz.canonical(a), a)


def test_rotate_image_and_anchoring_is_consistent():
    slicer = pytest.importorskip("training.slicer")
    from scipy.ndimage import gaussian_filter

    vol = gaussian_filter(np.random.default_rng(0).standard_normal((456, 528, 60)), 3)
    vol = ((vol - vol.min()) / np.ptp(vol) * 255).astype(np.float64)
    a = frame(30, 0, 0, inplane=40, half_w=60, half_h=80, centre_xy=(220.0, 260.0))
    img = slicer.render(vol, a, 160, 120).astype(np.uint8)
    rot, b = hz.rotate_image_and_anchoring(img, a, hz.image_rotation(a), fill=0)
    assert abs(hz.image_rotation(b)) < 1e-6
    re = slicer.render(vol, b, *rot.shape)
    inside = slicer.render(np.ones_like(vol), b, *rot.shape) > 0  # inside the volume
    inner = np.zeros_like(rot, bool)
    inner[rot.shape[0] // 3 : 2 * rot.shape[0] // 3, rot.shape[1] // 3 : 2 * rot.shape[1] // 3] = True
    sel = inside & inner
    assert np.corrcoef(rot[sel].astype(float), re[sel])[0, 1] > 0.95


def test_crop_to_tissue_is_exact():
    slicer = pytest.importorskip("training.slicer")
    vol = np.zeros((456, 528, 40))
    vol[150:300, 200:330, :] = 200.0  # a block of "tissue"
    a = frame(20, 0, 0, half_w=200, half_h=200, centre_xy=(225.0, 265.0))
    img = slicer.render(vol, a, 100, 100, order=0).astype(np.uint8)
    crop, b = hz.crop_to_tissue(img, a, margin=0.1)
    assert crop.shape[0] < 100 and crop.shape[1] < 100
    np.testing.assert_array_equal(slicer.render(vol, b, *crop.shape, order=0).astype(np.uint8), crop)


def test_canvas_round_trip_and_content():
    slicer = pytest.importorskip("training.slicer")
    from scipy.ndimage import gaussian_filter

    vol = gaussian_filter(np.random.default_rng(1).standard_normal((456, 528, 60)), 3)
    vol = (vol - vol.min()) / np.ptp(vol) * 200 + 20
    px_um = 50.0  # 2 voxels per pixel
    a = frame(30, 0, 0, inplane=20, half_w=120, half_h=90, centre_xy=(230.0, 270.0))
    W, H = int(240 * 25 / px_um), int(180 * 25 / px_um)
    img = slicer.render(vol, a, H, W).astype(np.uint8)
    canvas, region = hz.to_canvas(img, px_um, fov_um=14000, out_px=299)
    x0, y0, side = region
    # the canvas anchoring implied by the region maps back to the image exactly
    b = np.concatenate([a[0:3] + x0 / W * a[3:6] + y0 / H * a[6:9], side / W * a[3:6], side / H * a[6:9]])
    assert np.allclose(hz.canvas_to_image_anchoring(b, region, W, H), a)
    assert np.isclose(np.linalg.norm(b[3:6]), 14000 / 25) and np.isclose(np.linalg.norm(b[6:9]), 14000 / 25)
    # and the canvas content matches the volume rendered at that anchoring (where the image has data)
    re = slicer.render(vol, b, 299, 299)
    # image coordinates (s, t) of each canvas pixel, to compare only where the image has data
    P = slicer.pixel_grid(b, 299, 299)
    s_img = ((P - a[0:3, None, None]).transpose(1, 2, 0) @ np.linalg.pinv(np.stack([a[3:6], a[6:9]]))).transpose(2, 0, 1)
    inside = (s_img[0] > 0.02) & (s_img[0] < 0.98) & (s_img[1] > 0.02) & (s_img[1] < 0.98)
    assert np.corrcoef(canvas[inside].astype(float), re[inside])[0, 1] > 0.9
