"""Tests for the synthetic-section geometry in training/slicer.py."""

import json

from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from training import slicer

DATA_DIR = Path(__file__).resolve().parents[1] / "training" / "data"
HAS_VOLUMES = (DATA_DIR / "annotation_25.nrrd").exists() or (DATA_DIR / "mask_xyz.npy").exists()


@pytest.fixture(scope="module")
def volume():
    """Smooth random volume spanning the full atlas width (so the midline is meaningful)."""
    rng = np.random.default_rng(0)
    return gaussian_filter(rng.standard_normal((456, 48, 40)), 2).astype(np.float64)


def _anchoring():
    # oblique frame inside the small test volume
    O = np.array([150.0, 40.0, 30.0])
    U = np.array([160.0, -5.0, -8.0])
    V = np.array([10.0, -30.0, -15.0])
    return slicer.join_anchoring(O, U, V)


def test_crop_is_exact(volume):
    a = _anchoring()
    H, W = 40, 60
    full = slicer.render(volume, a, H, W)
    i0, i1, j0, j1 = 5, 33, 12, 50
    sub = slicer.render(volume, slicer.crop_anchoring(a, j0 / W, j1 / W, i0 / H, i1 / H), i1 - i0, j1 - j0)
    np.testing.assert_allclose(sub, full[i0:i1, j0:j1], atol=1e-6)


def test_flips_are_exact(volume):
    a = _anchoring()
    img = slicer.render(volume, a, 30, 50)
    np.testing.assert_allclose(slicer.render(volume, slicer.flip_lr_anchoring(a), 30, 50), img[:, ::-1], atol=1e-6)
    np.testing.assert_allclose(slicer.render(volume, slicer.flip_ud_anchoring(a), 30, 50), img[::-1, :], atol=1e-6)


def test_mirror_matches_flipped_volume(volume):
    a = _anchoring()
    np.testing.assert_allclose(
        slicer.render(volume[::-1], a, 30, 50),
        slicer.render(volume, slicer.mirror_anchoring(a), 30, 50),
        atol=1e-6,
    )


def test_canonical_flip_keeps_u_sign_and_matches_symmetric_volume(volume):
    sym = (volume + volume[::-1]) / 2  # bilaterally symmetric
    a = _anchoring()
    c = slicer.canonical_flip_lr(a)
    assert np.sign(c[3]) == np.sign(a[3])
    np.testing.assert_allclose(slicer.render(sym, c, 30, 50), slicer.render(sym, a, 30, 50)[:, ::-1], atol=1e-6)


def test_rotation_by_90_on_square_frame(volume):
    O = np.array([150.0, 44.0, 20.0])
    U = np.array([30.0, 0.0, 0.0])
    V = np.array([0.0, -30.0, 0.0])
    a = slicer.join_anchoring(O, U, V)
    img = slicer.render(volume, a, 32, 32)
    np.testing.assert_allclose(slicer.render(volume, slicer.rotate_anchoring(a, 90), 32, 32), np.rot90(img, 1), atol=1e-6)
    np.testing.assert_allclose(slicer.rotate_anchoring(a, 0), a, atol=1e-9)


def test_rotation_preserves_frame_size_and_plane():
    a = _anchoring()
    r = slicer.rotate_anchoring(a, 23.0)
    for i in (3, 6):
        assert np.isclose(np.linalg.norm(r[i : i + 3]), np.linalg.norm(a[i : i + 3]))
    assert np.allclose(np.abs(slicer.plane_normal(r) @ slicer.plane_normal(a)), 1.0)
    centre = lambda x: x[0:3] + x[3:6] / 2 + x[6:9] / 2  # noqa: E731
    assert np.allclose(centre(r), centre(a))


@pytest.mark.parametrize("pitch,roll", [(0, 0), (20, 0), (0, -25), (15, 10), (-30, 12)])
def test_tilt_recovered(pitch, roll):
    eu, ev = slicer._plane_axes(pitch, roll, inplane_deg=11.0)
    a = slicer.join_anchoring(np.zeros(3), 400 * eu, 500 * ev)
    p, r = slicer.horizontal_tilt(a)
    assert np.isclose(p, pitch, atol=1e-6) and np.isclose(r, roll, atol=1e-6)


def test_pitch_and_roll_signs():
    # pitch > 0: anterior (top of image, -V direction) is higher
    eu, ev = slicer._plane_axes(20, 0, 0)
    assert ev[2] < 0
    # roll > 0: animal's right (+U direction) is higher
    eu, ev = slicer._plane_axes(0, 20, 0)
    assert eu[2] > 0 and eu[0] > 0


@pytest.mark.skipif(not HAS_VOLUMES, reason="Allen volumes not downloaded to training/data")
class TestWithAllenVolumes:
    @pytest.fixture(scope="class")
    @classmethod
    def vols(cls):
        return slicer.load_volumes(str(DATA_DIR))

    def test_coronal_orientation_matches_examples(self, vols):
        from PIL import Image
        from skimage.transform import resize

        example = Path(__file__).resolve().parents[1] / "examples" / "example_brain" / "GLTa"
        d = json.loads((example / "GLTa.json").read_text())
        rs = []
        for s in d["slices"][::5]:
            img = np.asarray(Image.open(example / s["filename"]).convert("L"), float)
            img = resize(img, (128, 128), anti_aliasing=True)
            r = slicer.render(vols["ara_nissl_25"], s["anchoring"], 128, 128)
            rs.append(np.corrcoef(gaussian_filter(r, 1).ravel(), (255 - img).ravel())[0, 1])
        assert np.mean(rs) > 0.7

    def test_sampler_labels_are_canonical_and_hemispheres_correct(self, vols):
        rng = np.random.default_rng(3)
        seen = set()
        for _ in range(60):
            s = slicer.sample_horizontal_section(rng, vols["mask"])
            seen.add(s.hemisphere)
            O, U, V = slicer.split_anchoring(s.anchoring)
            assert U[0] > 0, "U must point to the animal's right (dorsal view)"
            assert V[1] < 0, "V must point posterior (anterior at the top)"
            p, r = slicer.horizontal_tilt(s.anchoring)
            assert np.hypot(p, r) <= 35.0 + 1e-6
            if s.hemisphere != "both":
                img, tissue = slicer.render_section(vols["ara_nissl_25"], vols["mask"], s, 96, 96)
                P = slicer.pixel_grid(s.anchoring, 96, 96)
                x, y = P[0][tissue], P[1][tissue]
                cfg = slicer.HorizontalSamplerConfig()
                # the cut is offset and slanted, so allow for both
                tol = cfg.hemi_offset_vox + np.tan(np.radians(cfg.hemi_angle_deg)) * np.abs(y - slicer.BRAIN_CENTRE_Y) + 1
                sign = -1 if s.hemisphere == "left" else 1
                assert np.all(sign * (x - slicer.MIDLINE_X) > -tol)
        assert seen == {"both", "left", "right"}
