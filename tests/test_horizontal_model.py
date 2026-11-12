"""End-to-end plumbing test of DSModel(orientation="horizontal").

Until horizontal weights exist, the coronal weights stand in for them, so this checks
that every step runs and keeps its guarantees, not prediction accuracy.
"""

import copy
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tensorflow")

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "DeepSlice" / "metadata" / "weights"
DATA = ROOT / "training" / "data"
pytestmark = pytest.mark.skipif(
    not (WEIGHTS / "Allen_Mixed_Best.h5").exists() or not (DATA / "mask_xyz.npy").exists(),
    reason="needs DeepSlice weights and the Allen volumes in training/data",
)


@pytest.fixture
def series_dir(tmp_path):
    """A few synthetic horizontal sections (one hemisphere included) named _sNNN."""
    from PIL import Image

    from training import augment, slicer

    vols = slicer.load_volumes(str(DATA))
    rng = np.random.default_rng(0)
    truth = {}
    cfg = slicer.HorizontalSamplerConfig(max_tilt_deg=0.0, max_inplane_deg=0.0, p_hemisphere=0.0, p_partial=0.0)
    for i, z in enumerate(np.linspace(110, 230, 7)):
        cfg.depth_range = (z, z)
        s = slicer.sample_horizontal_section(rng, vols["mask"], cfg)
        if i == 3:
            s.halfspaces = [(np.array([-1.0, 0, 0]), -slicer.MIDLINE_X)]  # keep the left hemisphere
        img, tissue = slicer.render_section(vols["ara_nissl_25"], vols["mask"], s, 360, 300)
        img = augment.dapi_appearance(img, tissue, rng, k=1)
        name = f"brain_s{10 + 2 * i:03d}.png"
        Image.fromarray(img).save(tmp_path / name)
        truth[name] = s.anchoring
    return tmp_path, truth


@pytest.fixture
def model(monkeypatch):
    from DeepSlice import DSModel
    from DeepSlice.metadata import metadata_loader

    config, path = metadata_loader.load_config()
    config = copy.deepcopy(config)
    config["weight_file_paths"]["mouse_horizontal"]["primary"] = config["weight_file_paths"]["mouse"]["primary"]
    monkeypatch.setattr(metadata_loader, "load_config", lambda: (config, path))
    return DSModel("mouse", orientation="horizontal")


def test_horizontal_pipeline(model, series_dir, tmp_path, monkeypatch):
    from DeepSlice import main
    from DeepSlice.coord_post_processing import horizontal as hz

    folder, truth = series_dir
    real = main.neural_network.predictions_util

    def noisy_truth(*args, **kwargs):
        # run the network for real, then swap in noisy true anchorings: the stand-in
        # coronal weights cannot predict horizontal planes
        df = real(*args, **kwargs)
        rng = np.random.default_rng(1)
        df[hz.COLUMNS] = [truth[f] + rng.normal(0, 2, 9) for f in df["Filenames"]]
        return df

    monkeypatch.setattr(main.neural_network, "predictions_util", noisy_truth)
    model.predict(str(folder) + "/", section_numbers=True)
    p = model.predictions
    assert {"hemisphere", "tissue_fraction"} <= set(p.columns)
    assert p.loc[p["Filenames"] == "brain_s016.png", "hemisphere"].item() == "left"
    assert (p.loc[p["Filenames"] != "brain_s016.png", "hemisphere"] == "both").all()
    assert list(p["nr"]) == sorted(p["nr"])
    model.set_bad_sections(["_s016"])
    model.propagate_angles()
    tilts = np.array([hz.tilt(a) for a in model.predictions[hz.COLUMNS].values])
    assert np.allclose(tilts, tilts[0], atol=1e-6)
    model.enforce_index_order()
    model.enforce_index_spacing(section_thickness=50)
    d = model.predictions["depths"].values
    assert np.allclose(np.abs(np.diff(d)), 2 * 50 / 25)  # sections numbered 2 apart
    model.set_hemisphere("left")
    model.propagate_scale(pixel_size_um=40)
    out = tmp_path / "result"
    model.save_predictions(str(out))
    saved = json.loads(Path(str(out) + ".json").read_text())
    assert len(saved["slices"]) == 7 and saved["target"] == "ABA_Mouse_CCFv3_2017_25um.cutlas"
    assert "hemisphere" in pd.read_csv(str(out) + ".csv").columns


def test_coronal_methods_reject_horizontal(model):
    with pytest.raises(NotImplementedError):
        model.adjust_angles(0, 0)


def test_horizontal_requires_mouse():
    from DeepSlice import DSModel

    with pytest.raises(ValueError):
        DSModel("rat", orientation="horizontal")


def test_canvas_model_maps_predictions_back_to_images(monkeypatch, tmp_path):
    """Fixed-field-of-view models: canvases in, anchorings of the original images out."""
    from PIL import Image

    from DeepSlice import DSModel, main
    from DeepSlice.coord_post_processing import horizontal as hz
    from DeepSlice.metadata import metadata_loader
    from training import slicer

    vols = slicer.load_volumes(str(DATA))
    rng = np.random.default_rng(4)
    px_um, truth = 40.0, {}
    for i, z in enumerate([150.0, 170.0, 190.0]):
        s = slicer.sample_horizontal_section(rng, vols["mask"], depth=z)
        lu, lv = np.linalg.norm(s.anchoring[3:6]), np.linalg.norm(s.anchoring[6:9])
        w, h = int(round(lu * 25 / px_um)), int(round(lv * 25 / px_um))
        a = slicer.crop_anchoring(s.anchoring, 0, w * px_um / 25 / lu, 0, h * px_um / 25 / lv)  # square pixels
        img, _ = slicer.render_section(vols["ara_nissl_25"], vols["mask"], slicer.SectionSample(a, s.halfspaces), h, w)
        name = f"sq_s{i + 1:03d}.png"
        Image.fromarray((np.clip(img, 0, 1) * 230).astype(np.uint8)).save(tmp_path / name)
        truth[name] = a

    regions = {}
    real_prepare = hz.prepare_canvases

    def recording_prepare(paths, px, fov, folder):
        out = real_prepare(paths, px, fov, folder)
        regions.update(out)
        return out

    def exact_canvas_anchorings(model, generator, *args, **kwargs):
        names = [os.path.basename(f) for f in generator.filenames]
        rows = []
        for n in names:
            (x0, y0, side), (w, h) = regions[n]
            a = truth[n]
            rows.append(np.concatenate([a[0:3] + x0 / w * a[3:6] + y0 / h * a[6:9], side / w * a[3:6], side / h * a[6:9]]))
        df = pd.DataFrame(np.array(rows), columns=hz.COLUMNS)
        df.insert(0, "Filenames", names)
        return df

    config, path = metadata_loader.load_config()
    config = copy.deepcopy(config)
    config["weight_file_paths"]["mouse_horizontal"] = dict(config["weight_file_paths"]["mouse"], field_of_view_um=14000)
    config["weight_file_paths"]["mouse_horizontal"].pop("secondary")
    monkeypatch.setattr(metadata_loader, "load_config", lambda: (config, path))
    monkeypatch.setattr(hz, "prepare_canvases", recording_prepare)
    monkeypatch.setattr(main.neural_network, "predictions_util", exact_canvas_anchorings)
    model = DSModel("mouse", orientation="horizontal")
    with pytest.raises(ValueError):
        model.predict(str(tmp_path) + "/")  # pixel size is required
    model.predict(str(tmp_path) + "/", pixel_size_um=px_um)
    p = model.predictions.set_index("Filenames")
    for name, a in truth.items():
        assert np.allclose(p.loc[name, hz.COLUMNS].values.astype(float), a, atol=1e-6)
        assert (p.loc[name, "width"], p.loc[name, "height"]) == Image.open(tmp_path / name).size
