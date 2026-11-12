"""Training/inference consistency checks that need TensorFlow."""

from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from DeepSlice.neural_network import neural_network  # noqa: E402
from training import augment  # noqa: E402
from training.export_weights import export_deepslice_weights  # noqa: E402

WEIGHTS = Path(__file__).resolve().parents[1] / "DeepSlice" / "metadata" / "weights"
XCEPTION = WEIGHTS / "xception_weights_tf_dim_ordering_tf_kernels.h5"
MIXED = WEIGHTS / "Allen_Mixed_Best.h5"
needs_xception = pytest.mark.skipif(not XCEPTION.exists(), reason="ImageNet Xception weights not downloaded")


def test_preprocessing_matches_inference_loader(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    paths, expected = [], []
    for i, shape in enumerate([(299, 299), (640, 512), (437, 911)]):
        img = (rng.uniform(size=shape) ** 2 * 255).astype(np.uint8)
        p = tmp_path / f"img_s{i:03d}.png"
        Image.fromarray(img).save(p)
        paths.append(str(p))
        expected.append(augment.network_input(augment.to_network_size(img)))
    gen, _, _ = neural_network.load_images_from_list(paths)
    for i in range(len(paths)):
        np.testing.assert_allclose(next(gen)[0], expected[i], rtol=1e-5, atol=1e-5)


def _predict(model, x):
    return model.predict(x, verbose=0)


@needs_xception
def test_export_round_trip(tmp_path):
    x = np.random.default_rng(1).standard_normal((2, 299, 299, 3)).astype(np.float32)
    model = neural_network.initialise_network(str(XCEPTION), None, "mouse")
    # perturb weights so the test cannot pass by both models holding ImageNet values
    for layer in model.layers[0].layers[-12:] + model.layers[1:]:
        layer.set_weights([w + 0.01 for w in layer.get_weights()])
    out = tmp_path / "w.h5"
    export_deepslice_weights(model, str(out))
    # the second model in the session has differently numbered batch-norm layers
    reloaded = neural_network.initialise_network(str(XCEPTION), str(out), "mouse")
    np.testing.assert_allclose(_predict(reloaded, x), _predict(model, x), rtol=1e-4, atol=1e-3)


@needs_xception
@pytest.mark.skipif(not MIXED.exists(), reason="Allen_Mixed_Best.h5 not downloaded")
def test_every_model_in_a_session_gets_all_saved_weights():
    """Auto-numbered layers (conv2d_N, batch_normalization_N) shift with each new model."""
    import h5py

    with h5py.File(MIXED, "r") as f:
        g = f["xception"]
        order = list(dict.fromkeys(n.decode().split("/")[0] for n in g.attrs["weight_names"]))
        saved = [{k: np.array(v) for k, v in g[name].items()} for name in order]
    for _ in range(3):
        model = neural_network.initialise_network(str(XCEPTION), str(MIXED), "mouse")
        layers = [layer for layer in model.layers[0].layers if layer.weights and layer.name != "predictions"]
        assert len(layers) == len(saved)
        for layer, values in zip(layers, saved):
            for w in layer.weights:
                key = f"{w.name.split('/')[-1].split(':')[0]}:0"
                np.testing.assert_array_equal(w.numpy(), values[key], err_msg=layer.name)
