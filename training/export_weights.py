"""Save a trained model in the HDF5 layout that DeepSlice's ``load_xception_weights`` reads.

Layout (matches the published ``Allen_Mixed_Best.h5``)::

    /xception/<layer>/<weight>:0        attrs["weight_names"] = [b"<layer>/<weight>:0", ...]
    /dense/dense/{kernel:0,bias:0}
    /dense_1/dense_1/{kernel:0,bias:0}
    /dense_2/dense_2/{kernel:0,bias:0}

Unlike the published files, the Xception ``predictions`` layer is saved too (the loader
sets every layer it finds), so exported weights do not depend on the ImageNet file.
"""

from __future__ import annotations

import re

import h5py
import numpy as np

_AUTO_NAMED = ("batch_normalization", "conv2d")  # unnamed layers in keras Xception


def _short_name(weight) -> str:
    # mirrors load_xception_weights: Keras 2 names are "layer/kernel:0", Keras 3 "kernel"
    try:
        return weight.name.split("/")[1]
    except IndexError:
        return f"{weight.name}:0"


def export_deepslice_weights(model, path: str, xception_idx: int = 0) -> None:
    import keras

    xception = model.layers[xception_idx]
    dense_layers = model.layers[xception_idx + 1 : xception_idx + 4]
    # save auto-numbered layers under the numbers a fresh session would give them
    canonical = {}
    for prefix in _AUTO_NAMED:
        pattern = re.compile(rf"{prefix}(_\d+)?")
        auto = [layer.name for layer in xception.layers if pattern.fullmatch(layer.name)]
        canonical.update({n: prefix + (f"_{i}" if i else "") for i, n in enumerate(auto)})
    with h5py.File(path, "w") as f:
        f.attrs["backend"] = "tensorflow"
        f.attrs["keras_version"] = keras.__version__
        f.attrs["layer_names"] = np.array([b"xception", b"dense", b"dense_1", b"dense_2"])
        g = f.create_group("xception")
        names = []
        for layer in xception.layers:
            if not layer.weights:
                continue
            lname = canonical.get(layer.name, layer.name)
            lg = g.create_group(lname)
            for w in layer.weights:
                wname = _short_name(w)
                lg.create_dataset(wname, data=np.asarray(w.numpy()))
                names.append(f"{lname}/{wname}".encode())
        g.attrs["weight_names"] = np.array(names)
        for gname, layer in zip(["dense", "dense_1", "dense_2"], dense_layers):
            kernel, bias = layer.get_weights()
            lg = f.create_group(gname).create_group(gname)
            lg.create_dataset("kernel:0", data=kernel)
            lg.create_dataset("bias:0", data=bias)
