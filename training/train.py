"""Fine-tune the DeepSlice network on synthetic horizontal sections.

Two phases, as in the original training (Carey et al. 2023): first only the dense head
is trained with the Xception backbone frozen, then the whole backbone is unfrozen at a
lower learning rate, which is reduced further when validation loss stops improving.

The Xception ``predictions`` layer (ImageNet classifier, whose 1000 softmax outputs feed
the head) is kept frozen by default: the published DeepSlice weights do not include it,
so inference always uses the ImageNet values for that layer.

Example::

    python -m training.train --out-dir training/runs/h01 --val-set training/data/val_5k.npz
"""

from __future__ import annotations

import argparse
import json
import os
import time

import keras
import numpy as np

from . import dataset, evaluate
from .export_weights import export_deepslice_weights


def build_model(init: str, init_weights: str = None):
    from DeepSlice.metadata import metadata_loader
    from DeepSlice.neural_network import neural_network

    config, path = metadata_loader.load_config()
    xception = metadata_loader.get_data_path(config["weight_file_paths"]["xception_imagenet"], path)
    weights = {
        "mixed": config["weight_file_paths"]["mouse"]["primary"],
        "synthetic": config["weight_file_paths"]["mouse"]["secondary"],
        "imagenet": None,
    }[init]
    if weights is not None:
        weights = metadata_loader.get_data_path(weights, path)
    if init_weights:
        weights = init_weights
    return neural_network.initialise_network(xception, weights, "mouse"), xception


def set_backbone_trainable(model, trainable: bool, train_predictions: bool = False):
    backbone = model.layers[0]
    backbone.trainable = trainable
    if trainable and not train_predictions:
        backbone.get_layer("predictions").trainable = False


class DeepSliceExport(keras.callbacks.Callback):
    """Export DeepSlice-format weights whenever val_loss improves."""

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        loss = (logs or {}).get("val_loss", np.inf)
        if loss < self.best:
            self.best = loss
            export_deepslice_weights(self.model, self.path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-set", required=True, help=".npz made with python -m training.dataset")
    p.add_argument("--init", choices=["mixed", "synthetic", "imagenet"], default="mixed")
    p.add_argument("--init-weights", help="DeepSlice-format weights to start from (overrides --init)")
    p.add_argument("--data-dir", default=dataset.DEFAULT_DATA_DIR)
    p.add_argument("--train-shards", help="directory of pre-generated shards (otherwise generate on the fly)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps-per-epoch", type=int, default=1000)
    p.add_argument("--head-epochs", type=int, default=2)
    p.add_argument("--epochs", type=int, default=60, help="epochs with the backbone unfrozen")
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--patience", type=int, default=12, help="early-stopping patience (epochs)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--train-predictions-layer", action="store_true")
    a = p.parse_args(argv)

    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    fixed = dataset.load_fixed_set(a.val_set)
    val = dataset.make_fixed_dataset(fixed, a.batch_size)
    if a.train_shards:
        train = dataset.make_shard_dataset(a.train_shards, a.batch_size, a.steps_per_epoch, workers=min(a.workers, 12))
    else:
        train = dataset.make_train_dataset(a.batch_size, a.steps_per_epoch, a.data_dir, workers=a.workers)
    model, _ = build_model(a.init, a.init_weights)
    best_h5 = os.path.join(a.out_dir, "best_deepslice.h5")
    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(a.out_dir, "history.csv"), append=True),
        keras.callbacks.ModelCheckpoint(os.path.join(a.out_dir, "best.weights.h5"), save_best_only=True, save_weights_only=True),
        DeepSliceExport(best_h5),
    ]

    t0 = time.time()
    if a.head_epochs > 0:
        set_backbone_trainable(model, False)
        model.compile(optimizer=keras.optimizers.Adam(a.lr_head), loss="mse", metrics=["mae"])
        model.fit(train, validation_data=val, epochs=a.head_epochs, callbacks=callbacks, verbose=2)

    set_backbone_trainable(model, True, a.train_predictions_layer)
    model.compile(optimizer=keras.optimizers.Adam(a.lr), loss="mse", metrics=["mae"])
    callbacks += [
        keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=4, min_lr=a.min_lr, verbose=1),
        keras.callbacks.EarlyStopping(patience=a.patience, restore_best_weights=True, verbose=1),
    ]
    model.fit(train, validation_data=val, initial_epoch=a.head_epochs, epochs=a.head_epochs + a.epochs, callbacks=callbacks, verbose=2)
    print(f"training took {(time.time() - t0) / 3600:.2f} h")

    # evaluate the exported best weights through the DeepSlice inference path
    pred = evaluate.predict_fixed(best_h5, fixed, a.batch_size)
    res, per = evaluate.summarise(pred, fixed)
    res["depth_reliability"] = evaluate.depth_reliability_curve(per["depth_true"], per["distance"])
    with open(os.path.join(a.out_dir, "val_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "depth_reliability"}, indent=2))


if __name__ == "__main__":
    main()
