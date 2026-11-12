# Training DeepSlice on horizontal sections

This folder trains a DeepSlice model for **horizontal mouse sections tilted up to ~30°**,
imaged as nuclear-stain (DAPI-like) fluorescence. It is not part of the installed package.

As in the original paper's synthetic arm, training images are virtual sections cut
through the Allen CCFv3 Nissl and STPT (average template) volumes. Allen has no
horizontal series, so the plane positions are sampled rather than taken from metadata.

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e . "tensorflow[and-cuda]" nibabel pynrrd scipy pytest matplotlib
```

On NEMO GPU nodes run `ml CUDA` first (the submit scripts do): without it TensorFlow
cannot load cuSOLVER and silently falls back to the CPU.

Allen volumes (from `download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/`)
go in `training/data/`: `ara_nissl/ara_nissl_25.nrrd`, `average_template/average_template_25.nrrd`
and `annotation/ccf_2017/annotation_25.nrrd`. They are converted to QuickNII axis order
and cached as `*_xyz.npy` on first use.

DeepSlice weights go in `DeepSlice/metadata/weights/` (the EBRAINS/CSCS download host is
not reachable from NEMO; fetch them elsewhere and copy them over).

## Steps (from the repository root)

```bash
python -m training.slicer --out preview.png          # look at random sections
sbatch training/submit_make_val.sh                   # fixed validation set (5k)
sbatch --export=ALL,RUN=h01 training/submit_train.sh # train on one GPU
python -m training.evaluate --weights training/runs/h01/best_deepslice.h5 \
    --val-set training/data/val_5k.npz --out training/runs/h01/eval.json
python -m training.eval_series --out-dir training/runs/series  # whole series through DSModel
python -m training.eval_real --images B1/png --truth B1/aligned.json --thickness 50 \
    --hemisphere-crops --derotate --out-dir training/runs/real_B1  # hand-aligned real brain
```

Training shards: `sbatch --array=0-49 training/submit_make_shards.sh` writes 1M samples
(about 90 GB) to `training/data/shards/`; pass `SHARDS=training/data/shards` (and
`LOCAL_COPY=1`, which is faster than reading over NEMO storage) to `submit_train.sh`.

`best_deepslice.h5` is in the layout `load_xception_weights` reads; copy it to
`DeepSlice/metadata/weights/DeepSlice_mouse_horizontal.h5` to use it with
`DSModel("mouse", orientation="horizontal")`. Put `depth_reliability` from the evaluation
into `config.json` under `depth_weights.mouse_horizontal`: the series post-processing uses
it to weight sections by how reliable the network is at their depth.

## Conventions

- Coordinates are QuickNII `ABA_Mouse_CCFv3_2017_25um` voxels: x towards the animal's right
  (0–455, midline 227.5), y towards anterior (0–527), z towards superior (0–319).
- Horizontal images are a **dorsal view with anterior at the top**: U points to the
  animal's right, V points posterior. Sections mounted ventral side up are mirror images;
  the model labels them as the mirrored position (see `slicer.mirror_anchoring`), and
  `DSModel.set_hemisphere` moves one-hemisphere sections to the known side.
- Tilt is two slope angles: pitch (dz/dy, anterior up is positive) and roll (dz/dx,
  right side up is positive).
- Preprocessing must match inference exactly (`augment.network_input`): nearest-neighbour
  resize to 299×299, grey, subtract the per-image mean, divide by the per-image standard
  deviation, copy into three channels. `tests/test_training_tf.py` checks this against
  `load_images_from_list`.
- The Xception `predictions` layer is frozen during training (the published weights do
  not include it, so inference uses the ImageNet values).

## Files

| File | Purpose |
|---|---|
| `slicer.py` | volume loading, anchoring geometry and label-exact transforms, plane/frame sampling, hemisphere and partial cuts |
| `augment.py` | DAPI-like appearance, inference-matched preprocessing |
| `dataset.py` | on-the-fly Keras training data, fixed validation sets |
| `train.py` | two-phase fine-tuning, DeepSlice-format export of the best weights |
| `export_weights.py` | writes the HDF5 layout of `Allen_Mixed_Best.h5` |
| `evaluate.py` | paper metric (mean in-brain point distance), per-tilt/depth/hemisphere breakdown, depth reliability curve |
| `eval_series.py` | synthetic dorsal-to-ventral series (incl. face-down hemispheres) through the full `DSModel` pipeline, before/after post-processing |
| `eval_real.py` | real sections aligned by hand in QuickNII/VisuAlign: error before/after post-processing, true image rotations, `--derotate` (upright via the hand alignment), `--hemisphere-crops` (midline crops with exact labels), `--weights` to compare runs |
