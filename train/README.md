# Training

Trains the Portcullis YOLOv4-tiny detector with Darknet on Google Colab. It
produces the paired release artifacts described in ARCHITECTURE.md §7.2:

- Raw two-head FP32 ONNX
- Fully INT8 TFLite (CPU fallback)
- Hailo-8L HEF
- `labels.txt`
- `manifest.json`

The model contract:

- Classes, in order (the labels of the legacy `cats.zip` dataset):
  `cat_with_mouse`, `cat_with_worm`, `cat_without_prey`, `catface_with_mouse`,
  `catface_with_worm`, `catface_without_prey`
- Input: 896×672 RGB, letterboxed (the 4:3 camera frames fill it without bars).
  The cfg's `[net]` width and height are the single source; export, calibration
  and `manifest.json` read them from there.
- Outputs: two raw heads with 33 channels each, at stride 32 (21×28 grid) and
  stride 16 (42×56 grid)

Box decoding and NMS run on the Pi CPU.

## Run it

1. Upload the dataset zip to Google Drive as `MyDrive/nn/cats.zip` (see
   [Dataset](#dataset)).
2. Open `portcullis_hailo.ipynb` in Colab and select a **T4 GPU** runtime.
3. Set `RUN_NAME`, then choose **Run all**.

The notebook calls a single command, which you can also run on any x86_64
Linux CUDA machine:

```bash
uv run --locked --extra export pipeline.py \
  --root /content/drive/MyDrive/portcullis-training \
  --dataset /content/drive/MyDrive/nn/cats.zip \
  --run v1
```

| Step      | What it does                                                                                                     | Skipped when                     |
| --------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `setup`   | Builds pinned Darknet, or restores it from `<root>/cache/` (see below). Fetches the checksummed `yolov4-tiny.conv.29`. | already present in `--work`      |
| `data`    | Unpacks the dataset to local disk, checks labels, splits it by clip, writes the Darknet `.data` file and lists  | never (scratch is per session)   |
| `train`   | Runs `darknet detector train … -map`. Output is logged to `train.log`; checkpoints go to Drive.                  | `portcullis_final.weights` exists |
| `export`  | Exports best weights to raw-head ONNX with Darknet's `darknet_onnx_export -noboxes`, then to INT8 TFLite, and writes the manifest             | manifest records the exported `model_int8.tflite` |
| `compile` | Builds the HEF with the Hailo Dataflow Compiler                                                                  | `model.hef` exists or no DFC wheel |

The Darknet build is cached on Drive as `<root>/cache/darknet-<revision>-<gpu>-sm<cap>-cuda<ver>-ubuntu<ver>`,
e.g. `darknet-f64b531e-tesla-t4-sm75-cuda12.5-ubuntu22.04`. A different GPU, CUDA toolkit,
OS or Darknet revision gets its own entry. A restored binary is checked with `ldd`; if
Colab's image has dropped a library it needs, `setup` rebuilds and overwrites the entry.

To run a subset of steps, use `--steps data,export`. For a quick end-to-end
smoke run on a new run name, add `--max-batches 200`.

## Google Drive layout

```text
MyDrive/nn/cats.zip         # dataset, passed with --dataset
MyDrive/portcullis-training/
  toolchain/hailo_dataflow_compiler-<version>-py3-none-linux_x86_64.whl   # optional
  runs/<run-name>/
    portcullis.cfg        # frozen copy of darknet/yolov4-tiny-portcullis.cfg
    backup/               # portcullis_last/best/final/<N>.weights
    train.log, chart.png
    export/               # model.onnx, model_int8.tflite, model.hef, model.har,
                          # labels.txt, calibration.txt, manifest.json
```

## Dataset

`--dataset` is a zip file or a folder. Every image (`.jpg`/`.png`) in it, at any
depth, needs a YOLO label file with the same name and a `.txt` extension. Each
line is `<class> <x_center> <y_center> <width> <height>`, with coordinates
normalized to 0–1, and an empty `.txt` file marks an intentional negative image.
Other files, such as the legacy `cats_train.txt`, `cats_valid.txt` and
`cats.cfg`, are ignored.

The v2 dataset lives in `dataset/v2/`: `originals/` holds the 864×648 camera
frames and `resized/` the same frames at the 896×672 network input, which is what
training uses. Both are split into `worm/`, `mouse/` and `no_prey/`, and
`provenance.json` records each frame's source, label origin and clip.

The labels are those of the legacy `cats.zip` dataset. Its plain `cat` class
(id 0) is not part of the model, so dataset labels keep their legacy ids and the
`data` step shifts them down by one. A label with class 0 is rejected.

| Dataset id | Model id | Class                  | Box                                |
| ---------- | -------- | ---------------------- | ---------------------------------- |
| 1          | 0        | `cat_with_mouse`       | whole cat carrying a mouse         |
| 2          | 1        | `cat_with_worm`        | whole cat carrying a worm          |
| 3          | 2        | `cat_without_prey`     | whole cat without prey             |
| 4          | 3        | `catface_with_mouse`   | cat face with a mouse in its mouth |
| 5          | 4        | `catface_with_worm`    | cat face with a worm in its mouth  |
| 6          | 5        | `catface_without_prey` | cat face without prey              |

Image names are capture times, such as `2025-05-04-02-42-56-529894.jpg`.
Offline augmentations of a frame add an `_aug<N>` suffix, such as
`2025-05-04-02-42-56-529894_aug0.jpg`.

The `data` step makes the train/valid split itself, because the legacy lists
use the same images for both:

- Frames less than 30 seconds apart form one clip. A clip is never split
  across sets.
- Clips are grouped by their rarest class, and the newest 20% of each group
  (at least one clip, when the group has two or more) goes to `valid`. This
  keeps every class in validation.
- Augmented frames stay with their original's clip in `train`. Augmentations of
  validation frames are dropped, and so is augmented data in INT8 calibration.

The step prints the image and box counts for each split.

## Resuming

Free Colab disconnects. Darknet writes `portcullis_last.weights` every 100
iterations and `portcullis_<N>.weights` every 1000 iterations, directly to Drive.
Run all resumes from `_last`, falling back to the newest numbered checkpoint.
Darknet keeps the iteration count inside the weights file.

If a disconnect corrupts `_last` mid-write, Darknet fails to load it. Delete that
file and Run all again.

The run's cfg is frozen when the run is first created. To change hyperparameters,
use a new `RUN_NAME`.

## Hailo compilation

The Dataflow Compiler is licensed, so the pipeline never downloads it. To set it
up:

1. Download the DFC wheel for Linux/Python 3.10 from the
   [Hailo Developer Zone](https://hailo.ai/developer-zone/).
2. Put the wheel in `toolchain/` on Drive.

The `compile` step does the rest:

- Installs graphviz.
- Creates a Python 3.10 environment and installs the wheel into it. A failed
  install, or a different wheel in `toolchain/`, is reinstalled on the next run.
- Runs `hailo_compile.py`, which parses the ONNX, applies `hailo/portcullis.alls`
  (on-chip /255 normalization, no NMS), quantizes with the same (up to 256)
  training calibration images as TFLite, and compiles for `hailo8l`.

Without the wheel, the step prints a notice and is skipped.

After a DFC upgrade, replace the wheel, delete `export/model.hef` and rerun to recompile.

## Releasing

A run is not a release until the decoded predictions from the TFLite and HEF
models have been compared, and the HEF has been checked on a real Pi 5 with a
Hailo-8L (ARCHITECTURE.md §7.2). Those checks are manual release gates.

## Development

```bash
uv run --locked --extra export pytest
```

```bash
uv run --locked --extra export ruff check .
```
