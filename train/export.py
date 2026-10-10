"""Convert trained Darknet weights into raw-head ONNX, fully INT8 TFLite, and a manifest.

Both graphs end at the two raw YOLO head convolutions; box decoding and NMS run
on the Pi CPU (ARCHITECTURE.md section 7.2). Calibration images are letterboxed
exactly like inference input.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnx
import onnx.version_converter
import onnxruntime
import onnxsim
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

import pipeline
from pipeline import CLASS_NAMES, Paths, replace_atomically, sha256_file

LETTERBOX_FILL = 128
CALIBRATION_IMAGES = 256
# Darknet emits the stride-32 head first, then the upsampled stride-16 head.
OUTPUTS = (("stride32", 32), ("stride16", 16))
HEAD_CHANNELS = 3 * (len(CLASS_NAMES) + 5)


def input_size(cfg: Path) -> tuple[int, int]:
    """(width, height) of the network input, read from the cfg's [net] section."""
    net = re.split(r"(?m)^(?=\[)", cfg.read_text(encoding="utf-8"))[1]
    width, height = (
        int(re.search(rf"(?m)^{key}\s*=\s*(\d+)", net)[1]) for key in ("width", "height")
    )
    return width, height


def letterbox(path: Path, size: tuple[int, int]) -> np.ndarray:
    """Return an RGB HWC float32 image in [0, 1], aspect-preserving resize plus gray padding."""
    input_width, input_height = size
    image = Image.open(path).convert("RGB")
    scale = min(input_width / image.width, input_height / image.height)
    width, height = round(image.width * scale), round(image.height * scale)
    canvas = Image.new("RGB", size, (LETTERBOX_FILL,) * 3)
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    canvas.paste(resized, ((input_width - width) // 2, (input_height - height) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def calibration_array(paths: Paths) -> Path:
    """Deterministic set of real training frames as an NHWC float32 .npy on scratch disk.

    The frames are recorded in calibration.txt, so the HEF is later quantized with the same
    frames as the TFLite model.
    """
    if paths.calibration.is_file():
        return paths.calibration
    listing = paths.export_dir / "calibration.txt"
    if listing.is_file():
        names = listing.read_text(encoding="utf-8").splitlines()
        selected = [paths.local_dataset / name for name in names]
        if missing := [image for image in selected if not image.is_file()]:
            raise SystemExit(f"Calibration frame {missing[0]} is not in the prepared dataset")
    else:
        # Augmented copies are not representative installation footage (ARCHITECTURE.md §7.2).
        images = [
            image
            for image in pipeline.list_images(paths.local_dataset / "train")
            if not pipeline.is_augmented(image)
        ]
        selected = random.Random(0).sample(images, min(CALIBRATION_IMAGES, len(images)))
        paths.export_dir.mkdir(parents=True, exist_ok=True)
        replace_atomically(
            listing,
            lambda partial: partial.write_text(
                "".join(f"{image.relative_to(paths.local_dataset)}\n" for image in selected),
                encoding="utf-8",
            ),
        )

    size = input_size(paths.cfg)

    def save(partial: Path) -> None:
        with partial.open("wb") as file:  # np.save appends .npy to a bare path
            np.save(file, np.stack([letterbox(image, size) for image in selected]))

    replace_atomically(paths.calibration, save)
    return paths.calibration


def best_weights(paths: Paths) -> Path:
    for name in ("portcullis_best.weights", "portcullis_final.weights"):
        if (paths.backup / name).is_file():
            return paths.backup / name
    raise SystemExit(f"No trained weights in {paths.backup}; run the train step first")


def export_onnx(paths: Paths, weights: Path, output: Path) -> None:
    """Export with Darknet's own tool, ending at the raw YOLO heads (no box decoding or NMS)."""
    # The tool writes <cfg stem>.onnx next to the cfg, so run it on scratch copies.
    scratch = paths.work / "onnx-export"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)
    cfg = shutil.copyfile(paths.cfg, scratch / "portcullis.cfg")
    names = shutil.copyfile(pipeline.NAMES, scratch / "portcullis.names")
    copied = shutil.copyfile(weights, scratch / "portcullis.weights")
    pipeline.run(
        [str(paths.darknet_onnx_export), "-noboxes", "-fp32", str(cfg), str(names), str(copied)]
    )
    model = onnx.load(str(scratch / "portcullis.onnx"))
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != len(OUTPUTS):
        raise SystemExit(
            f"Unexpected ONNX signature: {len(graph.input)} inputs, {len(graph.output)} outputs"
        )
    # Darknet names tensors after its layers; give the graph the names the runtime expects.
    # Its outputs follow the [yolo] sections, stride-32 first; the shape check below verifies.
    renames = {graph.input[0].name: "input"} | {
        value.name: name for value, (name, _) in zip(graph.output, OUTPUTS, strict=True)
    }
    for value in [*graph.input, *graph.output]:
        value.name = renames[value.name]
    for node in graph.node:
        node.input[:] = [renames.get(name, name) for name in node.input]
        node.output[:] = [renames.get(name, name) for name in node.output]
    # Darknet picks the lowest opset its layers need (10 for tiny); onnx2tf and the Hailo
    # parser were validated on opset 13.
    model = onnx.version_converter.convert_version(model, 13)
    # onnx2tf's layout inference mis-axes the CSP route Concats unless the graph is
    # simplified first (the route and yolo layers export as Identity nodes).
    simplified, ok = onnxsim.simplify(model)
    if not ok:
        raise SystemExit("onnxsim could not validate the simplified ONNX graph")
    onnx.checker.check_model(simplified)
    graph = simplified.graph
    shapes = {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in graph.output
    }
    width, height = input_size(paths.cfg)
    expected = {
        name: [1, HEAD_CHANNELS, height // stride, width // stride] for name, stride in OUTPUTS
    }
    input_shape = [dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim]
    if input_shape != [1, 3, height, width] or shapes != expected:
        raise SystemExit(
            f"Unexpected ONNX signature: input {input_shape}, outputs {shapes}, "
            f"expected outputs {expected}"
        )
    onnx.save(simplified, str(output))


def check_onnx_runs(onnx_path: Path, images: list[Path], size: tuple[int, int]) -> None:
    """Smoke-test the exported graph on real frames; numeric parity is a release gate."""
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    for image in images:
        batch = letterbox(image, size).transpose(2, 0, 1)[np.newaxis]
        for (name, _), output in zip(OUTPUTS, session.run(None, {"input": batch}), strict=True):
            if not np.isfinite(output).all():
                raise SystemExit(f"ONNX output {name} is not finite on {image}")
    print(f"ONNX runs on {len(images)} validation images")


def export_tflite(onnx_path: Path, calibration: Path, output: Path, scratch: Path) -> None:
    shutil.rmtree(scratch, ignore_errors=True)
    identity = "[[[[0.0,0.0,0.0]]]]", "[[[[1.0,1.0,1.0]]]]"  # calibration data is already in [0, 1]
    subprocess.run(
        [
            sys.executable, "-m", "onnx2tf",
            "-i", str(onnx_path),
            "-o", str(scratch),
            # flatbuffer_direct's INT8 pass gives a tensor feeding two Concats (layer 23)
            # conflicting scales, which the TFLite runtime rejects.
            "-tb", "tf_converter",
            "-oiqt",
            "-iqd", "int8",
            "-oqd", "int8",
            "-cind", "input", str(calibration), *identity,
            "-n",
        ],
        check=True,
    )  # fmt: skip
    shutil.copyfile(scratch / f"{onnx_path.stem}_full_integer_quant.tflite", output)


def tflite_tensors(path: Path) -> dict[str, list[dict[str, object]]]:
    """Describe the INT8 interface and reject any float tensor left in the graph."""
    interpreter = Interpreter(model_path=str(path))
    floats = [t["name"] for t in interpreter.get_tensor_details() if t["dtype"] == np.float32]
    if floats:
        raise SystemExit(f"{path} is not fully INT8; float tensors: {floats[:5]}")

    def describe(details: list[dict]) -> list[dict[str, object]]:
        return [
            {
                "name": detail["name"],
                "shape": detail["shape"].tolist(),
                "dtype": np.dtype(detail["dtype"]).name,
                "scale": float(detail["quantization"][0]),
                "zero_point": int(detail["quantization"][1]),
            }
            for detail in details
        ]

    return {
        "inputs": describe(interpreter.get_input_details()),
        "outputs": describe(interpreter.get_output_details()),
    }


def yolo_heads(cfg: Path) -> tuple[list[list[int]], list[dict[str, object]]]:
    sections = re.split(r"(?m)^(?=\[)", cfg.read_text(encoding="utf-8"))
    heads = []
    anchors: list[list[int]] = []
    for section in sections:
        if not section.startswith("[yolo]"):
            continue
        options = dict(
            (key.strip(), value.strip())
            for key, value in re.findall(r"(?m)^([a-z_]+)\s*=\s*(.+?)\s*$", section)
        )
        values = [int(value) for value in options["anchors"].split(",")]
        anchors = [values[index : index + 2] for index in range(0, len(values), 2)]
        heads.append(
            {
                "anchor_mask": [int(value) for value in options["mask"].split(",")],
                "scale_x_y": float(options.get("scale_x_y", 1)),
            }
        )
    return anchors, heads


def write_manifest(
    paths: Paths,
    provenance: dict[str, object] | None = None,
    tflite_tensors: dict[str, object] | None = None,
) -> None:
    """(Re)write manifest.json, refreshing hashes of whichever artifacts exist."""
    path = paths.export_dir / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    anchors, heads = yolo_heads(paths.cfg)
    width, height = input_size(paths.cfg)
    manifest.update(
        {
            "model_version": paths.run,
            "architecture": "yolov4-tiny",
            "labels": list(CLASS_NAMES),
            "input": {
                "width": width,
                "height": height,
                "color": "RGB",
                "resize": "letterbox",
                "letterbox_fill": LETTERBOX_FILL,
                "scale": "1/255",
            },
            "anchors": anchors,
            "outputs": [
                {
                    "name": name,
                    "stride": stride,
                    # [rows, columns], the spatial order of the NCHW head tensor
                    "grid": [height // stride, width // stride],
                    "channels": HEAD_CHANNELS,
                    "channel_layout": "per anchor: x, y, w, h, objectness, class scores",
                    **head,
                }
                for (name, stride), head in zip(OUTPUTS, heads, strict=True)
            ],
            "decode": "boxes and NMS are computed on the CPU; graphs end at the raw heads",
        }
    )
    if provenance is not None:
        manifest["provenance"] = provenance
    artifacts = manifest.setdefault("artifacts", {})
    for name in ("model.onnx", "model_int8.tflite", "model.hef", "labels.txt"):
        if (paths.export_dir / name).is_file():
            artifacts.setdefault(name, {})["sha256"] = sha256_file(paths.export_dir / name)
    if tflite_tensors is not None:
        artifacts["model_int8.tflite"]["tensors"] = tflite_tensors
    manifest["updated_at"] = datetime.now(UTC).isoformat()
    text = json.dumps(manifest, indent=2) + "\n"
    replace_atomically(path, lambda partial: partial.write_text(text, encoding="utf-8"))


def export_complete(paths: Paths) -> bool:
    """True once the TFLite model, labels and a manifest recording that exact model exist.

    Drive writes can be cut short by a Colab disconnect, so a present file alone does not
    prove a finished export; the manifest is written last and vouches for the model.
    """
    tflite = paths.export_dir / "model_int8.tflite"
    manifest = paths.export_dir / "manifest.json"
    if not (tflite.is_file() and (paths.export_dir / "labels.txt").is_file()):
        return False
    try:
        artifact = json.loads(manifest.read_text(encoding="utf-8"))["artifacts"][tflite.name]
    except (OSError, ValueError, KeyError):
        return False
    return "tensors" in artifact and artifact.get("sha256") == sha256_file(tflite)


def export(paths: Paths) -> None:
    tflite = paths.export_dir / "model_int8.tflite"
    if export_complete(paths):
        print(f"Already exported: {tflite}")
        return
    paths.export_dir.mkdir(parents=True, exist_ok=True)
    # A rerun exports anew, so it reselects calibration frames from the current dataset
    # and drops the manifest that vouched for a previous model.
    (paths.export_dir / "calibration.txt").unlink(missing_ok=True)
    paths.calibration.unlink(missing_ok=True)
    (paths.export_dir / "manifest.json").unlink(missing_ok=True)
    weights = best_weights(paths)
    print(f"Exporting {weights}")
    onnx_path = paths.export_dir / "model.onnx"
    export_onnx(paths, weights, paths.work / "model.onnx")
    valid = pipeline.list_images(paths.local_dataset / "valid")
    check_onnx_runs(paths.work / "model.onnx", valid[:2], input_size(paths.cfg))
    replace_atomically(
        onnx_path, lambda partial: shutil.copyfile(paths.work / "model.onnx", partial)
    )

    calibration = calibration_array(paths)
    export_tflite(onnx_path, calibration, paths.work / "model_int8.tflite", paths.work / "onnx2tf")
    tensors = tflite_tensors(paths.work / "model_int8.tflite")
    replace_atomically(
        tflite, lambda partial: shutil.copyfile(paths.work / "model_int8.tflite", partial)
    )

    (paths.export_dir / "labels.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    write_manifest(
        paths,
        {
            "weights": weights.name,
            "weights_sha256": sha256_file(weights),
            "darknet_revision": pipeline.DARKNET_REVISION,
            "train_images": len(pipeline.list_images(paths.local_dataset / "train")),
            "valid_images": len(valid),
            "calibration_images": int(np.load(calibration, mmap_mode="r").shape[0]),
        },
        tensors,
    )
    print(f"Wrote {onnx_path}, {tflite} and manifest.json")
