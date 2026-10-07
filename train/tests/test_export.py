from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnx2tf", reason="needs the export extra")

import export  # noqa: E402
import pipeline  # noqa: E402
from pipeline import Paths  # noqa: E402


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    paths = Paths(
        root=tmp_path / "drive", run="v1", work=tmp_path / "work", dataset=tmp_path / "cats"
    )
    paths.export_dir.mkdir(parents=True)
    shutil.copyfile(pipeline.CFG_TEMPLATE, paths.cfg)
    return paths


def test_letterbox_preserves_aspect_with_gray_bars(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "frame.png"
    Image.new("RGB", (240, 180), (255, 0, 0)).save(source)
    image = export.letterbox(source, (416, 416))
    assert image.shape == (416, 416, 3)
    assert image.dtype == np.float32
    np.testing.assert_allclose(image[0, 0], [128 / 255] * 3)  # 416x312 content, 52px bars
    np.testing.assert_allclose(image[208, 208], [1, 0, 0])


def test_letterbox_fills_a_matching_aspect_without_bars(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "frame.png"
    Image.new("RGB", (864, 648), (255, 0, 0)).save(source)
    image = export.letterbox(source, (896, 672))
    assert image.shape == (672, 896, 3)
    np.testing.assert_allclose(image[0, 0], [1, 0, 0])
    np.testing.assert_allclose(image[-1, -1], [1, 0, 0])


def test_input_size_comes_from_the_cfg() -> None:
    assert export.input_size(pipeline.CFG_TEMPLATE) == (896, 672)


def test_manifest_describes_the_raw_heads(paths: Paths) -> None:
    (paths.export_dir / "model.onnx").write_bytes(b"onnx")
    export.write_manifest(paths, {"weights": "portcullis_best.weights"})
    manifest = json.loads((paths.export_dir / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["labels"] == list(pipeline.CLASS_NAMES)
    assert manifest["input"]["width"] == 896 and manifest["input"]["height"] == 672
    assert manifest["anchors"] == [
        [96, 112],
        [154, 208],
        [257, 317],
        [208, 446],
        [306, 490],
        [521, 601],
    ]
    assert [(o["name"], o["grid"], o["anchor_mask"]) for o in manifest["outputs"]] == [
        ("stride32", [21, 28], [3, 4, 5]),
        ("stride16", [42, 56], [0, 1, 2]),
    ]
    assert all(o["channels"] == 33 and o["scale_x_y"] == 1.05 for o in manifest["outputs"])
    assert set(manifest["artifacts"]) == {"model.onnx"}

    # A later compile step refreshes hashes but keeps the export provenance.
    (paths.export_dir / "model.hef").write_bytes(b"hef")
    export.write_manifest(paths)
    manifest = json.loads((paths.export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["artifacts"]) == {"model.onnx", "model.hef"}
    assert manifest["provenance"] == {"weights": "portcullis_best.weights"}


def test_export_is_complete_only_when_the_manifest_vouches_for_the_model(paths: Paths) -> None:
    tflite = paths.export_dir / "model_int8.tflite"
    tflite.write_bytes(b"tflite")
    assert not export.export_complete(paths)  # disconnected before labels and manifest

    (paths.export_dir / "labels.txt").write_text("\n".join(pipeline.CLASS_NAMES) + "\n")
    export.write_manifest(paths, tflite_tensors={"input": {}})
    assert export.export_complete(paths)

    tflite.write_bytes(b"tfl")  # truncated by a later, interrupted copy
    assert not export.export_complete(paths)


def test_compile_reuses_the_recorded_calibration_frames(paths: Paths) -> None:
    from PIL import Image

    def prepare(names: list[str]) -> None:
        shutil.rmtree(paths.local_dataset, ignore_errors=True)
        paths.calibration.unlink(missing_ok=True)  # as prepare_data does
        train = paths.local_dataset / "train"
        train.mkdir(parents=True)
        for name in names:
            Image.new("RGB", (32, 24)).save(train / f"{name}.jpg")
            (train / f"{name}.txt").write_text("")

    paths.work.mkdir(parents=True)
    prepare(["2025-05-04-02-42-56-529894", "2025-05-04-02-42-57-529894"])
    calibration = export.calibration_array(paths)
    assert np.load(calibration).shape == (2, 672, 896, 3)
    listing = (paths.export_dir / "calibration.txt").read_text()

    # A new session re-prepares the same dataset plus a frame; compile keeps the export's frames.
    prepare(
        ["2025-05-04-02-42-56-529894", "2025-05-04-02-42-57-529894", "2025-06-01-10-00-00-000000"]
    )
    assert np.load(export.calibration_array(paths)).shape == (2, 672, 896, 3)
    assert (paths.export_dir / "calibration.txt").read_text() == listing

    prepare(["2025-06-01-10-00-00-000000"])
    with pytest.raises(SystemExit, match="not in the prepared dataset"):
        export.calibration_array(paths)


def test_compile_requires_a_finished_export(paths: Paths) -> None:
    paths.toolchain.mkdir(parents=True)
    (paths.toolchain / "hailo_dataflow_compiler-3.0-py3-none-linux_x86_64.whl").write_bytes(b"w")
    (paths.export_dir / "model.onnx").write_bytes(b"onnx")
    with pytest.raises(SystemExit, match="run the export step first"):
        pipeline.compile_hef(paths)


def test_compile_reinstalls_a_replaced_wheel_only_after_success(paths: Paths) -> None:
    from unittest.mock import patch

    wheel = paths.toolchain / "hailo_dataflow_compiler-3.0-py3-none-linux_x86_64.whl"
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"wheel-a")
    python = paths.work / "hailo-venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    installs = []

    def fake_run(command: list[str], **_: object) -> None:
        if command[:3] == ["uv", "pip", "install"]:
            installs.append(command[-1])
            if wheel.read_bytes() == b"broken":
                raise RuntimeError("install failed")

    def compile_once() -> None:
        with (
            patch("export.export_complete", return_value=True),
            patch("export.calibration_array", return_value=paths.calibration),
            patch("export.write_manifest"),
            patch("pipeline.shutil.which", return_value="/usr/bin/dot"),
            patch("pipeline.run", side_effect=fake_run),
        ):
            pipeline.compile_hef(paths)

    compile_once()
    compile_once()
    assert len(installs) == 1

    wheel.write_bytes(b"wheel-b")  # same name and size, different content
    compile_once()
    assert len(installs) == 2

    wheel.write_bytes(b"broken")
    with pytest.raises(RuntimeError):
        compile_once()
    with pytest.raises(RuntimeError):
        compile_once()  # the failed install is retried, not recorded
    assert len(installs) == 4
