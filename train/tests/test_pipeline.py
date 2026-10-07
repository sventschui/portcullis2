from __future__ import annotations

import re
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

import pipeline
from pipeline import Paths


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    return Paths(
        root=tmp_path / "drive", run="v1", work=tmp_path / "work", dataset=tmp_path / "cats"
    )


def write_sample(folder: Path, name: str, label: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.jpg").write_bytes(b"\xff\xd8")
    (folder / f"{name}.txt").write_text(label, encoding="utf-8")


def frame(second: int, augmentation: int | None = None) -> str:
    """Capture-time frame name; frames more than 30 s apart belong to different clips."""
    minutes, seconds = divmod(second, 60)
    hours, minutes = divmod(minutes, 60)
    suffix = "" if augmentation is None else f"_aug{augmentation}"
    return f"2025-05-04-{hours:02}-{minutes:02}-{seconds:02}-000000{suffix}"


def test_cfg_has_two_heads_matching_the_names() -> None:
    names = pipeline.NAMES.read_text(encoding="utf-8").split()
    assert tuple(names) == pipeline.CLASS_NAMES
    sections = re.split(r"(?m)^(?=\[)", pipeline.CFG_TEMPLATE.read_text(encoding="utf-8"))
    heads = [index for index, section in enumerate(sections) if section.startswith("[yolo]")]
    assert len(heads) == 2
    for index in heads:
        assert re.search(r"(?m)^classes=6$", sections[index])
        assert sections[index - 1].startswith("[convolutional]")
        assert re.search(r"(?m)^filters=33$", sections[index - 1])
    assert re.search(r"(?m)^letter_box=1$", sections[1])


def test_prepare_data_unpacks_zip_and_writes_absolute_lists(paths: Paths, tmp_path: Path) -> None:
    source = tmp_path / "zip-source"
    write_sample(source / "set01", frame(0), "3 0.5 0.5 0.2 0.2\n6 0.4 0.6 0.1 0.1\n")
    write_sample(source / "set01_augmented", frame(0, 0), "3 0.5 0.5 0.2 0.2\n")
    write_sample(source / "set01", frame(100), "3 0.5 0.5 0.2 0.2\n6 0.4 0.6 0.1 0.1\n")
    write_sample(source / "__MACOSX" / "set01", f"._{frame(100)}", "")
    paths = Paths(paths.root, paths.run, paths.work, tmp_path / "cats.zip")
    with zipfile.ZipFile(paths.dataset, "w") as archive:
        for file in source.rglob("*.*"):
            archive.write(file, file.relative_to(source))
    paths.work.mkdir(parents=True)
    paths.calibration.write_bytes(b"previous dataset")

    pipeline.prepare_data(paths)

    train = (paths.work / "train.txt").read_text(encoding="utf-8").splitlines()
    assert train == [
        str(paths.local_dataset / "train" / "set01" / f"{frame(0)}.jpg"),
        str(paths.local_dataset / "train" / "set01_augmented" / f"{frame(0, 0)}.jpg"),
    ]
    valid = (paths.work / "valid.txt").read_text(encoding="utf-8").splitlines()
    assert valid == [str(paths.local_dataset / "valid" / "set01" / f"{frame(100)}.jpg")]
    # Legacy class ids are shifted down by one past the dropped plain "cat" class.
    label = paths.local_dataset / "valid" / "set01" / f"{frame(100)}.txt"
    assert label.read_text(encoding="utf-8") == "2 0.5 0.5 0.2 0.2\n5 0.4 0.6 0.1 0.1\n"
    data = paths.data_file.read_text(encoding="utf-8")
    assert "classes = 6" in data
    assert f"backup = {paths.backup}" in data
    assert paths.backup.is_dir()
    assert not paths.calibration.exists()


def test_split_keeps_clips_together_and_covers_rare_classes(tmp_path: Path) -> None:
    images = []
    # Six no-prey clips, then two mouse clips; frames 10 s apart, clips 100 s apart.
    for clip, label in enumerate(["3"] * 6 + ["1"] * 2):
        for offset in (0, 10):
            for augmentation in (None, 0):
                name = frame(clip * 100 + offset, augmentation)
                write_sample(tmp_path, name, f"{label} 0.5 0.5 0.2 0.2\n")
                images.append(tmp_path / f"{name}.jpg")

    split = pipeline.split_clips(images)

    valid = {image.stem for image in split["valid"]}
    # The newest clip of each group is held out, without its augmentations.
    assert valid == {frame(500), frame(510), frame(700), frame(710)}
    train = {image.stem for image in split["train"]}
    assert not valid & train
    assert train == {image.stem for image in images} - valid - {f"{stem}_aug0" for stem in valid}


@pytest.mark.parametrize(
    ("name", "error"),
    [("frame-1", "capture-time name"), (frame(0, 1), "without their original")],
)
def test_split_rejects_unknown_frame_names(tmp_path: Path, name: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        pipeline.split_clips([tmp_path / f"{name}.jpg"])


@pytest.mark.parametrize(
    ("label", "error"),
    [
        (None, "missing label"),
        ("0 0.5 0.5 0.2 0.2\n", "class 0"),
        ("7 0.5 0.5 0.2 0.2\n", "class 7"),
        ("1 0.5 0.5 0.2\n", "expected"),
        ("1 0.5 0.5 1.2 0.2\n", "normalized"),
    ],
)
def test_prepare_data_rejects_bad_labels(paths: Paths, label: str | None, error: str) -> None:
    write_sample(paths.dataset, frame(0), label or "")
    if label is None:
        (paths.dataset / f"{frame(0)}.txt").unlink()
    write_sample(paths.dataset, frame(100), "")
    with pytest.raises(ValueError, match=error):
        pipeline.prepare_data(paths)


def test_resume_prefers_last_then_newest_numbered_then_pretrained(paths: Paths) -> None:
    paths.backup.mkdir(parents=True)
    assert pipeline.resume_weights(paths) == paths.pretrained

    for name in ("portcullis_1000.weights", "portcullis_3000.weights", "portcullis_best.weights"):
        (paths.backup / name).write_bytes(b"w")
    assert pipeline.resume_weights(paths) == paths.backup / "portcullis_3000.weights"

    (paths.backup / "portcullis_last.weights").write_bytes(b"")  # empty = interrupted write
    assert pipeline.resume_weights(paths) == paths.backup / "portcullis_3000.weights"

    (paths.backup / "portcullis_last.weights").write_bytes(b"w")
    assert pipeline.resume_weights(paths) == paths.backup / "portcullis_last.weights"


def test_darknet_command(paths: Paths) -> None:
    weights = paths.backup / "portcullis_last.weights"
    assert pipeline.darknet_train_command(paths, weights) == [
        str(paths.darknet),
        "detector",
        "train",
        str(paths.data_file),
        str(paths.cfg),
        str(weights),
        "-dont_show",
        "-map",
    ]


def test_run_cfg_is_frozen_on_first_use(paths: Paths) -> None:
    pipeline.write_run_cfg(paths, max_batches=200)
    cfg = paths.cfg.read_text(encoding="utf-8")
    assert re.search(r"(?m)^max_batches = 200$", cfg)
    assert re.search(r"(?m)^steps=160,180$", cfg)

    pipeline.write_run_cfg(paths, max_batches=None)
    assert paths.cfg.read_text(encoding="utf-8") == cfg


def test_finished_training_is_not_rerun(paths: Paths) -> None:
    paths.backup.mkdir(parents=True)
    (paths.backup / "portcullis_final.weights").write_bytes(b"w")
    with patch("pipeline.subprocess.Popen") as popen:
        pipeline.train(paths, max_batches=None)
    popen.assert_not_called()


def test_setup_skips_existing_build_and_download(paths: Paths) -> None:
    paths.darknet.parent.mkdir(parents=True)
    paths.darknet.write_bytes(b"binary")
    paths.darknet_onnx_export.write_bytes(b"binary")
    paths.pretrained.write_bytes(b"weights")
    with (
        patch("pipeline.sha256_file", return_value=pipeline.PRETRAINED_SHA256),
        patch("pipeline.urllib.request.urlretrieve") as download,
        patch("pipeline.run") as run,
        patch("pipeline.require_cuda_gpu") as gpu,
    ):
        pipeline.setup(paths)
    download.assert_not_called()
    gpu.assert_not_called()
    assert all("cmake" not in call.args[0] for call in run.call_args_list)


def fake_install(prefix: Path, content: bytes) -> None:
    for relative in ("bin/darknet", "bin/darknet_onnx_export", "lib/libdarknet.so"):
        (prefix / relative).parent.mkdir(parents=True, exist_ok=True)
        (prefix / relative).write_bytes(content)


def stub_setup_downloads(paths: Paths) -> None:
    paths.pretrained.parent.mkdir(parents=True, exist_ok=True)
    paths.pretrained.write_bytes(b"weights")


def test_setup_restores_cached_build_instead_of_compiling(paths: Paths) -> None:
    stub_setup_downloads(paths)
    cached = paths.build_cache / "darknet-tag"
    fake_install(cached, b"binary")
    with (
        patch("pipeline.sha256_file", return_value=pipeline.PRETRAINED_SHA256),
        patch("pipeline.checkout"),
        patch("pipeline.require_cuda_gpu", return_value=("Tesla T4", "75")),
        patch("pipeline.build_tag", return_value="darknet-tag"),
        patch("pipeline.binary_links_resolve", return_value=True),
        patch("pipeline.run") as run,
    ):
        pipeline.setup(paths)
    assert paths.darknet.read_bytes() == b"binary"
    assert (paths.darknet_install / "lib" / "libdarknet.so").read_bytes() == b"binary"
    run.assert_not_called()


def test_setup_rebuilds_and_caches_when_cached_build_no_longer_links(paths: Paths) -> None:
    stub_setup_downloads(paths)
    cached = paths.build_cache / "darknet-tag"
    fake_install(cached, b"stale")
    (cached / "bin" / "leftover").write_bytes(b"stale")

    def fake_cmake(command: list[str], *, cwd: Path | None = None) -> None:
        if command[:2] == ["cmake", "--install"]:
            fake_install(paths.darknet_install, b"fresh")

    with (
        patch("pipeline.sha256_file", return_value=pipeline.PRETRAINED_SHA256),
        patch("pipeline.checkout"),
        patch("pipeline.require_cuda_gpu", return_value=("Tesla T4", "75")),
        patch("pipeline.require_build_dependencies"),
        patch("pipeline.build_tag", return_value="darknet-tag"),
        patch("pipeline.binary_links_resolve", return_value=False),
        patch("pipeline.run", side_effect=fake_cmake),
    ):
        pipeline.setup(paths)
    assert paths.darknet.read_bytes() == b"fresh"
    assert (cached / "bin" / "darknet_onnx_export").read_bytes() == b"fresh"
    assert not (cached / "bin" / "leftover").exists()


def test_build_tag_separates_gpu_architectures() -> None:
    t4 = pipeline.build_tag("Tesla T4", "75")
    assert "tesla-t4-sm75" in t4
    assert t4 != pipeline.build_tag("NVIDIA L4", "89")


def test_gpu_check_explains_how_to_pick_a_gpu_runtime() -> None:
    with (
        patch("pipeline.shutil.which", return_value=None),
        pytest.raises(SystemExit, match="T4 GPU"),
    ):
        pipeline.require_cuda_gpu()


def test_unknown_step_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        pipeline.main(
            ["--root", str(tmp_path), "--run", "v1", "--dataset", str(tmp_path), "--steps", "nope"]
        )


def test_main_resolves_relative_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with patch("pipeline.setup") as setup:
        pipeline.main(
            ["--root", "drive", "--run", "v1", "--work", "work", "--dataset", "cats.zip",
             "--steps", "setup"]
        )  # fmt: skip
    paths = setup.call_args.args[0]
    assert (paths.root, paths.work, paths.dataset) == (
        tmp_path.resolve() / "drive",
        tmp_path.resolve() / "work",
        tmp_path.resolve() / "cats.zip",
    )
