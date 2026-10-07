"""Train the Portcullis YOLOv4-tiny detector on Colab and export its release artifacts.

Every step is idempotent: after a Colab disconnect, rerun the same command and
finished work is skipped while Darknet resumes from the newest weights on Drive.
Only ``--root`` (Google Drive) is persistent; ``--work`` is expendable scratch.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent
CFG_TEMPLATE = TRAIN_DIR / "darknet" / "yolov4-tiny-portcullis.cfg"
NAMES = TRAIN_DIR / "darknet" / "portcullis.names"
ALLS = TRAIN_DIR / "hailo" / "portcullis.alls"

DARKNET_REPOSITORY = "https://github.com/hank-ai/darknet.git"
DARKNET_REVISION = "f64b531e394daefd3cb8c41e760ffe598b2d6141"  # v6.0
PRETRAINED_URL = (
    "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29"
)
PRETRAINED_SHA256 = "3c794b8420b12ed2a609daa71d075880e7bd152671323f5eb3e8d9e57e31785b"

# The legacy cats.zip labels in their original order, without its unused plain "cat"
# class 0. Dataset labels keep the legacy ids and are shifted down by one when copied.
LEGACY_FIRST_CLASS = 1
CLASS_NAMES = (
    "cat_with_mouse",
    "cat_with_worm",
    "cat_without_prey",
    "catface_with_mouse",
    "catface_with_worm",
    "catface_without_prey",
)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
SPLITS = ("train", "valid")
# Frames are named after their capture time; offline augmentations add an _aug<N> suffix.
FRAME_NAME = re.compile(r"(\d{4}(?:-\d{2}){5}-\d{6})(_aug\d+)?")
FRAME_TIME = "%Y-%m-%d-%H-%M-%S-%f"
FRAME_EXAMPLE = "2025-05-04-02-42-56-529894"
CLIP_GAP_SECONDS = 30
VALID_FRACTION = 0.2
STEPS = ("setup", "data", "train", "export", "compile")


@dataclass(frozen=True)
class Paths:
    root: Path
    run: str
    work: Path
    dataset: Path  # zip file or folder of frames with YOLO labels

    @property
    def toolchain(self) -> Path:
        return self.root / "toolchain"

    @property
    def build_cache(self) -> Path:
        return self.root / "cache"

    @property
    def run_dir(self) -> Path:
        return self.root / "runs" / self.run

    @property
    def backup(self) -> Path:
        return self.run_dir / "backup"

    @property
    def export_dir(self) -> Path:
        return self.run_dir / "export"

    @property
    def cfg(self) -> Path:
        # Weights are named after the cfg stem: portcullis_last.weights etc.
        return self.run_dir / "portcullis.cfg"

    @property
    def darknet_source(self) -> Path:
        return self.work / "darknet-src"

    @property
    def darknet_install(self) -> Path:
        # bin/ holds darknet and darknet_onnx_export, which load lib/libdarknet.so.
        return self.work / "darknet"

    @property
    def darknet(self) -> Path:
        return self.darknet_install / "bin" / "darknet"

    @property
    def darknet_onnx_export(self) -> Path:
        return self.darknet_install / "bin" / "darknet_onnx_export"

    @property
    def pretrained(self) -> Path:
        return self.work / "yolov4-tiny.conv.29"

    @property
    def unpacked_dataset(self) -> Path:
        return self.work / "source"

    @property
    def local_dataset(self) -> Path:
        return self.work / "dataset"

    @property
    def data_file(self) -> Path:
        return self.work / "portcullis.data"

    @property
    def calibration(self) -> Path:
        # Derived from local_dataset, so prepare_data discards it along with the dataset.
        return self.work / "calibration.npy"


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """Write via a sibling .partial file and rename, so a disconnect never leaves it truncated."""
    partial = path.with_name(path.name + ".partial")
    write(partial)
    partial.replace(path)


# --- setup -------------------------------------------------------------------


def require_cuda_gpu() -> tuple[str, str]:
    """Return the GPU ``(name, compute capability)``, e.g. ``("Tesla T4", "75")``.

    Darknet cannot train on TPU/CPU.
    """
    smi = shutil.which("nvidia-smi")
    result = (
        subprocess.run(
            [smi, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if smi
        else None
    )
    if result is None or result.returncode != 0 or not result.stdout.strip():
        raise SystemExit(
            "No NVIDIA GPU found. In Colab choose Runtime > Change runtime type > T4 GPU "
            "(TPU and CPU runtimes cannot run Darknet), then Run all again."
        )
    name, capability = (field.strip() for field in result.stdout.splitlines()[0].split(","))
    print(f"GPU: {name} (compute capability {capability})")
    return name, capability.replace(".", "")


def require_build_dependencies() -> None:
    """Darknet needs the OpenCV dev headers, and protobuf for its ONNX export tool.

    Colab can have a ``protoc`` on PATH without the protobuf headers, so check the packages.
    """
    packages = ["libopencv-dev", "libprotobuf-dev", "protobuf-compiler"]
    installed = subprocess.run(
        ["dpkg-query", "-W", "-f=${Status}\n", *packages], capture_output=True, text=True
    )
    if installed.stdout.count("install ok installed") == len(packages):
        return
    run(["apt-get", "install", "-y", "-qq", *packages])


def checkout(repository: str, revision: str, destination: Path) -> None:
    if not (destination / ".git").is_dir():
        run(["git", "clone", "--filter=blob:none", repository, str(destination)])
    run(["git", "-C", str(destination), "checkout", "--quiet", "--detach", revision])


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError:
        return ""
    return result.stdout if result.returncode == 0 else ""


def build_tag(gpu_name: str, capability: str) -> str:
    """Name a Darknet build after everything its binary depends on.

    The library holds machine code for one GPU architecture and links dynamically against
    the runtime's CUDA, cuDNN, OpenCV and protobuf, so a cached build is only valid when the
    GPU, CUDA toolkit and OS match; ``binary_links_resolve`` catches the remaining drift.
    """
    os_release = command_output(["cat", "/etc/os-release"])
    version = re.search(r'(?m)^VERSION_ID="?([^"\n]+)', os_release)
    ubuntu = version.group(1) if version else "unknown"
    nvcc = re.search(r"release (\d+\.\d+)", command_output(["nvcc", "--version"]))
    gpu = re.sub(r"[^a-z0-9]+", "-", gpu_name.lower()).strip("-")
    return (
        f"darknet-{DARKNET_REVISION[:8]}-{gpu}-sm{capability}"
        f"-cuda{nvcc.group(1) if nvcc else 'unknown'}-ubuntu{ubuntu}"
    )


def binary_links_resolve(binary: Path) -> bool:
    """A cached binary from an older Colab image fails to start on missing shared libraries."""
    listing = command_output(["ldd", str(binary)])
    return bool(listing) and "not found" not in listing


def restore_cached_build(cached: Path, paths: Paths) -> bool:
    if not cached.is_dir():
        return False
    shutil.rmtree(paths.darknet_install, ignore_errors=True)
    shutil.copytree(cached, paths.darknet_install)
    # Google Drive does not preserve the executable bit, so restore it on the copied binaries.
    for binary in (paths.darknet_install / "bin").iterdir():
        binary.chmod(binary.stat().st_mode | 0o111)
    if all(binary_links_resolve(binary) for binary in (paths.darknet, paths.darknet_onnx_export)):
        print(f"Restored cached Darknet build: {cached}")
        return True
    print(f"Cached Darknet build {cached.name} no longer links; rebuilding")
    shutil.rmtree(paths.darknet_install)
    return False


def store_cached_build(paths: Paths, cached: Path) -> None:
    cached.parent.mkdir(parents=True, exist_ok=True)
    partial = cached.with_name(cached.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    shutil.copytree(paths.darknet_install, partial)
    # A directory cannot be renamed over a non-empty one; drop the stale build first.
    shutil.rmtree(cached, ignore_errors=True)
    partial.rename(cached)
    print(f"Cached Darknet build: {cached}")


def setup(paths: Paths) -> None:
    paths.work.mkdir(parents=True, exist_ok=True)
    if not paths.pretrained.is_file() or sha256_file(paths.pretrained) != PRETRAINED_SHA256:
        print(f"Downloading {PRETRAINED_URL}")
        urllib.request.urlretrieve(PRETRAINED_URL, paths.pretrained)
        if sha256_file(paths.pretrained) != PRETRAINED_SHA256:
            raise SystemExit(f"Checksum mismatch for {paths.pretrained}")
    if paths.darknet.is_file() and paths.darknet_onnx_export.is_file():
        print(f"Darknet already built: {paths.darknet_install}")
        return
    gpu_name, capability = require_cuda_gpu()
    # A cached binary links against the OpenCV and protobuf shared libraries these packages
    # provide, so they must be installed before `ldd` judges the restored build.
    require_build_dependencies()
    cached = paths.build_cache / build_tag(gpu_name, capability)
    if restore_cached_build(cached, paths):
        return
    source = paths.darknet_source
    checkout(DARKNET_REPOSITORY, DARKNET_REVISION, source)
    build = source / "build"
    # Darknet's CMake derives its version from `git describe` in the current directory.
    run(
        [
            "cmake", "-S", str(source), "-B", str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            # "native" would also work, but naming the architecture matches the cache key.
            f"-DDARKNET_CUDA_ARCHITECTURES={capability}",
            "-DDARKNET_TRY_ROCM=OFF",
            "-DDARKNET_TRY_ONNX=ON",
            f"-DCMAKE_INSTALL_PREFIX={paths.darknet_install}",
            # Installed binaries find libdarknet.so next to them, wherever the tree is copied.
            "-DCMAKE_INSTALL_RPATH=$ORIGIN/../lib",
            # The distro protoc must match the distro headers, not another protoc on PATH.
            "-DProtobuf_PROTOC_EXECUTABLE=/usr/bin/protoc",
        ],
        cwd=source,
    )  # fmt: skip
    run(["cmake", "--build", str(build), "--parallel", "4"])
    run(["cmake", "--install", str(build)])
    # CMake silently skips the ONNX tool when it cannot find protobuf.
    if not paths.darknet_onnx_export.is_file():
        raise SystemExit("Darknet built without darknet_onnx_export; see the CMake output")
    store_cached_build(paths, cached)


# --- data --------------------------------------------------------------------


def check_label(label: Path, first_class: int = 0) -> None:
    """Reject labels Darknet would silently misread. An empty file is a negative image."""
    for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            if len(fields) != 5:
                raise ValueError
            class_id = int(fields[0])
            x, y, width, height = (float(value) for value in fields[1:])
        except ValueError:
            raise ValueError(f"{label}:{number}: expected '<class> <x> <y> <w> <h>'") from None
        if class_id - first_class not in range(len(CLASS_NAMES)):
            classes = range(first_class, first_class + len(CLASS_NAMES))
            raise ValueError(
                f"{label}:{number}: class {class_id} is not in {classes.start}–{classes.stop - 1}"
            )
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise ValueError(f"{label}:{number}: box is not normalized to [0, 1]")


def list_images(folder: Path, first_class: int = 0) -> list[Path]:
    images = sorted(
        path
        for path in folder.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
        and not any(part.startswith((".", "__MACOSX")) for part in path.relative_to(folder).parts)
    )
    if not images:
        raise ValueError(f"no images in {folder}")
    for image in images:
        label = image.with_suffix(".txt")
        if not label.is_file():
            raise ValueError(f"missing label {label}")
        check_label(label, first_class)
    return images


def is_augmented(image: Path) -> bool:
    match = FRAME_NAME.fullmatch(image.stem)
    return bool(match and match.group(2))


def relabel(label: Path, destination: Path) -> None:
    """Copy a legacy label, shifting its class ids to the model's."""
    lines = []
    for line in label.read_text(encoding="utf-8").splitlines():
        if line.strip():
            class_id, *box = line.split()
            lines.append(" ".join([str(int(class_id) - LEGACY_FIRST_CLASS), *box]) + "\n")
    destination.write_text("".join(lines), encoding="utf-8")


def label_classes(image: Path) -> list[int]:
    lines = image.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
    return [int(line.split()[0]) for line in lines if line.strip()]


def split_clips(images: list[Path]) -> dict[str, list[Path]]:
    """Split by capture clip so adjacent frames and their augmentations never straddle sets.

    Clips are grouped by their rarest class so each class reaches validation, and the
    newest clips of each group are held out. Augmented frames are training-only.
    """
    originals: dict[str, Path] = {}
    augmented: dict[str, list[Path]] = {}
    for image in images:
        match = FRAME_NAME.fullmatch(image.stem)
        if match is None:
            raise ValueError(f"{image.name}: expected a capture-time name like {FRAME_EXAMPLE}")
        if match.group(2):
            augmented.setdefault(match.group(1), []).append(image)
        elif match.group(1) in originals:
            raise ValueError(f"{image.name}: duplicate frame {originals[match.group(1)]}")
        else:
            originals[match.group(1)] = image
    if orphans := sorted(set(augmented) - set(originals)):
        raise ValueError(f"augmented frames without their original: {', '.join(orphans[:5])}")

    clips: list[list[str]] = []
    previous = None
    for stamp in sorted(originals):
        captured = datetime.strptime(stamp, FRAME_TIME)
        if previous is None or (captured - previous).total_seconds() > CLIP_GAP_SECONDS:
            clips.append([])
        clips[-1].append(stamp)
        previous = captured

    frequency = Counter(c for image in originals.values() for c in label_classes(image))
    groups: dict[int, list[list[str]]] = {}
    for clip in clips:
        classes = {c for stamp in clip for c in label_classes(originals[stamp])}
        rarest = min(classes, key=lambda c: (frequency[c], c), default=-1)  # -1: negatives only
        groups.setdefault(rarest, []).append(clip)

    split: dict[str, list[Path]] = {name: [] for name in SPLITS}
    for group in groups.values():
        held_out = max(1, round(len(group) * VALID_FRACTION)) if len(group) > 1 else 0
        for index, clip in enumerate(group):
            if index >= len(group) - held_out:
                split["valid"] += [originals[stamp] for stamp in clip]
            else:
                split["train"] += [originals[stamp] for stamp in clip]
                split["train"] += [a for stamp in clip for a in augmented.get(stamp, [])]
    return split


def unpack_dataset(paths: Paths) -> Path:
    if paths.dataset.is_dir():
        return paths.dataset
    if not zipfile.is_zipfile(paths.dataset):
        raise ValueError(f"{paths.dataset} is neither a folder nor a zip file")
    shutil.rmtree(paths.unpacked_dataset, ignore_errors=True)
    print(f"Unpacking {paths.dataset}")
    with zipfile.ZipFile(paths.dataset) as archive:
        archive.extractall(paths.unpacked_dataset)
    return paths.unpacked_dataset


def prepare_data(paths: Paths) -> None:
    """Split the dataset onto fast local disk and write the Darknet .data/.txt lists."""
    if NAMES.read_text(encoding="utf-8").split() != list(CLASS_NAMES):
        raise ValueError(f"{NAMES} must list {CLASS_NAMES} in order")
    paths.work.mkdir(parents=True, exist_ok=True)
    paths.backup.mkdir(parents=True, exist_ok=True)
    source = unpack_dataset(paths)
    shutil.rmtree(paths.local_dataset, ignore_errors=True)
    paths.calibration.unlink(missing_ok=True)
    lists = {}
    for split, images in split_clips(list_images(source, LEGACY_FIRST_CLASS)).items():
        if not images:
            raise ValueError(f"the dataset split left {split} empty")
        copies = []
        for image in images:
            destination = paths.local_dataset / split / image.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, destination)
            relabel(image.with_suffix(".txt"), destination.with_suffix(".txt"))
            copies.append(destination)
        lists[split] = paths.work / f"{split}.txt"
        lists[split].write_text("".join(f"{image}\n" for image in sorted(copies)), encoding="utf-8")
        boxes = Counter(c for image in copies for c in label_classes(image))
        summary = ", ".join(f"{name} {boxes[index]}" for index, name in enumerate(CLASS_NAMES))
        augmented = sum(map(is_augmented, images))
        print(f"{split}: {len(images)} images ({augmented} augmented); boxes: {summary}")
    paths.data_file.write_text(
        f"classes = {len(CLASS_NAMES)}\n"
        f"train = {lists['train']}\n"
        f"valid = {lists['valid']}\n"
        f"names = {NAMES}\n"
        f"backup = {paths.backup}\n",
        encoding="utf-8",
    )


# --- train -------------------------------------------------------------------


def write_run_cfg(paths: Paths, max_batches: int | None) -> None:
    """Freeze the cfg into the run folder on first use so a resumed run cannot drift."""
    if paths.cfg.exists():
        if max_batches is not None:
            print(f"Ignoring --max-batches: {paths.cfg} already exists for this run")
        return
    text = CFG_TEMPLATE.read_text(encoding="utf-8")
    if max_batches is not None:
        text = re.sub(r"(?m)^max_batches\s*=.*$", f"max_batches = {max_batches}", text)
        steps = f"{max_batches * 8 // 10},{max_batches * 9 // 10}"
        text = re.sub(r"(?m)^steps\s*=.*$", f"steps={steps}", text)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.cfg.write_text(text, encoding="utf-8")


def resume_weights(paths: Paths) -> Path:
    """Prefer Darknet's rolling _last checkpoint, then the newest numbered one, then pretrained."""
    last = paths.backup / "portcullis_last.weights"
    if last.is_file() and last.stat().st_size > 0:
        return last
    numbered = []
    for path in paths.backup.glob("portcullis_*.weights"):
        match = re.fullmatch(r"portcullis_(\d+)\.weights", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if numbered:
        return max(numbered)[1]
    return paths.pretrained


def darknet_train_command(paths: Paths, weights: Path) -> list[str]:
    return [
        str(paths.darknet),
        "detector",
        "train",
        str(paths.data_file),
        str(paths.cfg),
        str(weights),
        "-dont_show",
        "-map",
    ]


def train(paths: Paths, max_batches: int | None) -> None:
    final = paths.backup / "portcullis_final.weights"
    if final.is_file():
        print(f"Training already finished: {final}")
        return
    write_run_cfg(paths, max_batches)
    weights = resume_weights(paths)
    print(f"Starting from {weights}")
    command = darknet_train_command(paths, weights)
    print("+", " ".join(command), flush=True)
    # Run in the run folder so Darknet's chart.png lands on Drive; tee output to train.log.
    with (paths.run_dir / "train.log").open("ab") as log:
        process = subprocess.Popen(
            command, cwd=paths.run_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.read1(8192), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            log.write(chunk)
            log.flush()
    if process.wait() != 0:
        raise SystemExit(f"Darknet exited with {process.returncode}; rerun to resume")
    if not final.is_file():
        raise SystemExit(f"Darknet finished without writing {final}")


# --- compile -----------------------------------------------------------------


def compile_hef(paths: Paths) -> None:
    import export  # heavy dependencies; only needed from here on

    hef = paths.export_dir / "model.hef"
    if hef.is_file():
        print(f"HEF already compiled: {hef}")
        return
    wheels = sorted(paths.toolchain.glob("hailo_dataflow_compiler-*.whl"))
    if not wheels:
        print(
            f"Skipping HEF compilation: put the licensed Hailo Dataflow Compiler wheel "
            f"(hailo_dataflow_compiler-*.whl from the Hailo Developer Zone) in {paths.toolchain}"
        )
        return
    if len(wheels) > 1:
        raise SystemExit(f"Keep exactly one Dataflow Compiler wheel in {paths.toolchain}")
    if not export.export_complete(paths):
        raise SystemExit(f"No finished export in {paths.export_dir}; run the export step first")
    venv = paths.work / "hailo-venv"
    python = venv / "bin" / "python"
    if not python.is_file():
        # The DFC supports Python 3.10 only, so it gets its own environment.
        run(["uv", "venv", "--python", "3.10", str(venv)])
    # Recorded only after a successful install, so a failed install or a replaced wheel
    # is (re)installed on the next run instead of compiling with a stale or missing DFC.
    installed = venv / "installed-wheel"
    wheel_id = f"{wheels[0].name} {sha256_file(wheels[0])}"
    if not installed.is_file() or installed.read_text(encoding="utf-8") != wheel_id:
        if shutil.which("dot") is None:  # pygraphviz, a DFC dependency, builds against graphviz
            run(["apt-get", "install", "-y", "-qq", "graphviz", "graphviz-dev"])
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--reinstall-package",
                "hailo_dataflow_compiler",
                str(wheels[0]),
            ]
        )
        installed.write_text(wheel_id, encoding="utf-8")
    calibration = export.calibration_array(paths)
    # The DFC compiler reads $USER, which Colab (running as root) leaves unset.
    env = {**os.environ, "USER": os.environ.get("USER") or getpass.getuser()}
    run(
        [
            str(python),
            str(TRAIN_DIR / "hailo_compile.py"),
            "--onnx",
            str(paths.export_dir / "model.onnx"),
            "--calibration",
            str(calibration),
            "--alls",
            str(ALLS),
            "--output",
            str(hef),
        ],
        env=env,
    )
    export.write_manifest(paths)


# --- main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True, help="persistent folder (Drive)")
    parser.add_argument("--run", required=True, help="run name, e.g. v1")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="zip file or folder of frames with YOLO labels, e.g. MyDrive/nn/cats.zip",
    )
    parser.add_argument("--work", type=Path, default=Path("/content/work"), help="scratch folder")
    parser.add_argument(
        "--steps",
        default=",".join(STEPS),
        help=f"comma-separated subset of {','.join(STEPS)}",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="override max_batches for a new run (smoke tests); ignored once the run exists",
    )
    args = parser.parse_args(argv)
    steps = args.steps.split(",")
    if unknown := set(steps) - set(STEPS):
        parser.error(f"unknown steps: {', '.join(sorted(unknown))}")
    # Training runs Darknet from the run folder, so relative paths would no longer resolve.
    paths = Paths(
        root=args.root.expanduser().resolve(),
        run=args.run,
        work=args.work.expanduser().resolve(),
        dataset=args.dataset.expanduser().resolve(),
    )

    if "setup" in steps:
        setup(paths)
    if "data" in steps:
        prepare_data(paths)
    if "train" in steps:
        train(paths, args.max_batches)
    if "export" in steps:
        import export  # heavy dependencies; keep setup/data/train stdlib-only

        export.export(paths)
    if "compile" in steps:
        compile_hef(paths)
    print(f"Artifacts: {paths.export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
