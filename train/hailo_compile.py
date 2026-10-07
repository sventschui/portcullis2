"""Compile the raw-head ONNX into a Hailo-8L HEF with the Hailo Dataflow Compiler.

Runs inside the separate Python 3.10 environment that pipeline.py creates from the
licensed DFC wheel; it only depends on hailo_sdk_client and numpy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from hailo_sdk_client import ClientRunner


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True, help="NHWC float32 in [0, 1]")
    parser.add_argument("--alls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runner = ClientRunner(hw_arch="hailo8l")
    runner.translate_onnx_model(str(args.onnx), "portcullis")
    outputs = runner.get_hn_dict()["net_params"].get("output_layers_order")
    if outputs is not None and len(outputs) != 2:
        raise SystemExit(f"Expected the two raw YOLO heads as outputs, parsed {outputs}")

    # The model script normalizes on-chip, so calibrate with uint8-range RGB like the Pi feeds.
    calibration = np.load(args.calibration) * 255.0
    runner.load_model_script(args.alls.read_text(encoding="utf-8"))
    runner.optimize(calibration)
    runner.save_har(str(args.output.with_suffix(".har")))

    hef = runner.compile()
    temporary = args.output.with_suffix(".hef.partial")
    temporary.write_bytes(hef)
    temporary.replace(args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
