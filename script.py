#!/usr/bin/env python3
"""DACON submission entry point for DeepVoice inference."""

import argparse
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
BASE_DIR = Path(__file__).resolve().parent
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Production archives keep reusable source under model/deepvoice so their
# top-level layout remains model/, script.py, and requirements.txt only.
PACKAGED_SOURCE_DIR = BASE_DIR / "model"
if (PACKAGED_SOURCE_DIR / "deepvoice").is_dir():
    sys.path.insert(0, str(PACKAGED_SOURCE_DIR))

from deepvoice.fusion import FUSION_NAMES
from deepvoice.pipeline import run_all_fusions, run_inference

DEFAULT_FUSION = "baseline"


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DeepVoice offline inference.")
    parser.add_argument("--test-dir", type=Path, default=BASE_DIR / "data" / "test")
    parser.add_argument(
        "--sample-submission",
        type=Path,
        default=BASE_DIR / "data" / "sample_submission.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=BASE_DIR / "output" / "submission.csv"
    )
    parser.add_argument("--model-dir", type=Path, default=BASE_DIR / "model")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--fusion", choices=FUSION_NAMES, default=DEFAULT_FUSION)
    parser.add_argument(
        "--all-fusions",
        action="store_true",
        help="Write one CSV per fusion in the output directory after a shared inference pass.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    if args.all_fusions:
        run_all_fusions(
            test_dir=args.test_dir,
            sample_submission=args.sample_submission,
            output_dir=args.output.parent,
            model_dir=args.model_dir,
            device_name=args.device,
        )
        return
    run_inference(
        test_dir=args.test_dir,
        sample_submission=args.sample_submission,
        output_path=args.output,
        model_dir=args.model_dir,
        device_name=args.device,
        fusion_name=args.fusion,
    )


if __name__ == "__main__":
    main()
