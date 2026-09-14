#!/usr/bin/env python3
"""Prepare ignored local asset directories and verify extracted baseline models."""

import argparse
import hashlib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_HASHES = {
    "df_arena_1b/pytorch_model.bin": (
        "780bc14fd4c15e65d58efdef728427cf03cd29cd60be528e97badf8c89087988"
    ),
    "htdemucs/955717e8-8726e21a.th": (
        "8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4"
    ),
    "panns/Cnn14_mAP=0.431.pth": (
        "0dc499e40e9761ef5ea061ffc77697697f277f6a960894903df3ada000e34b31"
    ),
}
DATA_DIRECTORIES = ("incoming", "train", "validation", "test")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare local DeepVoice assets excluded from Git."
    )
    parser.add_argument("--model-dir", type=Path, default=PROJECT_DIR / "model")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "data")
    parser.add_argument("--prepare-data-dirs", action="store_true")
    parser.add_argument("--check-model-hashes", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_data_directories(data_dir: Path) -> None:
    for directory_name in DATA_DIRECTORIES:
        (data_dir / directory_name).mkdir(parents=True, exist_ok=True)
    print(f"Prepared local data directories under {data_dir}")


def check_model_hashes(model_dir: Path) -> None:
    failures = []
    for relative_path, expected_hash in MODEL_HASHES.items():
        path = model_dir / relative_path
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            failures.append(f"checksum mismatch: {path}")
            continue
        print(f"OK: {relative_path}")
    if failures:
        raise RuntimeError("Baseline model validation failed:\n" + "\n".join(failures))


def main() -> None:
    args = parse_arguments()
    if not args.prepare_data_dirs and not args.check_model_hashes:
        raise SystemExit("Select --prepare-data-dirs, --check-model-hashes, or both.")
    if args.prepare_data_dirs:
        prepare_data_directories(args.data_dir)
    if args.check_model_hashes:
        check_model_hashes(args.model_dir)


if __name__ == "__main__":
    main()
