#!/usr/bin/env python3
"""Create one DACON-compatible submit zip per file-level fusion strategy."""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

FUSION_NAMES = ("baseline", "component_max", "soft_or")
PROJECT_DIR = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DeepVoice DACON submissions.")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_DIR / "model")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "dist")
    return parser.parse_args()


def write_archive(source_dir: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))


def build_submission(model_dir: Path, output_dir: Path, fusion_name: str) -> Path:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"deepvoice_{fusion_name}_") as temporary:
        staging = Path(temporary)
        shutil.copytree(model_dir, staging / "model")
        shutil.copytree(PROJECT_DIR / "deepvoice", staging / "model" / "deepvoice")
        script_text = (PROJECT_DIR / "script.py").read_text(encoding="utf-8")
        script_text = script_text.replace(
            'DEFAULT_FUSION = "baseline"', f'DEFAULT_FUSION = "{fusion_name}"', 1
        )
        (staging / "script.py").write_text(script_text, encoding="utf-8")
        shutil.copy2(PROJECT_DIR / "requirements.txt", staging / "requirements.txt")
        archive_path = output_dir / f"submit_{fusion_name}.zip"
        write_archive(staging, archive_path)
    return archive_path


def main() -> None:
    args = parse_arguments()
    for fusion_name in FUSION_NAMES:
        print(f"Built {build_submission(args.model_dir, args.output_dir, fusion_name)}")


if __name__ == "__main__":
    main()
