#!/usr/bin/env python3
"""Create a source-only archive; never include local data, credentials or history."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT_FILES = (
    "README.md", "CONTRIBUTING.md", "THIRD_PARTY.md", ".gitignore", ".env.example",
    "runtime_paths.env.example", "requirements.txt", "requirements-data.txt",
    "requirements-serve.txt", "compose.yaml",
)
SOURCE_DIRS = ("harness_search", "runtime", "runner", "datagen", "dataset_runs", "scripts", "tests", "docs", "licenses", ".github")
SUFFIXES = {".py", ".sh", ".md", ".txt", ".yaml", ".yml"}


def source_files(root: Path):
    paths = [root / name for name in ROOT_FILES if (root / name).exists()]
    for name in SOURCE_DIRS:
        folder = root / name
        if folder.is_symlink():
            raise ValueError(f"Refusing source directory symlink: {folder}")
        for path in folder.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Refusing source symlink: {path}")
            if path.is_file() and path.suffix in SUFFIXES and not any(
                part.startswith("._") or part in {"__pycache__", ".pytest_cache"}
                for part in path.relative_to(root).parts
            ):
                paths.append(path)
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Refusing source symlink: {path}")
    return sorted(paths)


def export(root: Path, output: Path):
    files = source_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in files
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in files:
            info = archive.gettarinfo(str(path), arcname=str(Path("Harness-Search") / path.relative_to(root)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mode = 0o755 if path.suffix == ".sh" else 0o644
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    temporary.replace(output)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path, len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("dist/Harness-Search-source.tar.gz"))
    args = parser.parse_args()
    manifest, count = export(Path(__file__).resolve().parents[1], args.output.resolve())
    print(f"Exported {count} source files to {args.output}; manifest: {manifest}")


if __name__ == "__main__":
    main()
