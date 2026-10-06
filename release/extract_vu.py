#!/usr/bin/env python3
"""Recover private VU images from a player's supported ELF.

The public recipe contains only coordinates and hashes. The images and all
source generated from them belong in the local installation, never a release.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def extract_images(elf: bytes, recipe: dict) -> dict[str, bytes]:
    if recipe.get("schema") != 1:
        raise ValueError("Unsupported extraction recipe version")
    if (len(elf) != recipe["elf_size"] or
            hashlib.sha256(elf).hexdigest() != recipe["elf_sha256"]):
        raise ValueError("Unsupported game executable; US retail SCUS-97197 is required")
    images = {}
    for item in recipe["images"]:
        key, unit, size = item["id"], item["unit"], item["size"]
        if not re.fullmatch(r"[0-9a-f]{16}", key) or unit not in (0, 1):
            raise ValueError("Invalid image identifier")
        if size != (4096 if unit == 0 else 16384):
            raise ValueError("Invalid image size")
        data, written = bytearray(size), bytearray(size)
        for span in item["spans"]:
            dest, offset, count = (span[k] for k in ("destination", "offset", "length"))
            if (not all(type(n) is int for n in (dest, offset, count)) or
                    dest < 0 or offset < 0 or count <= 0 or count % 8 or dest % 8 or
                    dest + count > size or offset + count > len(elf)):
                raise ValueError("Extraction span lies outside its input or output")
            if any(written[dest:dest+count]):
                raise ValueError("Overlapping extraction spans")
            data[dest:dest+count] = elf[offset:offset+count]
            written[dest:dest+count] = b"\1" * count
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Recovered microcode failed its integrity check")
        name = f"vu{unit}_image_{key}.bin"
        if name in images:
            raise ValueError("Duplicate image identifier")
        images[name] = bytes(data)
    if not images:
        raise ValueError("Empty extraction recipe")
    return images


def write_images(images: dict[str, bytes], destination: Path) -> None:
    # Validate the whole operation before touching any existing install files.
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in images.items():
        target = destination / name
        if target.is_symlink() or (target.exists() and target.read_bytes() != data):
            raise ValueError(f"Existing local output differs: {name}")
    for name, data in images.items():
        target = destination / name
        if not target.exists():
            with target.open("xb") as stream:
                stream.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("elf", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    recipe = json.loads(Path(__file__).with_name("vu-extraction.json").read_text(encoding="utf-8"))
    images = extract_images(args.elf.read_bytes(), recipe)
    write_images(images, args.destination)
    print(f"Recovered and verified {len(images)} local microcode images.")


if __name__ == "__main__":
    main()
