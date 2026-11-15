#!/usr/bin/env python3
"""Copy desktop runtime source and notice material into an installed prefix."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", required=True, type=Path)
    parser.add_argument("--qt-source", required=True, type=Path)
    parser.add_argument("--extra-source", action="append", default=[], type=Path)
    args = parser.parse_args()
    sources = args.install / "share/colite/sources"
    notices = args.install / "share/colite/licenses/runtime/Qt-5.15.2"
    sources.mkdir(parents=True, exist_ok=True)
    notices.mkdir(parents=True, exist_ok=True)
    inventory = []
    for archive in [args.qt_source, *args.extra_source]:
        shutil.copy2(archive, sources / archive.name)
        inventory.append({"file": archive.name, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()})
    notice_inventory = []
    with tarfile.open(args.qt_source) as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if not member.isfile() or member.size > 4 * 1024 * 1024 or ".." in name.parts:
                continue
            if not (name.name.upper().startswith(("LICENSE", "COPYING", "NOTICE")) or name.name == "qt_attribution.json"):
                continue
            target = notices / (f"{len(notice_inventory):04d}-" + name.name)
            with archive.extractfile(member) as input_file, target.open("wb") as output_file:
                shutil.copyfileobj(input_file, output_file)
            notice_inventory.append({"file": target.name, "source_path": str(name)})
    (notices / "notice-inventory.json").write_text(json.dumps(notice_inventory, indent=2) + "\n")
    (sources / "source-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(sources.resolve())


if __name__ == "__main__":
    main()
