#!/usr/bin/env python3
"""Fetch the exact desktop SDK used by the workbench from Qt's repository."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET


def fetch(url, destination):
    temporary = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(destination)
            return
        except (OSError, TimeoutError):
            if attempt == 2:
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=("linux", "windows"), required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--sevenzip", default="7z")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)
    host, architecture = ("linux_x64", "gcc_64") if args.platform == "linux" else ("windows_x86", "win64_msvc2019_64")
    repository = f"https://download.qt.io/online/qtsdkrepository/{host}/desktop/qt5_5152/"
    metadata = args.cache / f"qt-5.15.2-{args.platform}-updates.xml"
    if not metadata.exists():
        fetch(repository + "Updates.xml", metadata)
    package_name = "qt.qt5.5152." + architecture
    package = next(p for p in ET.parse(metadata).getroot().findall("PackageUpdate") if p.findtext("Name") == package_name)
    version = package.findtext("Version")
    wanted = [a.strip() for a in package.findtext("DownloadableArchives").split(",")
              if a.strip().startswith(("qtbase-", "qttools-", "qttranslations-", "qtwayland-", "icu-", "d3dcompiler_47-"))]
    if not any(a.startswith("qtbase-") for a in wanted):
        raise RuntimeError("Qt repository did not contain qtbase")
    receipt = {"version": "5.15.2", "package": package_name, "repository_version": version, "archives": []}
    for archive in wanted:
        url = repository + package_name + "/" + version + archive
        checksum = args.cache / (archive + ".sha1")
        if not checksum.exists():
            fetch(url + ".sha1", checksum)
        expected = checksum.read_text(encoding="ascii").strip().split()[0].lower()
        if len(expected) != 40 or any(c not in "0123456789abcdef" for c in expected):
            raise RuntimeError("Invalid official checksum")
        cached = args.cache / archive
        if not cached.exists() or hashlib.sha1(cached.read_bytes()).hexdigest() != expected:
            fetch(url, cached)
        content = cached.read_bytes()
        if hashlib.sha1(content).hexdigest() != expected:
            raise RuntimeError("Qt archive checksum mismatch: " + archive)
        subprocess.run([args.sevenzip, "x", str(cached.resolve()), "-o" + str(args.destination.resolve()), "-y"], check=True, stdout=subprocess.DEVNULL)
        receipt["archives"].append({"url": url, "sha1": expected, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)})
    (args.destination / "qt-sdk-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    prefix = args.destination / "5.15.2" / ("gcc_64" if args.platform == "linux" else "msvc2019_64")
    if not (prefix / "lib/cmake/Qt5/Qt5Config.cmake").is_file():
        raise RuntimeError("Qt SDK extraction is incomplete")
    print(prefix.resolve())


if __name__ == "__main__":
    main()
