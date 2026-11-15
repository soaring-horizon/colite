#!/usr/bin/env python3
"""Wrap a validated desktop prefix in an Ubuntu .deb without developer tools."""
import argparse
import json
import re
from pathlib import Path
import shutil
import subprocess
from package_linux import owning_package
from release_version import release_version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--version", default=release_version(debian=True))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9][A-Za-z0-9.+:~\-]*", args.version):
        raise ValueError("Invalid Debian package version")
    if args.work.exists():
        raise RuntimeError("Use an empty packaging work directory")
    root = args.work.resolve()
    shutil.copytree(args.prefix, root / "opt/colite", symlinks=True)
    record = json.loads((args.prefix / "package-identity.json").read_text())
    dependencies = {"libc6 (>= 2.35)", "fonts-noto-cjk"}
    for name, source in record["host_libraries"].items():
        if name in ("libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2", "librt.so.1", "libresolv.so.2"):
            continue
        package = owning_package(source).split(':')[0]
        if package == "libglib2.0-0":
            package = "libglib2.0-0 | libglib2.0-0t64"
        dependencies.add(package)
    (root / "DEBIAN").mkdir()
    installed_size = sum(p.stat().st_size for p in (root / "opt").rglob("*") if p.is_file()) // 1024
    control = (f"Package: colite\nVersion: {args.version}\nArchitecture: amd64\n"
               "Maintainer: Colite contributors\nSection: graphics\nPriority: optional\n"
               f"Installed-Size: {installed_size}\nDepends: {', '.join(sorted(dependencies))}\n"
               "Description: Colite photogrammetry workbench\n"
               " Native desktop GUI and CPU SfM/MVS reconstruction worker.\n")
    (root / "DEBIAN/control").write_text(control)
    applications = root / "usr/share/applications"
    applications.mkdir(parents=True)
    (applications / "colite.desktop").write_text(
        "[Desktop Entry]\nType=Application\nName=Colite\nComment=Photogrammetry workbench\n"
        "Exec=/opt/colite/bin/colite_gui %f\nIcon=colite\nTerminal=false\n"
        "Categories=Graphics;3DGraphics;Science;\nMimeType=application/x-colite-project;\n")
    icons = root / "usr/share/icons/hicolor/256x256/apps"
    icons.mkdir(parents=True)
    shutil.copy2(Path(__file__).resolve().parent.parent / "gui/assets/colite.png", icons / "colite.png")
    mime = root / "usr/share/mime/packages"
    mime.mkdir(parents=True)
    (mime / "colite.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">'
        '<mime-type type="application/x-colite-project"><comment>Colite project</comment>'
        '<glob pattern="*.colite.json"/></mime-type></mime-info>\n')
    (root / "usr/bin").mkdir()
    for name in ("colite", "colite_gui"):
        (root / "usr/bin" / name).symlink_to("/opt/colite/bin/" + name)
    # DrvFS inputs may report 0777 for every file. Normalize the archive modes.
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        executable = relative.startswith("opt/colite/bin/") or relative.startswith("opt/colite/libexec/") and path.suffix != '.py' and path.name != 'qt.conf'
        path.chmod(0o755 if path.is_dir() or executable else 0o644)
    if (root / 'DEBIAN').stat().st_mode & 0o777 != 0o755:
        raise RuntimeError('Packaging work directory must support Unix permissions; use a Linux filesystem rather than DrvFS')
    subprocess.run(["dpkg-deb", "--root-owner-group", "--build", str(root), str(args.output.resolve())], check=True)


if __name__ == "__main__":
    main()
