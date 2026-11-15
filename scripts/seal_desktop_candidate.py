#!/usr/bin/env python3
"""Probe, seal and verify local desktop candidates; never publishes remotely."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

from release_version import release_version


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pending.replace(path)


def package_file(package, relative):
    path = package / relative
    if not path.resolve().is_relative_to(package.resolve()):
        raise ValueError("Payload leaves package")
    return path


def probe(package):
    """Run on the target platform. Bind embedded build identity to binary hashes."""
    windows = (package / "bin/colite.exe").is_file()
    result = {}
    for name in ("colite", "colite_gui"):
        launcher = package / "bin" / (name + (".exe" if windows else ""))
        binary = launcher if windows else package / "libexec" / name
        identity = json.loads(subprocess.check_output([str(launcher), "--build-info"], timeout=30, text=True))
        result[name] = {**identity, "binary": binary.relative_to(package).as_posix(), "sha256": sha(binary)}
    write(package / "build-probe.json", result)


def verify(package):
    identity = read(package / "package-identity.json")
    if not identity.get("source_frozen"):
        raise ValueError(f"Not sealed: {package}")
    expected = set()
    for item in identity["payload"]:
        path = package_file(package, item["file"])
        if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
            raise ValueError(f"Payload mismatch: {path}")
        expected.add(item["file"])
    actual = {p.relative_to(package).as_posix() for p in package.rglob("*")
              if p.is_file() and p != package / "package-identity.json"}
    if expected != actual:
        raise ValueError("Unexpected or missing payload files")
    return identity


def seal(args):
    repo = Path(__file__).resolve().parents[1]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], cwd=repo, check=True)
    unknown = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, text=True).splitlines()
    if any(not any(name == root or name.startswith(root.rstrip("/") + "/") for root in args.allow_untracked) for name in unknown):
        raise ValueError("Commit source files first; explicitly allow unrelated untracked directories")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    version = release_version()
    # Verify binaries before creating or modifying any sealed package.
    for package in args.package:
        identity = read(package / "package-identity.json")
        if identity["version"] != version:
            raise ValueError(f"Package version mismatch: {package}")
        builds = read(package / "build-probe.json")
        if set(builds) != {"colite", "colite_gui"}:
            raise ValueError("Missing GUI/engine build probe")
        for item in builds.values():
            revision = item["source_revision"]
            if item["version"] != version or len(revision) < 12 or not commit.startswith(revision):
                raise ValueError(f"Rebuild binaries from frozen commit {commit}: {package}")
            if sha(package_file(package, item["binary"])) != item["sha256"]:
                raise ValueError(f"Binary changed after native probe: {package}")
        sources = package / "share/colite/sources"
        inventory = read(sources / "source-inventory.json")
        if not inventory:
            raise ValueError("Missing third-party sources")
        for item in inventory:
            if sha(package_file(sources, item["file"])) != item["sha256"]:
                raise ValueError("Third-party source checksum mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / f"colite-{version}-source.tar"
    if archive.exists():
        raise FileExistsError(archive)
    subprocess.run(["git", "archive", "--format=tar", f"--prefix=colite-{version}/", "-o", str(archive), commit], cwd=repo, check=True)
    source = {"commit": commit, "archive": archive.name, "sha256": sha(archive)}
    for package in args.package:
        identity = read(package / "package-identity.json")
        identity.update(candidate=version, source_frozen=True, source=source)
        identity["builds"] = read(package / "build-probe.json")
        identity["payload"] = [{"file": p.relative_to(package).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)}
                               for p in sorted(package.rglob("*")) if p.is_file() and p != package / "package-identity.json"]
        write(package / "package-identity.json", identity)
        verify(package)
    write(args.output / "source-identity.json", source)


def archive_package(package, output):
    verify(package)
    output.mkdir(parents=True, exist_ok=True)
    windows = (package / "bin/colite.exe").is_file()
    destination = output / (package.name + (".zip" if windows else ".tar.gz"))
    if destination.exists():
        raise FileExistsError(destination)
    if windows:
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file():
                    archive.write(path, Path(package.name) / path.relative_to(package))
    else:
        def permissions(member):
            relative = member.name.removeprefix(package.name + "/")
            executable = relative.startswith("bin/") or (relative.startswith("libexec/") and not relative.endswith((".py", "qt.conf")))
            member.mode = 0o755 if member.isdir() or executable else 0o644
            member.uid = member.gid = 0
            member.uname = member.gname = "root"
            return member
        with tarfile.open(destination, "w:gz", compresslevel=5, dereference=True) as archive:
            archive.add(package, arcname=package.name, filter=permissions)
    print(f"{sha(destination)}  {destination.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "seal", "verify", "archive"))
    parser.add_argument("--package", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-untracked", action="append", default=[])
    args = parser.parse_args()
    args.package = [p.resolve() for p in args.package]
    if args.action in ("seal", "archive") and args.output is None:
        parser.error("--output is required")
    if args.output:
        args.output = args.output.resolve()
    if args.action == "seal":
        seal(args)
    else:
        for package in args.package:
            if args.action == "probe": probe(package)
            elif args.action == "verify": verify(package)
            else: archive_package(package, args.output)


if __name__ == "__main__":
    main()
