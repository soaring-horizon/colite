#!/usr/bin/env python3
"""Assemble a relocatable Linux desktop prefix with a small runtime closure."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from release_version import release_version

BUNDLED = ("libQt5", "libicu", "libceres", "libopencv", "libglog", "libgflags", "liblz4", "libunwind", "liblzma", "libonnxruntime")
CUDA_RUNTIME = ("libcudart", "libcublas", "libnvJitLink", "libnvrtc")


def is_elf(path):
    if not path.is_file():
        return False
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def command(*args, env=None):
    return subprocess.check_output(args, env=env, text=True, stderr=subprocess.STDOUT)


def owning_package(source):
    source = Path(source)
    candidates = [source, source.resolve()]
    for candidate in list(candidates):
        if str(candidate).startswith('/usr/lib/'):
            candidates.append(Path(str(candidate)[4:]))
        elif str(candidate).startswith('/lib/'):
            candidates.append(Path('/usr' + str(candidate)))
    for candidate in candidates:
        try:
            return command('dpkg-query', '-S', str(candidate)).splitlines()[0].split(': ', 1)[0]
        except subprocess.CalledProcessError:
            pass
    raise RuntimeError('Cannot identify the package owning ' + str(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("install", "qt", "dependencies", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--cuda-architectures", default="")
    parser.add_argument("--onnx-root", type=Path, help="CPU ONNX Runtime SDK root")
    args = parser.parse_args()
    if not all(is_elf(args.install / 'bin' / name) for name in ('colite', 'colite_gui')):
        raise RuntimeError('Install both the native GUI and worker before packaging')
    if args.output.exists():
        raise RuntimeError("Use a new output directory to prevent stale runtime files")
    shutil.copytree(args.install, args.output, symlinks=True)
    root = args.output.resolve()
    (root / "bin").rename(root / "libexec")
    (root / "bin").mkdir()
    (root / "lib").mkdir(exist_ok=True)
    plugin_names = ("platforms/libqxcb.so", "platforms/libqoffscreen.so", "imageformats/libqjpeg.so",
                    "xcbglintegrations/libqxcb-glx-integration.so", "xcbglintegrations/libqxcb-egl-integration.so")
    pending = [p for p in (root / "libexec").iterdir() if is_elf(p)]
    for plugin in plugin_names:
        target = root / "plugins" / plugin
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.qt / "plugins" / plugin, target)
        pending.append(target)
    search = [args.dependencies / "lib", args.qt / "lib"]
    if args.onnx_root:
        if (args.onnx_root / "lib/libonnxruntime_providers_cuda.so").exists():
            raise RuntimeError('Use the CPU ONNX SDK; CUDA providers require a separate cuDNN/runtime closure')
        search.append(args.onnx_root / "lib")
    if args.cuda_root:
        search.extend([args.cuda_root / "lib64", args.cuda_root / "targets/x86_64-linux/lib"])
    env = dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, search)))
    records, host, processed = {}, {}, set()
    notices = root / "share/colite/licenses/runtime"
    notices.mkdir(parents=True, exist_ok=True)
    if args.onnx_root:
        for name in ('LICENSE', 'ThirdPartyNotices.txt'):
            shutil.copy2(args.onnx_root / name, notices / ('ONNX-Runtime-' + name))
    while pending:
        binary = pending.pop()
        if str(binary) in processed:
            continue
        processed.add(str(binary))
        needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", command("readelf", "-d", str(binary)))
        linkage = command("ldd", str(binary), env=env)
        resolved = dict(re.findall(r"^\s*(\S+) => (\S+) \(", linkage, re.M))
        for interpreter in re.findall(r"^\s*(/\S+) \(", linkage, re.M):
            resolved[Path(interpreter).name] = interpreter
        for name in needed:
            source = Path(resolved.get(name, ""))
            if not source.is_file():
                raise RuntimeError(f"Cannot resolve {name} needed by {binary}")
            if name.startswith(CUDA_RUNTIME) and not args.cuda_root:
                raise RuntimeError('CUDA packages require an explicit --cuda-root')
            if name.startswith('libonnxruntime') and not args.onnx_root:
                raise RuntimeError('ONNX packages require --onnx-root for runtime and license deployment')
            if not name.startswith(BUNDLED + CUDA_RUNTIME):
                host[name] = str(source)
                continue
            if name in records:
                continue
            # Give each SONAME its own file: runtime linking needs no development symlinks.
            target = root / "lib" / name
            shutil.copy2(source.resolve(), target)
            records[name] = {"source": str(source.resolve()), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            pending.append(target)
            vendor_roots = [str(args.qt.resolve()), str(args.dependencies.resolve())]
            if args.cuda_root:
                vendor_roots.append(str(args.cuda_root.resolve()))
            if args.onnx_root:
                vendor_roots.append(str(args.onnx_root.resolve()))
            if not str(source.resolve()).startswith(tuple(vendor_roots)):
                package = owning_package(source)
                copyright_file = Path("/usr/share/doc") / package.split(":")[0] / "copyright"
                if not copyright_file.is_file():
                    raise RuntimeError("Missing runtime notice for " + package)
                shutil.copy2(copyright_file, notices / (package.replace(":", "-") + "-copyright.txt"))
                records[name]["package"] = package
    if args.cuda_root:
        copied_license = False
        for name in ('EULA.txt', 'LICENSE', 'version.json'):
            source = args.cuda_root / name
            if source.is_file():
                shutil.copy2(source, notices / ('NVIDIA-CUDA-' + name))
                copied_license |= name in ('EULA.txt', 'LICENSE')
        if not copied_license:
            raise RuntimeError('CUDA runtime license is missing from the supplied toolkit')
    for binary in (root / "libexec").iterdir():
        if not is_elf(binary):
            continue
        script = '#!/bin/sh\nset -eu\nAPPDIR=$(CDPATH= cd -- "$(dirname -- "$(readlink -f -- "$0")")/.." && pwd)\n'
        script += 'export LD_LIBRARY_PATH="$APPDIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
        if binary.name == "colite_gui":
            script += 'export QT_PLUGIN_PATH="$APPDIR/plugins"\nexport QT_QPA_PLATFORM_PLUGIN_PATH="$APPDIR/plugins/platforms"\nexport QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"\n'
        script += 'exec "$APPDIR/libexec/' + binary.name + '" "$@"\n'
        launcher = root / "bin" / binary.name
        launcher.write_text(script)
        launcher.chmod(0o755)
    (root / "libexec/qt.conf").write_text("[Paths]\nPrefix=..\nLibraries=lib\nPlugins=plugins\n")
    identity = {"schema": 1, "version": release_version(), "platform": "linux-x64", "qt": "5.15.2",
                "cuda": bool(args.cuda_root), "cuda_architectures": args.cuda_architectures,
                "source_frozen": False, "runtime": records, "host_libraries": host}
    (root / "package-identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(root)


if __name__ == "__main__":
    main()
