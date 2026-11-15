#!/usr/bin/env python3
"""Check an installed Colite prefix using a small synthetic scene (no dataset).

Uses only NumPy and modules shipped in the installation. Exercises extraction,
matching, geometric verification, mapping, sparse conversion, cache reuse and
repair. Native COLMAP interchange and full MVS are separate release checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_png(path, pixels):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data +
                struct.pack(">I", zlib.crc32(kind + data)))
    height, width = pixels.shape
    rows = b"".join(b"\0" + row.tobytes() for row in pixels)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" +
                     chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)) +
                     chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def check_geometry(model):
    lines = [line for line in (model / "images.txt").read_text().splitlines()
             if not line.startswith("#")]
    images = {}
    for index in range(0, len(lines), 2):
        row = lines[index].split()
        if not row:
            continue
        q = np.asarray(row[1:5], float)
        w, x, y, z = q / np.linalg.norm(q)
        rotation = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        translation = np.asarray(row[5:8], float)
        observations = lines[index+1].split()
        xy = np.array([observations[j:j+2]
                       for j in range(0, len(observations), 3)], float)
        images[int(row[0])] = rotation, translation, xy
    if len(images) != 3:
        raise RuntimeError(f"Expected three registered images, got {len(images)}")
    for first, second in ((1, 2), (1, 3), (2, 3)):
        ra, ta, _ = images[first]
        rb, tb, _ = images[second]
        relative = rb @ ra.T
        angle = np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1)))
        direction = tb - relative @ ta
        if angle > 0.1 or -direction[0] / np.linalg.norm(direction) < 0.999:
            raise RuntimeError("Recovered relative pose disagrees with synthetic geometry")
    errors = []
    points = 0
    for line in (model / "points3D.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        row = line.split()
        xyz = np.asarray(row[1:4], float)
        points += 1
        for j in range(8, len(row), 2):
            rotation, translation, xy = images[int(row[j])]
            local = rotation @ xyz + translation
            if local[2] <= 0:
                raise RuntimeError("Reconstruction contains a point behind a camera")
            projected = 420 * local[:2] / local[2] + [320, 240]
            errors.append(float(np.linalg.norm(projected - xy[int(row[j+1])])))
    if points < 100 or not errors or max(errors) > 0.01:
        raise RuntimeError("Synthetic reconstruction lost points or has excessive reprojection error")
    return dict(images=len(images), points=points, observations=len(errors),
                max_reprojection_error=max(errors))


def smoke(prefix, root):
    binary = prefix / ("bin/colite.exe" if os.name == "nt" else "bin/colite")
    modules = prefix / "share/colite/scripts"
    if not binary.is_file() or not (modules / "colite_workspace_features.py").is_file():
        raise RuntimeError("Prefix must contain the installed executable and offline modules")
    sys.path.insert(0, str(modules))
    from colite_workspace_features import write_feature, read_feature

    report = dict(complete=False, prefix=str(prefix), binary_sha256=sha(binary),
                  scope="synthetic installed SfM; native interchange/full MVS validated separately",
                  commands={})
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    cwd = root / "isolated"
    cwd.mkdir()

    def save():
        (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def run(tag, args, success=True):
        argv = [str(binary), *map(str, args)]
        result = subprocess.run(argv, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (root / (tag + ".log")).write_text(result.stdout)
        report["commands"][tag] = dict(argv=argv, returncode=result.returncode)
        save()
        if (result.returncode == 0) != success:
            raise RuntimeError(f"Unexpected exit status: {tag}; inspect {root / (tag + '.log')}")
        return result.stdout

    images, workspace = root / "images", root / "workspace"
    images.mkdir()
    (workspace / "features").mkdir(parents=True)
    (workspace / "cameras.txt").write_text(
        "1 PINHOLE 640 480 420 420 320 240 PRIOR_FOCAL=1\n")
    rng = np.random.default_rng(37)
    xyz = rng.uniform([-0.8, -0.8, 4], [0.8, 0.8, 8], (120, 3))
    descriptors = rng.uniform(0, 1, (120, 128)).astype("<f4")
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    pixels = rng.integers(0, 256, (480, 640), dtype=np.uint8)
    rows = []
    for image_id in (1, 2, 3):
        local = xyz - [(image_id-1)*0.6, 0, 0]
        xy = (420 * local[:, :2] / local[:, 2:] + [320, 240]).astype("<f4")
        write_feature(workspace / "features" / f"{image_id}.bin",
                      np.column_stack([xy, np.ones(120), np.zeros((120, 2))]),
                      descriptors, 0, source_keypoints=xy)
        image = images / f"{image_id}.png"
        write_png(image, np.roll(pixels, image_id*3, axis=1))
        rows.append(f"{image_id}\t1\t{image_id}.png\t{image}\n")
    (workspace / "images.tsv").write_text("".join(rows))

    extracted = root / "extracted"
    run("extract", ["feature_extractor", "--image_path", images,
                    "--Colite.workspace_path", extracted,
                    "--FeatureExtraction.use_gpu", 0,
                    "--FeatureExtraction.num_threads", 2,
                    "--FeatureExtraction.max_image_size", 320,
                    "--SiftExtraction.max_num_features", 256,
                    "--ImageReader.single_camera", 1])
    if sum(len(read_feature(p)["keypoints"]) for p in (extracted / "features").glob("*.bin")) == 0:
        raise RuntimeError("Installed feature extractor produced no features")

    run("match", ["exhaustive_matcher", "--Colite.workspace_path", workspace,
                  "--FeatureMatching.use_gpu", 0, "--FeatureMatching.num_threads", 2,
                  "--TwoViewGeometry.random_seed", 0])
    output = root / "sparse"
    args = ["mapper", "--Colite.workspace_path", workspace, "--image_path", images,
            "--output_path", output, "--Mapper.num_threads", 2, "--Mapper.random_seed", 0,
            "--Mapper.init_min_num_inliers", 30, "--Mapper.init_min_tri_angle", 1,
            "--Mapper.ba_refine_focal_length", 0, "--Mapper.ba_refine_principal_point", 0,
            "--Mapper.ba_refine_extra_params", 0,
            "--Mapper.structure_less_registration_fallback", 0]
    if "Sparse cache hit: 0" not in run("mapper_cold", args):
        raise RuntimeError("First mapper run unexpectedly reused a cache")
    model = output / "0"
    report["geometry"] = check_geometry(model)
    names = ("cameras.bin", "images.bin", "points3D.bin")
    hashes = {name: sha(model / name) for name in names}
    if "Sparse cache hit: 1" not in run("mapper_warm", args):
        raise RuntimeError("Second mapper run failed to reuse a valid cache")
    if hashes != {name: sha(model / name) for name in names}:
        raise RuntimeError("Cache reuse changed the saved model")
    (model / "points3D.bin").write_bytes(b"truncated")
    run("repair_export", args)
    if hashes != {name: sha(model / name) for name in names}:
        raise RuntimeError("Mapper did not repair the truncated exported model")

    text, binary_model = root / "text", root / "binary"
    for tag, source, target, kind in (("to_text", model, text, "TXT"),
                                      ("to_binary", text, binary_model, "BIN")):
        run(tag, ["model_converter", "--input_path", source,
                  "--output_path", target, "--output_type", kind])
    if hashes != {name: sha(binary_model / name) for name in names}:
        raise RuntimeError("BIN/TXT round trip changed sparse model fields")
    plan = root / "invalid_plan"
    run("invalid_dry_run", ["recon_auto", "--image_path", images,
                           "--output_path", plan, "--PatchMatchStereo.num_iterations", 0,
                           "--dry_run"], success=False)
    if plan.exists():
        raise RuntimeError("Invalid dry-run published an output directory")
    report.update(complete=True, sparse_sha256=hashes)
    save()
    print(json.dumps(report["geometry"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, help="New output directory; omitted uses temporary files")
    args = parser.parse_args()
    if args.output_path:
        root = args.output_path.resolve()
        root.mkdir(parents=True, exist_ok=False)
        smoke(args.prefix.resolve(), root)
    else:
        with tempfile.TemporaryDirectory(prefix="colite-installed-") as directory:
            smoke(args.prefix.resolve(), Path(directory))


if __name__ == "__main__":
    main()
