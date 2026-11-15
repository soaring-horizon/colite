#!/usr/bin/env python3
"""Quantize Colite float SIFT descriptors to COLMAP uint8/512 semantics."""

import argparse
from pathlib import Path

import numpy as np
from colite_workspace_features import read_feature, write_feature, colmap_keypoints, colmap_descriptors


def quantize(path: Path) -> tuple[int, int]:
    feature = read_feature(path)
    if feature["extractor"] != 1:
        raise ValueError(f"Unsupported Colite feature file: {path}")
    quantized, _ = colmap_descriptors(feature)
    descriptors = quantized.astype("<f4") / np.float32(512)
    original = colmap_keypoints(feature) if feature["source_columns"] else None
    write_feature(path, feature["keypoints"], descriptors, 0,
                  "L1_ROOT" if feature["normalization"] == 2 else "L2", original)
    return descriptors.shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    files = sorted((args.workspace / "features").glob("*.bin"))
    # A manual descriptor rewrite must not leave dependent stage caches valid,
    # including if a later file fails partway through the conversion.
    if files:
        for stage in ("features", "matches", "geometry", "sparse", "recon_sfm"):
            (args.workspace / "cache" / (stage + ".txt")).unlink(missing_ok=True)
    total = 0
    for path in files:
        rows, _ = quantize(path)
        total += rows
    print(f"files={len(files)} descriptors={total}")


if __name__ == "__main__":
    main()
