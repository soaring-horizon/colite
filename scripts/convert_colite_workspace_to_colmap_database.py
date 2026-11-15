#!/usr/bin/env python3
"""Export features, raw matches and optional verified geometries to COLMAP 4, offline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile

import numpy as np
from colite_workspace_features import read_feature, colmap_keypoints, colmap_descriptors
from colite_workspace_geometry import read_geometries
from convert_colmap_database_to_colite_workspace import CAMERA_MODELS, MAX_IMAGE_ID, POSE_PRIOR_COLUMNS, read_pose_priors

PARAM_COUNTS = (3, 4, 4, 5, 8, 8, 12, 5, 4, 5, 12, 16)
MODEL_IDS = {name: value for value, name in CAMERA_MODELS.items()}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def lines(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip() and not line.lstrip().startswith("#"):
                yield line.rstrip("\r\n")


def cameras(path):
    result = {}
    for line in lines(path):
        row = line.split()
        camera_id, name, width, height = int(row[0]), row[1], int(row[2]), int(row[3])
        if name not in MODEL_IDS:
            raise ValueError("Unsupported COLMAP camera model: " + name)
        model = MODEL_IDS[name]
        count = PARAM_COUNTS[model]
        if len(row) not in (4 + count, 5 + count):
            raise ValueError("Invalid workspace camera parameter count")
        params = np.array(row[4:4 + count], dtype="<f8")
        prior = row[4 + count:] or ["PRIOR_FOCAL=0"]
        if (camera_id in result or not 0 <= camera_id < 2**32 - 1 or width <= 0 or height <= 0 or
                not np.isfinite(params).all() or prior[0] not in ("PRIOR_FOCAL=0", "PRIOR_FOCAL=1")):
            raise ValueError("Invalid workspace camera metadata")
        result[camera_id] = (model, width, height, params.tobytes(), int(prior[0][-1]))
    return result


def images(path, camera_data):
    result, names = {}, set()
    for line in lines(path):
        row = line.split("\t")
        if len(row) != 4:
            raise ValueError("Expected images.tsv without pose priors; prior covariance/coordinate exchange is not implemented")
        image_id, camera_id, name = int(row[0]), int(row[1]), row[2]
        if (not 0 <= image_id < MAX_IMAGE_ID or image_id in result or camera_id not in camera_data or
                not name or name in names):
            raise ValueError("Invalid image ID, name or camera reference")
        result[image_id] = (name, camera_id)
        names.add(name)
    return result


def match_pairs(path, counts):
    if not path.exists(): return
    current, values, empty = None, [], False
    for line in lines(path):
        row = line.split()
        if len(row) != 4: raise ValueError("Invalid raw match record")
        first, second, a, b = map(int, row)
        if first > second: first, second, a, b = second, first, b, a
        pair = (first, second)
        if first == second or first not in counts or second not in counts or (current and pair < current):
            raise ValueError("Raw pairs must reference distinct images and be ordered by canonical pair")
        if pair != current:
            if current is not None: yield current, np.asarray(values, dtype="<u4").reshape(-1, 2)
            current, values, empty = pair, [], False
        if a == b == -1:
            if values or empty: raise ValueError("Invalid empty raw pair sentinel")
            empty = True
        else:
            if empty or not (0 <= a < counts[first] and 0 <= b < counts[second]):
                raise ValueError("Raw match index out of range")
            values.append((a, b))
    if current is not None: yield current, np.asarray(values, dtype="<u4").reshape(-1, 2)


def create_database(path, colmap):
    result = subprocess.run([str(colmap), "database_creator", "--database_path", str(path)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError("COLMAP database_creator failed: " + result.stdout[-2000:])


def export_workspace(workspace, database, colmap="colmap", export_verified=True):
    workspace, database = Path(workspace).resolve(), Path(database).resolve()
    if database.exists(): raise ValueError("Output database must not exist")
    camera_path, image_path = workspace / "cameras.txt", workspace / "images.tsv"
    camera_data = cameras(camera_path)
    image_data = images(image_path, camera_data)
    counts, identities, report = {}, {camera_path: sha(camera_path), image_path: sha(image_path)}, {}
    prior_path = workspace / "colmap_pose_priors.tsv"
    if prior_path.exists(): identities[prior_path] = sha(prior_path)
    pose_priors = read_pose_priors(prior_path, {key: value[1] for key, value in image_data.items()})
    for image_id in image_data:
        path = workspace / "features" / f"{image_id}.bin"
        feature = read_feature(path)
        colmap_keypoints(feature)
        _, quantization = colmap_descriptors(feature)
        identities[path] = feature["sha256"]
        counts[image_id] = len(feature["keypoints"])
        report[str(image_id)] = dict(extractor=feature["extractor"], normalization=feature["normalization"],
                                     keypoints=counts[image_id], **quantization)
    raw_path = workspace / "matches/raw_matches.tsv"
    if raw_path.exists(): identities[raw_path] = sha(raw_path)
    raw_pairs = raw_rows = 0
    for _, matches in match_pairs(raw_path, counts):
        raw_pairs += 1
        raw_rows += len(matches)
    verified_path = workspace / "matches/verified_matches.tsv"
    verified_pairs = verified_rows = 0
    if export_verified and verified_path.exists():
        identities[verified_path] = sha(verified_path)
        for geometry in read_geometries(verified_path, counts):
            verified_pairs += 1
            verified_rows += len(geometry["matches"])
    database.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".colite-export-", dir=database.parent) as scratch:
        temporary = Path(scratch) / "database.db"
        create_database(temporary, colmap)
        connection = sqlite3.connect(temporary)
        try:
            required = {"cameras", "images", "keypoints", "descriptors", "matches", "two_view_geometries",
                        "rigs", "rig_sensors", "frames", "frame_data"}
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            columns = {row[1] for row in connection.execute("PRAGMA table_info(descriptors)")}
            if not required <= tables or "type" not in columns:
                raise ValueError("COLMAP 4 typed database schema is required")
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                for camera_id, values in camera_data.items():
                    connection.execute("INSERT INTO cameras(camera_id,model,width,height,params,prior_focal_length) VALUES(?,?,?,?,?,?)",
                                       (camera_id, *values))
                    connection.execute("INSERT INTO rigs(rig_id,ref_sensor_id,ref_sensor_type) VALUES(?,?,0)", (camera_id, camera_id))
                for image_id, (name, camera_id) in image_data.items():
                    connection.execute("INSERT INTO frames(frame_id,rig_id) VALUES(?,?)", (image_id, camera_id))
                    connection.execute("INSERT INTO frame_data(frame_id,data_id,sensor_id,sensor_type) VALUES(?,?,?,0)", (image_id, image_id, camera_id))
                    connection.execute("INSERT INTO images(image_id,name,camera_id) VALUES(?,?,?)", (image_id, name, camera_id))
                    feature = read_feature(workspace / "features" / f"{image_id}.bin")
                    if feature["sha256"] != identities[workspace / "features" / f"{image_id}.bin"]:
                        raise ValueError("Source feature changed during export")
                    keypoints = colmap_keypoints(feature)
                    descriptors, _ = colmap_descriptors(feature)
                    connection.execute("INSERT INTO keypoints(image_id,rows,cols,data) VALUES(?,?,?,?)",
                                       (image_id, *keypoints.shape, keypoints.tobytes(order="C")))
                    connection.execute("INSERT INTO descriptors(image_id,type,rows,cols,data) VALUES(?,?,?,?,?)",
                                       (image_id, feature["extractor"] - 1, len(descriptors), descriptors.shape[1] * descriptors.itemsize,
                                        descriptors.tobytes(order="C")))
                for (first, second), matches in match_pairs(raw_path, counts):
                    connection.execute("INSERT INTO matches(pair_id,rows,cols,data) VALUES(?,?,2,?)",
                                       (first * MAX_IMAGE_ID + second, len(matches), matches.tobytes()))
                if export_verified:
                    for geometry in read_geometries(verified_path, counts):
                        def matrix_blob(name):
                            matrix = geometry[name]
                            return None if matrix is None else np.asarray(matrix, dtype="<f8").tobytes(order="C")
                        pose = geometry["pose"]
                        first, second = geometry["pair"]
                        connection.execute("INSERT INTO two_view_geometries(pair_id,rows,cols,data,config,F,E,H,qvec,tvec) VALUES(?,?,2,?,?,?,?,?,?,?)",
                            (first * MAX_IMAGE_ID + second, len(geometry["matches"]), geometry["matches"].tobytes(),
                             geometry["config"], matrix_blob("F"), matrix_blob("E"), matrix_blob("H"),
                             None if pose is None else np.asarray(pose[:4], dtype="<f8").tobytes(),
                             None if pose is None else np.asarray(pose[4:], dtype="<f8").tobytes()))
                if pose_priors:
                    columns = {row[1] for row in connection.execute("PRAGMA table_info(pose_priors)")}
                    if not set(POSE_PRIOR_COLUMNS) <= columns:
                        raise ValueError("COLMAP 4 pose prior schema is required")
                    connection.executemany("INSERT INTO pose_priors(" + ",".join(POSE_PRIOR_COLUMNS) +
                                           ") VALUES(?,?,?,?,?,?,?,?)", pose_priors)
            if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or list(connection.execute("PRAGMA foreign_key_check")):
                raise ValueError("Exported database failed integrity validation")
        finally:
            connection.close()
        if any(sha(path) != expected for path, expected in identities.items()):
            raise ValueError("Source workspace changed during export")
        # Publish without overwriting a concurrently created destination.
        os.link(temporary, database)
    return dict(cameras=len(camera_data), images=len(image_data), pose_prior_metadata=len(pose_priors),
                raw_pairs=raw_pairs, raw_matches=raw_rows,
                verified_geometries=verified_pairs, verified_matches=verified_rows, geometric_verification_required=not bool(verified_pairs), features=report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", type=Path, required=True, help="Colite workspace")
    parser.add_argument("--database_path", type=Path, required=True, help="New COLMAP 4 database")
    parser.add_argument("--colmap_path", default="colmap", help="COLMAP 4 executable used to create its schema")
    parser.add_argument("--Colite.export_verified", dest="export_verified", type=int, choices=(0, 1), default=1,
                        help="Export complete verified geometry (default 1); use 0 for legacy feature/raw-only exchange.")
    args = parser.parse_args()
    print(json.dumps(export_workspace(args.input_path, args.database_path, args.colmap_path, bool(args.export_verified)), indent=2))


if __name__ == "__main__":
    main()
