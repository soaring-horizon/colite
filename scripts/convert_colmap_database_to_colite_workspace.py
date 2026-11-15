#!/usr/bin/env python3
"""Import typed COLMAP features and raw matches into a Colite workspace.

Original keypoint shape is preserved in feature v3. This does not migrate
Colite's algorithm storage to SQLite or promise a lossless database snapshot.
Keypoints, typed descriptors, raw matches and verified geometry are imported
together; missing descriptors are rejected rather than silently discarded.
"""

import argparse
from contextlib import closing
import sqlite3
import tempfile
import time
from pathlib import Path

import numpy as np

from analyze_sfm_geometric_merges import quaternion_rotation, read_model
from colite_workspace_features import write_feature
from colite_workspace_geometry import write_geometries


MAX_IMAGE_ID = 2_147_483_647
CAMERA_MODELS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
    11: "RAD_TAN_THIN_PRISM_FISHEYE",
}

# Exchange metadata only: incremental mapping does not consume gravity priors.
# Keep original NaN payloads/NULLs rather than round-tripping through decimal text.
POSE_PRIOR_COLUMNS = ("pose_prior_id", "corr_data_id", "corr_sensor_id", "corr_sensor_type",
                      "position", "position_covariance", "gravity", "coordinate_system")
POSE_PRIOR_HEADER = "# colmap_pose_priors_v1\t" + "\t".join(POSE_PRIOR_COLUMNS)


def validate_pose_priors(rows, image_cameras):
    ids, assignments = set(), set()
    for row in rows:
        if len(row) != 8:
            raise ValueError("Invalid COLMAP pose prior record")
        prior_id, image_id, sensor_id, sensor_type, position, covariance, gravity, system = row
        if (any(type(v) is not int for v in (prior_id, image_id, sensor_id, sensor_type, system)) or
                not 0 <= prior_id < 2**32 - 1 or prior_id in ids or image_id in assignments or
                image_id not in image_cameras or sensor_type != 0 or sensor_id != image_cameras[image_id]):
            raise ValueError("Invalid single-camera pose prior association")
        ids.add(prior_id)
        assignments.add(image_id)
        for name, blob, count in (("position", position, 3), ("covariance", covariance, 9),
                                  ("gravity", gravity, 3)):
            if blob is None:
                continue
            if not isinstance(blob, bytes) or len(blob) != count * 8:
                raise ValueError("Invalid pose prior " + name + " blob")
            values = np.frombuffer(blob, dtype="<f8")
            if name != "gravity":
                if not np.isnan(values).all():
                    raise ValueError("Pose prior coordinate/covariance interchange is not implemented")
            elif not (np.isnan(values).all() or np.isfinite(values).all()):
                raise ValueError("Invalid pose prior gravity")
        if system != -1:
            raise ValueError("Pose prior coordinate/covariance interchange is not implemented")
    return rows


def database_pose_priors(connection, tables, image_cameras):
    if "pose_priors" not in tables:
        return []
    if not connection.execute("SELECT 1 FROM pose_priors LIMIT 1").fetchone():
        return []
    columns = {row[1] for row in connection.execute("PRAGMA table_info(pose_priors)")}
    if not set(POSE_PRIOR_COLUMNS) <= columns:
        raise ValueError("COLMAP 4 pose prior schema is required")
    rows = list(connection.execute("SELECT " + ",".join(POSE_PRIOR_COLUMNS) +
                                   " FROM pose_priors ORDER BY pose_prior_id"))
    return validate_pose_priors(rows, image_cameras)


def write_pose_priors(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8") as stream:
        stream.write(POSE_PRIOR_HEADER + "\n")
        for row in rows:
            stream.write("\t".join("NULL" if value is None else value.hex() if isinstance(value, bytes)
                                    else str(value) for value in row) + "\n")


def read_pose_priors(path, image_cameras):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        if stream.readline().rstrip("\r\n") != POSE_PRIOR_HEADER:
            raise ValueError("Invalid COLMAP pose prior metadata header")
        rows = []
        for line in stream:
            values = line.rstrip("\r\n").split("\t")
            if len(values) != 8:
                raise ValueError("Invalid COLMAP pose prior metadata row")
            rows.append(tuple(None if value == "NULL" else bytes.fromhex(value) if 4 <= i <= 6
                              else int(value) for i, value in enumerate(values)))
    return validate_pose_priors(rows, image_cameras)


def decode_pair_id(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def read_array(blob, dtype, rows, cols):
    if rows < 0 or cols < 0:
        raise ValueError("Negative database matrix dimensions")
    expected_bytes = rows * cols * np.dtype(dtype).itemsize
    if len(blob or b"") != expected_bytes:
        raise ValueError(f"Invalid database blob length: expected {expected_bytes}")
    if rows == 0:
        return np.empty((0, cols), dtype=dtype)
    return np.frombuffer(blob, dtype=dtype).reshape((rows, cols))


def convert_keypoints(source):
    """Keep COLMAP pixel coordinates and row order; convert radians to degrees."""
    rows, cols = source.shape
    if cols not in (2, 4, 6) or not np.isfinite(source).all():
        raise ValueError("Expected finite COLMAP keypoints with 2, 4 or 6 columns")
    keypoints = np.zeros((rows, 5), dtype="<f4")
    keypoints[:, :2] = source[:, :2]
    keypoints[:, 2] = 1.0
    if cols == 4:
        keypoints[:, 2] = source[:, 2]
        keypoints[:, 3] = np.degrees(source[:, 3]) % 360.0
    elif cols == 6:
        # Match COLMAP ComputeScale/ComputeOrientation. Full shear/anisotropy
        # is retained separately in feature v3; core similarity fields use
        # the mean scale/orientation for existing geometric algorithms.
        keypoints[:, 2] = (np.hypot(source[:, 2], source[:, 4]) +
                           np.hypot(source[:, 3], source[:, 5])) / 2.0
        keypoints[:, 3] = np.degrees(np.arctan2(source[:, 4], source[:, 2])) % 360.0
    return keypoints


def decode_descriptors(feature_type, rows, byte_cols, blob):
    """COLMAP 4.0 cols counts bytes, including for ALIKED float32 data."""
    if feature_type == 0:  # SIFT
        if byte_cols != 128:
            raise ValueError("SIFT descriptors must have 128 bytes per row")
        descriptors = read_array(blob, "u1", rows, byte_cols).astype("<f4") / 512.0
    elif feature_type in (1, 2):  # ALIKED_N16ROT / ALIKED_N32
        if byte_cols != 512:
            raise ValueError("ALIKED descriptors must have 512 bytes per row")
        descriptors = read_array(blob, "<f4", rows, byte_cols // 4)
    else:
        raise ValueError(f"Unsupported COLMAP descriptor type: {feature_type}")
    if not np.isfinite(descriptors).all():
        raise ValueError("Non-finite COLMAP descriptors")
    return descriptors


def rotation_to_quaternion(rotation):
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray([
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        ])
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quaternion = np.asarray([
                (rotation[2, 1] - rotation[1, 2]) / scale,
                0.25 * scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
            ])
        elif axis == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quaternion = np.asarray([
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[0, 1] + rotation[1, 0]) / scale,
                0.25 * scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
            ])
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quaternion = np.asarray([
                (rotation[1, 0] - rotation[0, 1]) / scale,
                (rotation[0, 2] + rotation[2, 0]) / scale,
                (rotation[1, 2] + rotation[2, 1]) / scale,
                0.25 * scale,
            ])
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[0] >= 0.0 else -quaternion


def relative_pose(images, image_id1, image_id2):
    rotation1, translation1, _, _ = images[image_id1]
    rotation2, translation2, _, _ = images[image_id2]
    rotation21 = rotation2 @ rotation1.T
    translation21 = translation2 - rotation21 @ translation1
    translation_norm = np.linalg.norm(translation21)
    if translation_norm > 0.0:
        translation21 /= translation_norm
    return rotation_to_quaternion(rotation21), translation21


def pair_median_angle(pose_images, pose_points, owners, image_id1,
                      image_id2, matches):
    rotation1, translation1, _, _ = pose_images[image_id1]
    rotation2, translation2, _, _ = pose_images[image_id2]
    center1 = -rotation1.T @ translation1
    center2 = -rotation2.T @ translation2
    angles = []
    for idx1, idx2 in matches:
        point_id1 = owners.get((image_id1, int(idx1)))
        point_id2 = owners.get((image_id2, int(idx2)))
        if point_id1 is None or point_id1 != point_id2:
            continue
        xyz = pose_points[point_id1][0]
        ray1 = xyz - center1
        ray2 = xyz - center2
        denominator = np.linalg.norm(ray1) * np.linalg.norm(ray2)
        if denominator <= 0.0:
            continue
        cosine = np.clip(float(ray1 @ ray2 / denominator), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cosine)))
    return float(np.median(angles)) if angles else -1.0


def camera_rays(model, params, xy):
    """Unproject the camera models used by the controlled SfM benchmark."""
    if model in (0, 2, 3):
        norm = (xy - params[1:3]) / params[0]
    elif model == 1:
        norm = (xy - params[2:4]) / params[:2]
    else:
        raise ValueError("--compute_pair_angles supports PINHOLE/SIMPLE_PINHOLE/SIMPLE_RADIAL/RADIAL")
    if model in (2, 3):
        distorted_radius = np.linalg.norm(norm, axis=1)
        radius = distorted_radius.copy()
        k1 = params[3]
        k2 = params[4] if model == 3 else 0.0
        for _ in range(30):
            r2 = radius * radius
            derivative = 1 + 3 * k1 * r2 + 5 * k2 * r2 * r2
            radius -= (radius * (1 + k1 * r2 + k2 * r2 * r2) - distorted_radius) / derivative
        norm *= np.divide(radius, distorted_radius, out=np.ones_like(radius),
                          where=distorted_radius > 0)[:, None]
    rays = np.column_stack((norm, np.ones(len(norm))))
    return rays / np.linalg.norm(rays, axis=1)[:, None]


def relative_pair_median_angle(cameras, image_cameras, keypoints,
                               image_id1, image_id2, matches, qvec, tvec):
    if np.linalg.norm(tvec) <= 1e-12:
        raise ValueError("Relative pose missing: run COLMAP matching with TwoViewGeometry.compute_relative_pose=1")
    rays = []
    for image_id, indices in [(image_id1, matches[:, 0]), (image_id2, matches[:, 1])]:
        model, params = cameras[image_cameras[image_id]]
        rows, cols, blob = keypoints[image_id]
        xy = read_array(blob, "<f4", rows, cols)[indices, :2].astype(np.float64)
        rays.append(camera_rays(model, params, xy))
    rotation = quaternion_rotation(qvec)
    cosines = np.sum(rays[0] * (rays[1] @ rotation), axis=1)
    return float(np.median(np.degrees(np.arccos(np.clip(cosines, -1, 1)))))


def commit_workspace(staging, target):
    # Windows can briefly deny directory rename while an external reader holds
    # a child open. Retry only those OS errors, without copying a partial result
    # or replacing an existing workspace. Permanent permission failures surface.
    for attempt in range(5):
        if target.exists():
            raise FileExistsError(f"Output workspace already exists: {target}")
        try:
            staging.rename(target)
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 4:
                raise
            time.sleep(0.05 * 2 ** attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database_path", dest="database", type=Path, required=True)
    parser.add_argument("--image_path", type=Path, required=True)
    parser.add_argument("--pose_model", type=Path)
    parser.add_argument("--compute_pair_angles", action="store_true",
                        help="Derive pair parallax from database relative poses, without final SfM poses.")
    parser.add_argument("--output_path", dest="workspace", type=Path, required=True)
    parser.add_argument("--descriptor_normalization", dest="sift_normalization",
                        type=str.lower, choices=("l1_root", "l2"), default="l1_root",
                        help="Source SIFT normalization, absent from SQLite; default is COLMAP's l1_root.")
    args = parser.parse_args()

    target = args.workspace.resolve()
    if target.exists():
        raise ValueError("Output workspace must not exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".colite-import-", dir=target.parent) as scratch:
        args.workspace = Path(scratch) / "workspace"
        import_database(args)
        commit_workspace(args.workspace, target)


def import_database(args):
    if args.workspace.exists() and any(args.workspace.iterdir()):
        raise ValueError("Workspace must be empty; existing data is not overwritten")

    with closing(sqlite3.connect(args.database.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        _import_database_snapshot(args, connection)


def _import_database_snapshot(args, connection):
    connection.execute("BEGIN")  # One consistent snapshot for all source tables.
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "rig_sensors" in tables and connection.execute("SELECT 1 FROM rig_sensors LIMIT 1").fetchone():
        raise ValueError("Nontrivial rigs are not supported by this single-camera interchange")
    if "frame_data" in tables:
        if connection.execute("SELECT 1 FROM frame_data GROUP BY frame_id HAVING COUNT(*) != 1 LIMIT 1").fetchone():
            raise ValueError("Multi-image frames are not supported")
        if connection.execute("SELECT 1 FROM frame_data WHERE sensor_type != 0 LIMIT 1").fetchone():
            raise ValueError("Non-camera frame data is not supported")
    pose_images = None
    pose_points = None
    pose_owners = None
    if args.pose_model:
        _, pose_images, pose_points = read_model(args.pose_model)
        pose_owners = {
            observation: point_id
            for point_id, (_, track) in pose_points.items()
            for observation in track
        }
    cameras = list(connection.execute(
        "SELECT camera_id, model, width, height, params, prior_focal_length "
        "FROM cameras ORDER BY camera_id"
    ))
    images = list(connection.execute(
        "SELECT image_id, name, camera_id FROM images ORDER BY image_id"
    ))
    camera_values = {camera_id: (model, np.frombuffer(params_blob, dtype="<f8"))
                     for camera_id, model, _, _, params_blob, _ in cameras}
    image_cameras = {image_id: camera_id for image_id, _, camera_id in images}
    pose_priors = database_pose_priors(connection, tables, image_cameras)

    keypoints_by_image = {
        image_id: (rows, cols, blob)
        for image_id, rows, cols, blob in connection.execute(
            "SELECT image_id, rows, cols, data FROM keypoints")
    }
    descriptor_columns = {row[1] for row in connection.execute("PRAGMA table_info(descriptors)")}
    type_column = "type" if "type" in descriptor_columns else "0"
    descriptors_by_image = {
        image_id: (feature_type, rows, cols, blob)
        for image_id, feature_type, rows, cols, blob in connection.execute(
            f"SELECT image_id, {type_column}, rows, cols, data FROM descriptors")
    }
    # Reject invalid types/lengths before creating output files.
    for image_id, _, _ in images:
        if image_id not in keypoints_by_image:
            raise ValueError(f"Missing keypoints for image {image_id}")
        rows, cols, blob = keypoints_by_image[image_id]
        convert_keypoints(read_array(blob, "<f4", rows, cols))
        if image_id not in descriptors_by_image:
            raise ValueError(f"Missing descriptors for image {image_id}")
        feature_type, descriptor_rows, byte_cols, descriptor_blob = descriptors_by_image[image_id]
        if feature_type not in (0, 1, 2):
            raise ValueError(f"Unsupported COLMAP descriptor type: {feature_type}")
        if descriptor_rows != rows:
            raise ValueError(f"Keypoint/descriptor count mismatch for image {image_id}")
        decode_descriptors(feature_type, descriptor_rows, byte_cols, descriptor_blob)

    args.workspace.mkdir(parents=True, exist_ok=True)
    write_pose_priors(args.workspace / "colmap_pose_priors.tsv", pose_priors)
    features_path = args.workspace / "features"
    matches_path = args.workspace / "matches"
    features_path.mkdir(exist_ok=True)
    matches_path.mkdir(exist_ok=True)

    with (args.workspace / "cameras.txt").open("w", encoding="utf-8") as stream:
        stream.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[] PRIOR_FOCAL\n")
        for camera_id, model, width, height, params_blob, prior in cameras:
            if model not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id: {model}")
            params = np.frombuffer(params_blob, dtype="<f8")
            values = " ".join(format(float(value), ".17g") for value in params)
            stream.write(
                f"{camera_id} {CAMERA_MODELS[model]} {width} {height} "
                f"{values} PRIOR_FOCAL={1 if prior else 0}\n"
            )

    with (args.workspace / "images.tsv").open("w", encoding="utf-8") as stream:
        stream.write("# image_id\tcamera_id\tname\tfile_path\n")
        for image_id, name, camera_id in images:
            stream.write(
                f"{image_id}\t{camera_id}\t{name}\t"
                f"{(args.image_path / name).as_posix()}\n"
            )

    for image_id, _, _ in images:
        rows, cols, blob = keypoints_by_image[image_id]
        source = read_array(blob, "<f4", rows, cols)
        keypoints = convert_keypoints(source)
        feature_type, descriptor_rows, descriptor_cols, descriptor_blob = descriptors_by_image[image_id]
        descriptors = decode_descriptors(feature_type, descriptor_rows, descriptor_cols, descriptor_blob)
        write_feature(features_path / f"{image_id}.bin", keypoints,
                      descriptors, feature_type, args.sift_normalization.upper(), source)

    if "matches" in tables:
        with (matches_path / "raw_matches.tsv").open("w", encoding="utf-8") as stream:
            stream.write("# image_id1\timage_id2\tidx1\tidx2\n")
            for pair_id, rows, cols, blob in connection.execute("SELECT pair_id,rows,cols,data FROM matches ORDER BY pair_id"):
                first, second = decode_pair_id(pair_id)
                if (first >= second or first not in keypoints_by_image or second not in keypoints_by_image or cols != 2):
                    raise ValueError("Invalid raw match pair or dimensions")
                matches = read_array(blob, "<u4", rows, cols)
                if rows and (np.any(matches[:, 0] >= keypoints_by_image[first][0]) or
                             np.any(matches[:, 1] >= keypoints_by_image[second][0])):
                    raise ValueError("Raw match index out of range")
                if not rows: stream.write(f"{first}\t{second}\t-1\t-1\n")
                for idx1, idx2 in matches:
                    stream.write(f"{first}\t{second}\t{int(idx1)}\t{int(idx2)}\n")

    geometry_columns = {row[1] for row in connection.execute("PRAGMA table_info(two_view_geometries)")}
    optional_columns = ",".join(name if name in geometry_columns else "NULL" for name in ("F", "E", "H", "qvec", "tvec"))
    query = "SELECT pair_id,rows,cols,data,config," + optional_columns + " FROM two_view_geometries ORDER BY pair_id"

    def geometries():
        for pair_id, rows, cols, blob, config, f_blob, e_blob, h_blob, q_blob, t_blob in connection.execute(query):
            first, second = decode_pair_id(pair_id)
            if cols != 2: raise ValueError("Expected two verified match columns")
            matches = read_array(blob, "<u4", rows, cols)
            if (q_blob is None) != (t_blob is None):
                raise ValueError("Relative pose requires both qvec and tvec or both NULL")
            record = dict(pair=(first, second), config=config, angle=-1.0, matches=matches)
            for name, payload in (("F", f_blob), ("E", e_blob), ("H", h_blob)):
                record[name] = None if payload is None else read_array(payload, "<f8", 3, 3)
            record["pose"] = None if q_blob is None else np.r_[read_array(q_blob, "<f8", 1, 4)[0], read_array(t_blob, "<f8", 1, 3)[0]]
            if pose_images is not None:
                qvec, tvec = relative_pose(pose_images, first, second)
                record["pose"] = np.r_[qvec, tvec]
                record["angle"] = pair_median_angle(pose_images, pose_points, pose_owners, first, second, matches)
            elif args.compute_pair_angles and config == 2 and len(matches):
                if record["pose"] is None:
                    raise ValueError("Relative pose missing for --compute_pair_angles")
                record["angle"] = relative_pair_median_angle(camera_values, image_cameras, keypoints_by_image,
                                                            first, second, matches, record["pose"][:4], record["pose"][4:])
            yield record
    write_geometries(matches_path / "verified_matches.tsv", geometries(),
                     {image_id: data[0] for image_id, data in keypoints_by_image.items()})


if __name__ == "__main__":
    main()
