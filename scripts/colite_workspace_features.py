"""Validated Colite feature file IO used by offline COLMAP interchange tools."""
from pathlib import Path
import hashlib
import struct
import numpy as np

MAGIC = 0x434C4654
HEADER = struct.Struct("<IIIIIIQQQQQ")


def checksum(chunks):
    value = 14695981039346656037
    for chunk in chunks:
        for byte in chunk:
            value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def write_feature(path, keypoints, descriptors, feature_type,
                  sift_normalization="L1_ROOT", source_keypoints=None):
    keypoints = np.asarray(keypoints, dtype="<f4", order="C")
    descriptors = np.asarray(descriptors, dtype="<f4", order="C")
    if (feature_type not in (0, 1, 2) or keypoints.ndim != 2 or keypoints.shape[1] != 5 or
            descriptors.ndim != 2 or descriptors.shape[0] not in (0, len(keypoints)) or
            not np.isfinite(keypoints).all() or not np.isfinite(descriptors).all()):
        raise ValueError("Invalid feature dimensions, type or values")
    columns, shape = 0, b""
    if source_keypoints is not None:
        source = np.asarray(source_keypoints, dtype="<f4", order="C")
        if (source.ndim != 2 or len(source) != len(keypoints) or source.shape[1] not in (2, 4, 6) or
                not np.isfinite(source).all() or not np.array_equal(source[:, :2], keypoints[:, :2])):
            raise ValueError("Invalid original keypoint shapes")
        columns = source.shape[1]
        shape = source[:, 2:].tobytes(order="C")
    metric, normalization = ((1, 2 if sift_normalization == "L1_ROOT" else 3)
                             if feature_type == 0 else (2, 3))
    chunks = [keypoints.tobytes(), shape, descriptors.tobytes()]
    header = HEADER.pack(MAGIC, 3, feature_type + 1, 1, metric, normalization,
                         len(keypoints), *descriptors.shape, len(chunks[-1]),
                         checksum([struct.pack("<I", columns), *chunks]))
    Path(path).write_bytes(header + struct.pack("<II", columns, 0) + b"".join(chunks))


def read_feature(path):
    data = Path(path).read_bytes()
    if len(data) < 32:
        raise ValueError("Truncated feature header")
    magic, version = struct.unpack_from("<II", data)
    if magic != MAGIC or version not in (1, 2, 3):
        raise ValueError("Unsupported feature format")
    columns = 0
    if version == 1:
        count, rows, cols = struct.unpack_from("<QQQ", data, 8)
        offset, extractor, scalar, metric, normalization, expected = 32, 1, 1, 1, 2, None
        payload_bytes = rows * cols * 4
    else:
        if len(data) < HEADER.size:
            raise ValueError("Truncated feature header")
        (_, _, extractor, scalar, metric, normalization, count, rows, cols,
         payload_bytes, expected) = HEADER.unpack_from(data)
        offset = HEADER.size
        if version == 3:
            if len(data) < 72:
                raise ValueError("Truncated original shape metadata")
            columns, reserved = struct.unpack_from("<II", data, 64)
            if columns not in (0, 2, 4, 6) or reserved:
                raise ValueError("Invalid original keypoint shape metadata")
            offset += 8
    shape_cols = max(0, columns - 2)
    descriptor_offset = offset + count * (5 + shape_cols) * 4
    if (extractor not in (1, 2, 3) or scalar != 1 or metric not in (1, 2) or normalization not in (1, 2, 3) or
            rows not in (0, count) or payload_bytes != rows * cols * 4 or
            descriptor_offset + payload_bytes != len(data)):
        raise ValueError("Invalid feature metadata, dimensions or payload length")
    if expected is not None:
        chunks = [memoryview(data)[offset:]]
        if version == 3: chunks.insert(0, struct.pack("<I", columns))
        if checksum(chunks) != expected:
            raise ValueError("Feature payload checksum mismatch")
    keypoints = np.frombuffer(data, "<f4", count * 5, offset).reshape(count, 5)
    shapes = np.frombuffer(data, "<f4", count * shape_cols, offset + count * 20).reshape(count, shape_cols)
    descriptors = np.frombuffer(data, "<f4", rows * cols, descriptor_offset).reshape(rows, cols)
    if not all(np.isfinite(a).all() for a in (keypoints, shapes, descriptors)):
        raise ValueError("Non-finite feature data")
    return dict(version=version, extractor=extractor, metric=metric, normalization=normalization,
                keypoints=keypoints, source_columns=columns, shapes=shapes, descriptors=descriptors,
                sha256=hashlib.sha256(data).hexdigest())


def colmap_keypoints(feature):
    keypoints = feature["keypoints"]
    columns = feature["source_columns"]
    if columns == 2: return keypoints[:, :2].copy()
    if columns in (4, 6): return np.column_stack((keypoints[:, :2], feature["shapes"])).astype("<f4")
    result = keypoints[:, :4].copy()
    result[:, 3] = np.deg2rad(result[:, 3])
    return result


def colmap_descriptors(feature):
    data = feature["descriptors"]
    if data.shape != (len(feature["keypoints"]), 128):
        raise ValueError("Export requires one 128-dimensional descriptor per keypoint")
    if feature["extractor"] == 1:
        if feature["metric"] != 1 or feature["normalization"] not in (2, 3):
            raise ValueError("SIFT export requires angular L1_ROOT or L2 descriptors")
        scaled = np.float32(512) * data
        quantized = np.clip(np.floor(scaled + np.float32(0.5)), 0, 255).astype("u1")
        error = np.abs(quantized.astype(np.float32) / np.float32(512) - data)
        stats = dict(max_absolute_error=float(error.max()) if error.size else 0,
                     saturated_values=int(np.count_nonzero((scaled < 0) | (scaled > 255))))
        return quantized, stats
    if feature["metric"] != 2 or feature["normalization"] != 3:
        raise ValueError("ALIKED export requires cosine L2 descriptors")
    return data.astype("<f4", copy=False), dict(max_absolute_error=0, saturated_values=0)
