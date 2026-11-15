"""Versioned verified correspondences with COLMAP's optional geometry semantics.

The ordinary 13-column inlier rows remain usable by diagnostic TSV readers.
Authoritative pair records store matrices in row-major order and presence bits,
including empty/degenerate pairs. Legacy TSVs cannot establish these semantics.
"""
from pathlib import Path
import numpy as np

HEADER = "# COLITE_VERIFIED_GEOMETRIES 2"
MAX_IMAGE_ID = 2_147_483_647


def validate(record, counts=None):
    first, second = record["pair"]
    if not 0 <= first < second < MAX_IMAGE_ID or record["config"] not in range(10):
        raise ValueError("Invalid verified pair or COLMAP configuration")
    if not np.isfinite(record["angle"]):
        raise ValueError("Non-finite triangulation angle")
    for key, shape in (("F", (3, 3)), ("E", (3, 3)), ("H", (3, 3)), ("pose", (7,))):
        value = record.get(key)
        if value is not None and (np.shape(value) != shape or not np.isfinite(value).all()):
            raise ValueError("Invalid optional geometry: " + key)
    if record.get("pose") is not None and np.linalg.norm(record["pose"][:4]) <= 1e-12:
        raise ValueError("Invalid relative quaternion")
    matches = np.asarray(record["matches"])
    if matches.ndim != 2 or matches.shape[1] != 2 or matches.dtype.kind not in "ui" or (
            matches.size and (matches.min() < 0 or matches.max() >= 2**32 - 1)):
        raise ValueError("Invalid verified match indices")
    if counts is not None:
        if first not in counts or second not in counts:
            raise ValueError("Verified geometry references missing image")
        if len(matches) and (np.any(matches[:, 0] >= counts[first]) or np.any(matches[:, 1] >= counts[second])):
            raise ValueError("Verified match index out of range")


def prefix(record):
    pose = record.get("pose")
    if pose is None: pose = (1, 0, 0, 0, 0, 0, 0)
    return "\t".join(map(str, (*record["pair"], record["config"]))) + "\t" + "\t".join(
        format(float(v), ".17g") for v in (record["angle"], *pose)) + "\t"


def write_geometries(path, records, counts=None):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(HEADER + "\n# Columns: pair ID1 ID2 CONFIG FLAGS MEDIAN_ANGLE NUM_INLIERS F[9] E[9] H[9] QW QX QY QZ TX TY TZ\n")
        previous = None
        for record in records:
            validate(record, counts)
            if previous is not None and record["pair"] <= previous:
                raise ValueError("Verified pairs must be unique and sorted")
            previous = record["pair"]
            flags = sum(bit for key, bit in (("F", 1), ("E", 2), ("H", 4), ("pose", 8)) if record.get(key) is not None)
            values = []
            for key in ("F", "E", "H"):
                value = record.get(key)
                values.extend(np.zeros(9) if value is None else np.asarray(value).ravel(order="C"))
            pose = record.get("pose")
            values.extend((1, 0, 0, 0, 0, 0, 0) if pose is None else pose)
            stream.write("# pair " + " ".join(map(str, (*record["pair"], record["config"], flags))) +
                         f" {record['angle']:.17g} {len(record['matches'])} " +
                         " ".join(format(float(v), ".17g") for v in values) + "\n")
            row_prefix = prefix(record)
            if not len(record["matches"]): stream.write(row_prefix + "-1\t-1\n")
            stream.writelines(row_prefix + f"{int(a)}\t{int(b)}\n" for a, b in record["matches"])


def read_geometries(path, counts=None):
    """Stream modern pairs; refuse lossy legacy data rather than fabricate flags."""
    if not Path(path).exists(): return
    modern, record, values, empty, declared = False, None, [], False, 0
    previous_pair, row_prefix = None, None

    def finish():
        if record is None: return None
        if len(values) != declared or (declared == 0 and not empty):
            raise ValueError("Verified inlier count/sentinel mismatch")
        record["matches"] = np.asarray(values, dtype="<u4").reshape(-1, 2)
        validate(record, counts)
        return record

    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            line = line.rstrip("\r\n")
            if not line: continue
            if line.startswith("# COLITE_VERIFIED_GEOMETRIES"):
                if line != HEADER or modern or record is not None:
                    raise ValueError("Invalid verified geometry version")
                modern = True
                continue
            if line.startswith("# pair "):
                if not modern: raise ValueError("Missing verified geometry version")
                completed = finish()
                if completed is not None: yield completed
                row = line.split()[2:]
                if len(row) != 40: raise ValueError("Invalid verified pair metadata")
                first, second, config, flags = map(int, row[:4])
                angle, declared = float(row[4]), int(row[5])
                if not 0 <= flags <= 15 or declared < 0: raise ValueError("Invalid geometry flags/count")
                pair = (first, second)
                if previous_pair is not None and pair <= previous_pair: raise ValueError("Unordered/duplicate verified pair")
                previous_pair = pair
                numbers = np.asarray(row[6:], dtype="<f8")
                if not np.isfinite(numbers).all(): raise ValueError("Non-finite geometry payload")
                record = dict(pair=pair, config=config, angle=angle)
                for index, key in enumerate(("F", "E", "H")):
                    matrix = numbers[index * 9:(index + 1) * 9].reshape(3, 3)
                    record[key] = matrix.copy() if flags & (1 << index) else None
                    if record[key] is None and np.any(matrix): raise ValueError("Nonzero absent matrix")
                record["pose"] = numbers[27:].copy() if flags & 8 else None
                if record["pose"] is None and not np.array_equal(numbers[27:], (1, 0, 0, 0, 0, 0, 0)):
                    raise ValueError("Invalid absent pose placeholder")
                values, empty, row_prefix = [], False, None
                continue
            if line.startswith("#"): continue
            if not modern:
                raise ValueError("Legacy verified TSV lacks F/E/H and presence flags; rerun geometry or export with --Colite.export_verified 0")
            if record is None: raise ValueError("Missing verified pair metadata")
            # Parse the repeated metadata once per pair, then only indices.
            row = line.rsplit("\t", 2)
            if len(row) != 3: raise ValueError("Invalid verified inlier row")
            if row_prefix is None:
                fields = row[0].split()
                if len(fields) != 11 or tuple(map(int, fields[:2])) != record["pair"] or int(fields[2]) != record["config"]:
                    raise ValueError("Conflicting verified inlier metadata")
                expected_pose = record["pose"]
                if expected_pose is None: expected_pose = (1, 0, 0, 0, 0, 0, 0)
                if not np.array_equal(np.asarray(fields[3:], dtype="<f8"), (record["angle"], *expected_pose)):
                    raise ValueError("Conflicting verified inlier pose/angle")
                row_prefix = row[0]
            elif row[0] != row_prefix:
                raise ValueError("Conflicting verified inlier metadata")
            a, b = int(row[1]), int(row[2])
            if a == b == -1:
                if empty or values: raise ValueError("Invalid empty verified pair")
                empty = True
            else:
                if empty or not (0 <= a < 2**32 - 1 and 0 <= b < 2**32 - 1):
                    raise ValueError("Invalid verified match index")
                values.append((a, b))
                if len(values) > declared: raise ValueError("Too many verified inliers")
    completed = finish()
    if completed is not None: yield completed


def rewrite_inlier_rows(source, destination, rows):
    """Rewrite diagnostic edits while retaining optional pair metadata/empty pairs."""
    records = {record["pair"]: record for record in read_geometries(source)}
    matches = {pair: [] for pair in records}
    for line in rows:
        fields = line.split()
        if len(fields) != 13: raise ValueError("Expected 13-column verified inlier row")
        pair = tuple(map(int, fields[:2]))
        a, b = int(fields[11]), int(fields[12])
        if pair not in records: raise ValueError("Edited row has no original pair metadata")
        if a == b == -1: continue
        if not (0 <= a < 2**32 - 1 and 0 <= b < 2**32 - 1):
            raise ValueError("Invalid edited match indices")
        matches[pair].append((a, b))
    for pair, record in records.items():
        updated = np.asarray(matches[pair], dtype="<u4").reshape(-1, 2)
        if not np.array_equal(updated, record["matches"]): record["angle"] = -1.0
        record["matches"] = updated
    write_geometries(destination, records.values())
