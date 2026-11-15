#!/usr/bin/env python3
"""Check typed COLMAP 4.0 imports, including the native Colite feature reader."""
from contextlib import closing

import os
from pathlib import Path
import sqlite3
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

try:
    import numpy as np
except ImportError:
    print("numpy is required for the COLMAP database import test")
    sys.exit(77)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import convert_colmap_database_to_colite_workspace as importer
from colite_workspace_features import read_feature, colmap_keypoints, colmap_descriptors, checksum
import convert_colite_workspace_to_colmap_database as exporter
from colite_workspace_geometry import read_geometries, write_geometries


class ColmapDatabaseImportTest(unittest.TestCase):
    def test_commit_retries_transient_windows_lock_without_partial_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, target = Path(tmp) / 'staging', Path(tmp) / 'workspace'
            staging.mkdir()
            (staging / 'data').write_bytes(b'complete output')
            original = Path.rename
            locked = PermissionError('temporary Windows reader')
            locked.winerror = 32
            attempts = []
            def rename(source, destination):
                attempts.append(source)
                if len(attempts) == 1:
                    self.assertFalse(target.exists())
                    raise locked
                return original(source, destination)
            with mock.patch.object(Path, 'rename', rename), mock.patch.object(importer.time, 'sleep'):
                importer.commit_workspace(staging, target)
            self.assertEqual((target / 'data').read_bytes(), b'complete output')
            self.assertFalse(staging.exists())

    def test_commit_retains_staging_on_permanent_denial_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            staging, target = Path(tmp) / 'staging', Path(tmp) / 'workspace'
            staging.mkdir()
            (staging / 'data').write_bytes(b'new output')
            denied = PermissionError('denied')
            denied.winerror = 5
            with mock.patch.object(Path, 'rename', side_effect=denied), mock.patch.object(importer.time, 'sleep'):
                with self.assertRaises(PermissionError):
                    importer.commit_workspace(staging, target)
            self.assertFalse(target.exists())
            self.assertEqual((staging / 'data').read_bytes(), b'new output')
            target.mkdir()
            (target / 'data').write_bytes(b'keep existing')
            with self.assertRaises(FileExistsError):
                importer.commit_workspace(staging, target)
            self.assertEqual((target / 'data').read_bytes(), b'keep existing')

    def fixture(self, path, feature_type=0, malformed=False, columns=6):
        source = np.array([[10 + i, 20 + i, 0, -4, 2, 0]
                           for i in range(8)], dtype="<f4")
        if columns == 2:
            source = source[:, :2].copy()
        elif columns == 4:
            source = np.column_stack((source[:, :2], np.full(8, 3, dtype="<f4"),
                                      np.full(8, -1.234567, dtype="<f4")))
        descriptors = np.eye(8, 128, dtype="<f4")
        if feature_type == 0:
            descriptors = (descriptors * 255).astype("u1")
        raw = descriptors.tobytes()
        with closing(sqlite3.connect(path)) as database, database:
            database.executescript("""
                CREATE TABLE cameras(camera_id,model,width,height,params,prior_focal_length);
                CREATE TABLE images(image_id,name,camera_id);
                CREATE TABLE keypoints(image_id,rows,cols,data);
                CREATE TABLE descriptors(image_id,type,rows,cols,data);
                CREATE TABLE two_view_geometries(pair_id,rows,cols,data,config,qvec,tvec);
            """)
            database.execute("INSERT INTO cameras VALUES(3,0,640,480,?,1)",
                             (struct.pack("<ddd", 500, 320, 240),))
            for image_id in (7, 23):
                database.execute("INSERT INTO images VALUES(?,?,3)",
                                 (image_id, f"subfolder/{image_id}.jpg"))
                database.execute("INSERT INTO keypoints VALUES(?,8,?,?)",
                                 (image_id, columns, source.tobytes()))
                database.execute("INSERT INTO descriptors VALUES(?,?,8,?,?)",
                                 (image_id, feature_type, descriptors.shape[1] * descriptors.itemsize,
                                  raw[:-1] if malformed else raw))
        return source, descriptors

    def run_import(self, database, workspace):
        args = [sys.executable, str(ROOT / "scripts/convert_colmap_database_to_colite_workspace.py"),
                "--database_path", str(database), "--image_path", str(database.parent / "images"),
                "--output_path", str(workspace)]
        return subprocess.run(args, capture_output=True, text=True)

    def test_default_import_rejects_missing_descriptors_without_partial_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / 'source.db')
            with closing(sqlite3.connect(root / 'source.db')) as db, db:
                db.execute('DELETE FROM descriptors WHERE image_id=23')
            before = (root / 'source.db').read_bytes()
            result = self.run_import(root / 'source.db', root / 'workspace')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing descriptors for image 23', result.stderr)
            self.assertFalse((root / 'workspace').exists())
            self.assertEqual((root / 'source.db').read_bytes(), before)

    def test_empty_images_keep_typed_empty_descriptors_by_default(self):
        for feature_type in (0, 1, 2):
            with self.subTest(feature_type=feature_type), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root / 'source.db', feature_type=feature_type, columns=4)
                with closing(sqlite3.connect(root / 'source.db')) as db, db:
                    db.execute('UPDATE keypoints SET rows=0, data=NULL')
                    db.execute('UPDATE descriptors SET rows=0, data=NULL')
                result = self.run_import(root / 'source.db', root / 'workspace')
                self.assertEqual(result.returncode, 0, result.stderr)
                feature = read_feature(root / 'workspace/features/7.bin')
                self.assertEqual(feature['extractor'], feature_type + 1)
                self.assertEqual(feature['descriptors'].shape, (0, 128))
                data, _ = colmap_descriptors(feature)
                self.assertEqual(data.shape, (0, 128))
                self.assertEqual(data.dtype, np.dtype('u1' if feature_type == 0 else '<f4'))

    def test_typed_import_is_readable_by_native_matching(self):
        executable = os.environ.get("COLITE_EXE")
        self.assertTrue(executable and Path(executable).is_file(), "Set COLITE_EXE to the native executable")
        for feature_type in (0, 1, 2):
            with self.subTest(feature_type=feature_type), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, descriptors = self.fixture(root / "source.db", feature_type)
                workspace = root / "workspace"
                imported = self.run_import(root / "source.db", workspace)
                self.assertEqual(imported.returncode, 0, imported.stderr)
                payload = (workspace / "features/7.bin").read_bytes()
                header = struct.unpack_from("<IIIIIIQQQQQ", payload)
                self.assertEqual(header[:4], (0x434C4654, 3, feature_type + 1, 1))
                self.assertEqual(header[6:10], (8, 8, 128, 8 * 128 * 4))
                keypoints = np.frombuffer(payload, "<f4", 8 * 5, 72).reshape(8, 5)
                np.testing.assert_array_equal(keypoints[:, :2], source[:, :2])
                np.testing.assert_allclose(keypoints[:, 2], 3.0)
                np.testing.assert_allclose(keypoints[:, 3], 90.0)
                actual = np.frombuffer(payload, "<f4", offset=72 + 8 * 9 * 4).reshape(8, 128)
                np.testing.assert_array_equal(actual, descriptors / 512 if feature_type == 0 else descriptors)
                np.testing.assert_array_equal(colmap_keypoints(read_feature(workspace / "features/7.bin")), source)
                # Native reader checks the checksum and feature-family contract;
                # matching consumes the descriptors, not just our Python decoder.
                matcher = "SIFT_BRUTEFORCE" if feature_type == 0 else "ALIKED_BRUTEFORCE"
                matched = subprocess.run([executable, "exhaustive_matcher", "--Colite.workspace_path",
                                          str(workspace), "--FeatureMatching.type", matcher],
                                         capture_output=True, text=True)
                self.assertEqual(matched.returncode, 0, matched.stderr)
                self.assertTrue((workspace / "matches/raw_matches.tsv").is_file())

    def test_unsupported_type_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "source.db", 99)
            result = self.run_import(root / "source.db", root / "workspace")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Unsupported COLMAP descriptor type", result.stderr)
            self.assertFalse((root / "workspace").exists())

    def test_truncated_blob_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "source.db", 1, malformed=True)
            result = self.run_import(root / "source.db", root / "workspace")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Invalid database blob length", result.stderr)
            self.assertFalse((root / "workspace").exists())

    def test_similarity_orientation_units(self):
        source = np.array([[12.5, 24.5, 2.0, -np.pi / 2]], dtype="<f4")
        actual = importer.convert_keypoints(source)
        np.testing.assert_array_equal(actual[0, :3], source[0, :3])
        self.assertAlmostEqual(float(actual[0, 3]), 270.0, places=4)

    def test_source_sift_l2_metadata_preserves_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feature.bin"
            keypoints = np.zeros((1, 5), dtype="<f4")
            descriptors = np.full((1, 128), 127 / 512, dtype="<f4")
            importer.write_feature(path, keypoints, descriptors, 0, "L2")
            payload = path.read_bytes()
            self.assertEqual(struct.unpack_from("<IIIIII", payload)[-1], 3)
            self.assertEqual(payload[72 + 20:], descriptors.tobytes())

    def test_existing_workspace_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "source.db")
            workspace = root / "workspace"
            workspace.mkdir()
            sentinel = workspace / "keep.bin"
            sentinel.write_bytes(b"existing workspace")
            result = self.run_import(root / "source.db", workspace)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_bytes(), b"existing workspace")

    def test_original_columns_and_sift_rounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feature.bin"
            descriptors = np.zeros((1, 128), dtype="<f4")
            descriptors[0, :4] = [0, 0.5 / 512, 254.5 / 512, 1]
            for source in (np.array([[0.5, 2.5]], dtype="<f4"),
                           np.array([[0.5, 2.5, 1.25, -1.234567]], dtype="<f4"),
                           np.array([[0.5, 2.5, 0.5, -3, 2, 0.25]], dtype="<f4")):
                importer.write_feature(path, importer.convert_keypoints(source), descriptors, 0,
                                       source_keypoints=source)
                feature = read_feature(path)
                np.testing.assert_array_equal(colmap_keypoints(feature), source)
                encoded, stats = colmap_descriptors(feature)
                np.testing.assert_array_equal(encoded[0, :4], [0, 1, 255, 255])
                self.assertEqual(stats["saturated_values"], 1)

    def test_raw_pair_orientation_empty_and_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matches.tsv"
            path.write_text("23 7 4 2\n7 31 -1 -1\n")
            pairs = list(exporter.match_pairs(path, {7: 5, 23: 6, 31: 0}))
            self.assertEqual(pairs[0][0], (7, 23))
            np.testing.assert_array_equal(pairs[0][1], [[2, 4]])
            self.assertEqual(pairs[1][1].shape, (0, 2))
            path.write_text("7 23 5 1\n")
            with self.assertRaisesRegex(ValueError, "out of range"):
                list(exporter.match_pairs(path, {7: 5, 23: 6}))

    def test_invalid_matches_fail_without_partial_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "source.db")
            with closing(sqlite3.connect(root / "source.db")) as db, db:
                db.execute("CREATE TABLE matches(pair_id,rows,cols,data)")
                db.execute("INSERT INTO matches VALUES(?,1,2,?)", (7 * importer.MAX_IMAGE_ID + 23,
                           struct.pack("<II", 99, 0)))
            result = self.run_import(root / "source.db", root / "workspace")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "workspace").exists())

    def test_v2_feature_reader_remains_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.bin"
            keypoints = np.array([[0.5, 1.5, 2, 90, 0]], dtype="<f4")
            descriptors = np.zeros((1, 128), dtype="<f4")
            payload = keypoints.tobytes() + descriptors.tobytes()
            path.write_bytes(struct.pack("<IIIIIIQQQQQ", 0x434C4654, 2, 1, 1, 1, 2, 1, 1, 128,
                                         descriptors.nbytes, checksum([payload])) + payload)
            feature = read_feature(path)
            np.testing.assert_array_equal(feature["keypoints"], keypoints)
            np.testing.assert_array_equal(feature["descriptors"], descriptors)
            workspace = Path(tmp) / "workspace"
            (workspace / "features").mkdir(parents=True)
            shutil.copy2(path, workspace / "features/7.bin")
            shutil.copy2(path, workspace / "features/23.bin")
            (workspace / "cameras.txt").write_text("3 SIMPLE_PINHOLE 64 64 50 32 32\n")
            (workspace / "images.tsv").write_text("7\t3\t7.jpg\t/no-images/7.jpg\n23\t3\t23.jpg\t/no-images/23.jpg\n")
            native = subprocess.run([os.environ["COLITE_EXE"], "exhaustive_matcher", "--Colite.workspace_path", str(workspace),
                                     "--FeatureMatching.use_gpu", "0"], capture_output=True, text=True)
            self.assertEqual(native.returncode, 0, native.stderr)

    @unittest.skipUnless(os.environ.get("COLMAP_EXE"), "Set COLMAP_EXE for native schema/roundtrip verification")
    def test_native_database_feature_roundtrip(self):
        for feature_type, columns in ((0, 2), (0, 4), (0, 6), (1, 6), (2, 6)):
            with self.subTest(feature_type=feature_type, columns=columns), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source, descriptors = self.fixture(root / "source.db", feature_type, columns=columns)
                with closing(sqlite3.connect(root / "source.db")) as db, db:
                    db.execute("CREATE TABLE matches(pair_id,rows,cols,data)")
                    db.execute("INSERT INTO matches VALUES(?,2,2,?)", (7 * importer.MAX_IMAGE_ID + 23,
                               struct.pack("<IIII", 0, 2, 4, 5)))
                imported = self.run_import(root / "source.db", root / "workspace")
                self.assertEqual(imported.returncode, 0, imported.stderr)
                target = root / "export.db"
                report = exporter.export_workspace(root / "workspace", target, os.environ["COLMAP_EXE"])
                self.assertEqual(report["raw_matches"], 2)
                with closing(sqlite3.connect(target)) as db, db:
                    rows, cols, blob = db.execute("SELECT rows,cols,data FROM keypoints WHERE image_id=7").fetchone()
                    self.assertEqual((rows, cols), source.shape)
                    np.testing.assert_array_equal(np.frombuffer(blob, "<f4").reshape(rows, cols), source)
                    typ, rows, cols, blob = db.execute("SELECT type,rows,cols,data FROM descriptors WHERE image_id=7").fetchone()
                    self.assertEqual(typ, feature_type)
                    self.assertEqual(blob, descriptors.tobytes())
                    self.assertEqual(db.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0], 0)
                returned = self.run_import(target, root / "returned")
                self.assertEqual(returned.returncode, 0, returned.stderr)
                self.assertEqual((root / "workspace/features/7.bin").read_bytes(),
                                 (root / "returned/features/7.bin").read_bytes())
                consumed = root / "consumed.db"
                shutil.copy2(target, consumed)
                with closing(sqlite3.connect(consumed)) as db, db:
                    db.execute("DELETE FROM matches")
                matcher = "SIFT_BRUTEFORCE" if feature_type == 0 else "ALIKED_BRUTEFORCE"
                native = subprocess.run([os.environ["COLMAP_EXE"], "exhaustive_matcher", "--database_path", str(consumed),
                    "--FeatureMatching.type", matcher, "--FeatureMatching.use_gpu", "0", "--FeatureMatching.num_threads", "1"],
                    capture_output=True, text=True)
                self.assertEqual(native.returncode, 0, native.stdout + native.stderr)
                before = target.read_bytes()
                with self.assertRaisesRegex(ValueError, "must not exist"):
                    exporter.export_workspace(root / "workspace", target, os.environ["COLMAP_EXE"])
                self.assertEqual(target.read_bytes(), before)

    @unittest.skipUnless(os.environ.get("COLMAP_EXE"), "Set COLMAP_EXE for verified database roundtrip")
    def test_verified_geometry_roundtrip_preserves_config_matrices_and_nulls(self):
        for config in range(10):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root / "source.db")
                f = np.zeros((3, 3), dtype="<f8").tobytes() if config % 2 else None
                e = np.arange(9, dtype="<f8").tobytes() if config % 3 else None
                h = np.eye(3, dtype="<f8").tobytes() if config % 4 else None
                q = np.asarray((1, 0, 0, 0), dtype="<f8").tobytes() if config % 2 else None
                t = np.zeros(3, dtype="<f8").tobytes() if q else None
                indices = np.asarray([[1, 3], [7, 0]], dtype="<u4") if config > 1 else np.empty((0, 2), dtype="<u4")
                with closing(sqlite3.connect(root / "source.db")) as db, db:
                    for name in ("F", "E", "H"): db.execute("ALTER TABLE two_view_geometries ADD COLUMN " + name)
                    db.execute("INSERT INTO two_view_geometries(pair_id,rows,cols,data,config,F,E,H,qvec,tvec) VALUES(?,?,2,?,?,?,?,?,?,?)",
                        (7 * importer.MAX_IMAGE_ID + 23, len(indices), indices.tobytes(), config, f, e, h, q, t))
                imported = self.run_import(root / "source.db", root / "workspace")
                self.assertEqual(imported.returncode, 0, imported.stderr)
                record = list(read_geometries(root / "workspace/matches/verified_matches.tsv", {7: 8, 23: 8}))[0]
                self.assertEqual(record["config"], config)
                self.assertEqual(record["pose"] is None, q is None)
                report = exporter.export_workspace(root / "workspace", root / "result.db", os.environ["COLMAP_EXE"])
                self.assertEqual(report["verified_geometries"], 1)
                with closing(sqlite3.connect(root / "result.db")) as db, db:
                    actual = db.execute("SELECT rows,cols,data,config,F,E,H,qvec,tvec FROM two_view_geometries").fetchone()
                    self.assertEqual(actual, (len(indices), 2, indices.tobytes(), config, f, e, h, q, t))
                returned = self.run_import(root / "result.db", root / "returned")
                self.assertEqual(returned.returncode, 0, returned.stderr)
                self.assertEqual((root / "workspace/matches/verified_matches.tsv").read_bytes(),
                                 (root / "returned/matches/verified_matches.tsv").read_bytes())

    def test_verified_geometry_rejects_partial_pose_and_out_of_range_inliers(self):
        for partial_pose in (False, True):
            with self.subTest(partial_pose=partial_pose), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.fixture(root / "source.db")
                with closing(sqlite3.connect(root / "source.db")) as db, db:
                    db.execute("INSERT INTO two_view_geometries VALUES(?,1,2,?,2,?,NULL)",
                        (7 * importer.MAX_IMAGE_ID + 23, struct.pack("<II", 0 if partial_pose else 8, 1),
                         struct.pack("<dddd", 1, 0, 0, 0) if partial_pose else None))
                imported = self.run_import(root / "source.db", root / "workspace")
                self.assertNotEqual(imported.returncode, 0)
                self.assertIn("both qvec" if partial_pose else "out of range", imported.stderr)
                self.assertFalse((root / "workspace").exists())

    @unittest.skipUnless(os.environ.get("COLMAP_EXE"), "Set COLMAP_EXE for legacy feature-only export")
    def test_legacy_verified_requires_explicit_feature_only_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "source.db")
            imported = self.run_import(root / "source.db", root / "workspace")
            self.assertEqual(imported.returncode, 0, imported.stderr)
            (root / "workspace/matches/verified_matches.tsv").write_text("7\t23\t2\t-1\t1\t0\t0\t0\t0\t0\t0\t0\t1\n")
            with self.assertRaisesRegex(ValueError, "Legacy verified"):
                exporter.export_workspace(root / "workspace", root / "bad.db", os.environ["COLMAP_EXE"])
            self.assertFalse((root / "bad.db").exists())
            report = exporter.export_workspace(root / "workspace", root / "good.db", os.environ["COLMAP_EXE"], False)
            self.assertEqual(report["verified_geometries"], 0)
            self.assertTrue(report["geometric_verification_required"])

    def test_verified_codec_rejects_count_flag_and_prefix_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "verified.tsv"
            record = dict(pair=(7, 23), config=2, angle=-1, F=None, E=np.eye(3), H=None,
                          pose=None, matches=np.asarray([[0, 1], [1, 2]], dtype="<u4"))
            write_geometries(path, [record])
            original = path.read_text()
            for corrupted in (original.rsplit("\n", 2)[0] + "\n",
                              original.replace("# pair 7 23 2 2", "# pair 7 23 2 0"),
                              original.replace("7\t23\t2\t-1", "7\t23\t3\t-1")):
                path.write_text(corrupted)
                with self.assertRaises(ValueError): list(read_geometries(path))

    def test_diagnostic_edits_keep_modern_geometry_metadata_consistent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            (workspace / "matches").mkdir(parents=True)
            source = workspace / "matches/verified_matches.tsv"
            matrix = np.arange(9, dtype="<f8").reshape(3, 3)
            write_geometries(source, [dict(pair=(7, 23), config=2, angle=3.5, F=None, E=matrix, H=None,
                pose=None, matches=np.asarray([[0, 0], [0, 1]], dtype="<u4")),
                dict(pair=(7, 42), config=0, angle=-1, F=None, E=None, H=None, pose=None,
                     matches=np.empty((0, 2), dtype="<u4"))])
            filtered = root / "filtered.tsv"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/filter_verified_cycle_consistency.py"),
                                     "--input", str(source), "--output", str(filtered)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            records = list(read_geometries(filtered))
            self.assertEqual(len(records), 2)
            self.assertEqual(len(records[0]["matches"]), 1)
            self.assertEqual(records[0]["angle"], -1)
            np.testing.assert_array_equal(records[0]["E"], matrix)
            self.assertIsNone(records[0]["pose"])
            self.assertEqual(len(records[1]["matches"]), 0)
            model = root / "model"
            model.mkdir()
            for name in ("cameras.txt", "images.txt", "points3D.txt"): (model / name).write_text("")
            completed = root / "completed.tsv"
            result = subprocess.run([sys.executable, str(ROOT / "scripts/complete_verified_cycles.py"),
                                     "--workspace", str(workspace), "--model", str(model), "--output", str(completed)],
                                     capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            records = list(read_geometries(completed))
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["angle"], 3.5)
            self.assertEqual(len(records[0]["matches"]), 2)
            np.testing.assert_array_equal(records[0]["E"], matrix)
            from analyze_sfm_edge_support import read_edges
            self.assertEqual(len(read_edges(completed)), 2)

    def test_nontrivial_rig_rejected_without_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "source.db")
            with closing(sqlite3.connect(root / "source.db")) as db, db:
                db.execute("CREATE TABLE rig_sensors(rig_id,sensor_id,sensor_type,sensor_from_rig)")
                db.execute("INSERT INTO rig_sensors VALUES(1,3,0,NULL)")
            result = self.run_import(root / "source.db", root / "workspace")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Nontrivial rigs", result.stderr)
            self.assertFalse((root / "workspace").exists())

    def pose_prior_fixture(self, path):
        self.fixture(path)
        # Include non-default NaN bit patterns and NULLs to check byte preservation.
        nan = struct.pack("<Q", 0x7ff8000000000042)
        rows = [(101, 7, 3, 0, nan * 3, nan * 9, struct.pack("<ddd", 0, 1, 0), -1),
                (104, 23, 3, 0, None, None, None, -1)]
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("CREATE TABLE pose_priors(" + ",".join(importer.POSE_PRIOR_COLUMNS) + ")")
            db.executemany("INSERT INTO pose_priors VALUES(?,?,?,?,?,?,?,?)", rows)
        return rows

    @unittest.skipUnless(os.environ.get("COLMAP_EXE"), "Set COLMAP_EXE for prior metadata interchange")
    def test_exif_gravity_metadata_roundtrip_is_lossless(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = self.pose_prior_fixture(root / "source.db")
            before = (root / "source.db").read_bytes()
            result = self.run_import(root / "source.db", root / "workspace")
            self.assertEqual(result.returncode, 0, result.stderr)
            report = exporter.export_workspace(root / "workspace", root / "result.db", os.environ["COLMAP_EXE"])
            self.assertEqual(report["pose_prior_metadata"], 2)
            with closing(sqlite3.connect(root / "result.db")) as db, db:
                actual = list(db.execute("SELECT " + ",".join(importer.POSE_PRIOR_COLUMNS) +
                                         " FROM pose_priors ORDER BY pose_prior_id"))
            self.assertEqual(actual, expected)
            result = self.run_import(root / "result.db", root / "returned")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((root / "workspace/colmap_pose_priors.tsv").read_bytes(),
                             (root / "returned/colmap_pose_priors.tsv").read_bytes())
            self.assertEqual((root / "source.db").read_bytes(), before)

    def test_active_or_invalid_prior_rejected_atomically(self):
        updates = [("position", struct.pack("<ddd", 1, 2, 3)),
                   ("position_covariance", np.eye(3, dtype="<f8").tobytes()),
                   ("coordinate_system", 0), ("corr_sensor_id", 99), ("corr_data_id", 99),
                   ("corr_sensor_type", 1), ("gravity", b"bad"),
                   ("gravity", struct.pack("<ddd", 1, float('nan'), 0))]
        for field, value in updates:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                self.pose_prior_fixture(root / "source.db")
                with closing(sqlite3.connect(root / "source.db")) as db, db:
                    db.execute("UPDATE pose_priors SET " + field + "=? WHERE pose_prior_id=101", (value,))
                before = (root / "source.db").read_bytes()
                result = self.run_import(root / "source.db", root / "workspace")
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse((root / "workspace").exists())
                self.assertEqual((root / "source.db").read_bytes(), before)

    def test_corrupt_pose_metadata_export_rejected_before_database_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.pose_prior_fixture(root / "source.db")
            result = self.run_import(root / "source.db", root / "workspace")
            self.assertEqual(result.returncode, 0, result.stderr)
            path = root / "workspace/colmap_pose_priors.tsv"
            original = path.read_text()
            for corrupted in (original.replace("colmap_pose_priors_v1", "colmap_pose_priors_v9"),
                              original + original.splitlines()[1] + "\n",
                              original.replace("101\t7\t3\t0", "101\t7\t9\t0"),
                              original.replace("NULL\tNULL\tNULL", "bad\tNULL\tNULL")):
                path.write_text(corrupted)
                with self.assertRaises(ValueError):
                    exporter.export_workspace(root / "workspace", root / "bad.db", root / "not-needed")
                self.assertFalse((root / "bad.db").exists())

    def test_missing_export_program_leaves_no_partial_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root / "source.db")
            imported = self.run_import(root / "source.db", root / "workspace")
            self.assertEqual(imported.returncode, 0, imported.stderr)
            with self.assertRaises(FileNotFoundError):
                exporter.export_workspace(root / "workspace", root / "result.db", root / "missing-colmap")
            self.assertFalse((root / "result.db").exists())
            self.assertFalse(list(root.glob(".colite-export-*")))

    def test_quantization_invalidates_matching_cache_and_keeps_affine_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = self.fixture(root / "source.db")
            imported = self.run_import(root / "source.db", root / "workspace")
            self.assertEqual(imported.returncode, 0, imported.stderr)
            args = [os.environ["COLITE_EXE"], "exhaustive_matcher", "--Colite.workspace_path", str(root / "workspace"),
                    "--FeatureMatching.use_gpu", "0"]
            for expected in ("0", "1"):
                matched = subprocess.run(args, capture_output=True, text=True)
                self.assertEqual(matched.returncode, 0, matched.stderr)
                self.assertIn("Raw match cache hit: " + expected, matched.stdout)
            converted = subprocess.run([sys.executable, str(ROOT / "scripts/quantize_colite_descriptors.py"),
                                        "--workspace", str(root / "workspace")], capture_output=True, text=True)
            self.assertEqual(converted.returncode, 0, converted.stderr)
            matched = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(matched.returncode, 0, matched.stderr)
            self.assertIn("Raw match cache hit: 0", matched.stdout)
            self.assertIn("Geometry cache hit: 0", matched.stdout)
            np.testing.assert_array_equal(colmap_keypoints(read_feature(root / "workspace/features/7.bin")), source)


if __name__ == "__main__":
    unittest.main()
