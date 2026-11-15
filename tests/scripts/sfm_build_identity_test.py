"""Same-version SfM cache transitions between two actual solver builds."""
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

if sys.platform != "linux" or not all(os.environ.get(k) for k in
        ("COLITE_EXE", "COLITE_OTHER_SFM_BUILD_EXE")):
    raise SystemExit(77)
try:
    import numpy as np
except ImportError:
    raise SystemExit(77)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from colite_workspace_features import write_feature
from colite_workspace_geometry import write_geometries


def png(path):
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack(
            '>I', zlib.crc32(kind + payload))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' +
                    chunk(b'IHDR', struct.pack('>IIBBBBB', 1000, 800, 8, 2, 0, 0, 0)) +
                    chunk(b'IDAT', zlib.compress((b'\0' + b'\x80' * 3000) * 800)) +
                    chunk(b'IEND', b''))


def metadata(workspace, stage="sparse"):
    values = {}
    config = None
    for line in (workspace / 'cache' / (stage + '.txt')).read_text().splitlines():
        parts = line.split('\t')
        if parts[0] == 'value':
            values[parts[1]] = parts[2]
        elif parts[0] == 'config_hash':
            config = parts[1]
    return config, values


class SfmBuildIdentityTest(unittest.TestCase):
    def test_solver_change_invalidates_in_both_directions(self):
        binaries = [Path(os.environ[k]).resolve() for k in
                    ('COLITE_EXE', 'COLITE_OTHER_SFM_BUILD_EXE')]
        self.assertTrue(all(p.is_file() for p in binaries))
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workspace, images = base / 'workspace', base / 'images'
            (workspace / 'features').mkdir(parents=True)
            (workspace / 'matches').mkdir()
            images.mkdir()
            (workspace / 'cameras.txt').write_text(
                '1 PINHOLE 1000 800 700 700 500 400 PRIOR_FOCAL=1\n')
            xyz = np.random.default_rng(9).uniform([-.6, -.7, 4], [.6, .7, 8], (80, 3))
            descriptors = np.random.default_rng(37).uniform(1, 1000, (80, 128)).astype('<f4')
            descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
            rows = []
            for image_id in (1, 2, 3):
                local = xyz - [image_id - 1, 0, 0]
                xy = (700 * local[:, :2] / local[:, 2:] + [500, 400]).astype('<f4')
                write_feature(workspace / 'features' / f'{image_id}.bin',
                              np.column_stack([xy, np.ones(80), np.zeros((80, 2))]).astype('<f4'),
                              descriptors, 0, source_keypoints=xy)
                image = images / f'{image_id}.png'
                png(image)
                rows.append(f'{image_id}\t1\t{image_id}.png\t{image}\n')
            (workspace / 'images.tsv').write_text(''.join(rows))
            k = np.array([[700, 0, 500], [0, 700, 400], [0, 0, 1]], float)
            records = []
            for a, b in ((1, 2), (1, 3), (2, 3)):
                d = b - a
                e = np.array([[0, 0, 0], [0, 0, d], [0, -d, 0]], float)
                records.append(dict(pair=(a, b), config=2, angle=20,
                                    matches=np.column_stack([np.arange(80)] * 2).astype('<u4'),
                                    E=e, F=np.linalg.inv(k).T @ e @ np.linalg.inv(k),
                                    pose=(1, 0, 0, 0, -d, 0, 0)))
            write_geometries(workspace / 'matches/verified_matches.tsv', records)
            inputs = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in workspace.rglob('*') if p.is_file()}
            runner = base / 'colite_current'
            env = dict(os.environ, QT_QPA_PLATFORM='offscreen', OMP_NUM_THREADS='2',
                       OPENBLAS_NUM_THREADS='1')
            cold = []
            # Return to the first backend after warming the second one too.
            for index, binary in enumerate((binaries[0], binaries[1], binaries[0])):
                shutil.copy2(binary, runner)
                args = [str(runner), 'mapper', '--Colite.workspace_path', str(workspace),
                        '--Mapper.num_threads', '2', '--Mapper.random_seed', '0',
                        '--Mapper.init_min_num_inliers', '30', '--Mapper.init_min_tri_angle', '5']
                for replay in range(2):
                    p = subprocess.run(args, env=env, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, timeout=120)
                    self.assertEqual(p.returncode, 0, p.stdout)
                    self.assertEqual('Sparse cache hit: 1' in p.stdout, replay == 1, p.stdout)
                    key, info = metadata(workspace)
                    self.assertEqual(info['registered_images'], '3')
                    if replay == 0:
                        cold.append((key, info['pose_solver_backend'], info['implementation_version']))
                    self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest() == h
                                        for p, h in inputs.items()))
            self.assertEqual(cold[0][2], cold[1][2], 'Use two builds of the same source version')
            self.assertNotEqual(cold[0][1], cold[1][1], 'Use genuinely different solver builds')
            self.assertNotEqual(cold[0][0], cold[1][0])
            self.assertEqual(cold[0], cold[2])
            geometry_keys = []
            for index, binary in enumerate((binaries[0], binaries[1], binaries[0])):
                shutil.copy2(binary, runner)
                args = [str(runner), 'exhaustive_matcher', '--Colite.workspace_path', str(workspace),
                        '--FeatureMatching.use_gpu', '0', '--FeatureMatching.num_threads', '2',
                        '--TwoViewGeometry.random_seed', '0']
                for replay in range(2):
                    p = subprocess.run(args, env=env, text=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, timeout=120)
                    self.assertEqual(p.returncode, 0, p.stdout)
                    self.assertEqual('Raw match cache hit: 1' in p.stdout,
                                     index > 0 or replay > 0, p.stdout)
                    self.assertEqual('Geometry cache hit: 1' in p.stdout, replay > 0, p.stdout)
                    if replay == 0:
                        geometry_keys.append(metadata(workspace, 'geometry')[0])
                    rows = [line.split() for line in (workspace / 'matches/raw_matches.tsv').read_text().splitlines()
                            if line and not line.startswith('#')]
                    self.assertEqual(sum(row[2:] != ['-1', '-1'] for row in rows), 240)
                self.assertTrue(all(hashlib.sha256(p.read_bytes()).hexdigest() == h
                                    for p, h in inputs.items() if p.parent.name == 'features'))
            self.assertNotEqual(geometry_keys[0], geometry_keys[1])
            self.assertEqual(geometry_keys[0], geometry_keys[2])



if __name__ == '__main__':
    unittest.main()
