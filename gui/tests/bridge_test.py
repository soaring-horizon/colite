"""Run under Linux/WSL: validates cleanup of owned worker groups."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

BRIDGE = Path(__file__).resolve().parents[1] / 'workflow' / 'worker_bridge.py'


class BridgeTest(unittest.TestCase):
    def run_case(self, stale):
        with tempfile.TemporaryDirectory(prefix='colite gui ') as folder:
            root = Path(folder)
            control, pidfile = root / 'control', root / 'pid'
            control.write_text('alive')
            script = 'import os,time;from pathlib import Path;Path(' + repr(str(pidfile)) + ').write_text(str(os.getpid()));time.sleep(30)'
            bridge = subprocess.Popen([sys.executable, str(BRIDGE), '--control', str(control), '--', sys.executable, '-c', script])
            deadline = time.time() + 5
            while not pidfile.exists() and time.time() < deadline:
                time.sleep(.02)
            self.assertTrue(pidfile.exists())
            pid = int(pidfile.read_text())
            if stale:
                os.utime(control, (time.time() - 20, time.time() - 20))
            else:
                control.write_text('abort')
            self.assertEqual(bridge.wait(timeout=6), 130)
            self.assertFalse(Path('/proc', str(pid)).exists())

    def test_abort(self):
        self.run_case(False)

    def test_lost_desktop(self):
        self.run_case(True)


if __name__ == '__main__':
    unittest.main()
