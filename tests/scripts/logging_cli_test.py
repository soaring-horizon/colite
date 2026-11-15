"""Exercise global logging through the actual installed/executable entry point."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

BINARY = os.environ.get("COLITE_EXE", "")

@unittest.skipUnless(BINARY, "COLITE_EXE is required")
class LoggingCliTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([BINARY, *map(str, args)], text=True, capture_output=True)

    def test_standard_options_dispatch_anywhere_and_equals(self):
        for args in (("--log_target", "stderr", "--log_level", "2", "mapper", "--help"),
                     ("mapper", "--log_target=stdout", "--log_level=-1", "--log_color=0", "--log_severity=0", "--help")):
            result = self.run_cli(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--Mapper.ba_local_max_refinements", result.stdout)

    def test_targets_and_severity_filter_actual_controller_logs(self):
        with tempfile.TemporaryDirectory(prefix="colite_logging_") as tmp:
            common = ("image_undistorter", "--input_path", Path(tmp)/"missing", "--image_path", tmp,
                      "--output_path", Path(tmp)/"dense", "--dry_run", "--log_color", "0")
            for target in ("stderr", "stdout", "file", "stderr_and_file"):
                logs = Path(tmp)/target
                result = self.run_cli(*common, "--log_target", target, "--log_path", logs)
                # The model is intentionally missing; the stage INFO precedes I/O.
                text = result.stdout + result.stderr
                self.assertIn("Stage image_undistorter", text if target != "file" else "".join(p.read_text(errors="replace") for p in logs.glob("*.INFO.*") if p.is_file()))
                if target == "stdout":
                    if "requires glog >= 0.6" in result.stderr:
                        self.assertIn("Stage image_undistorter", result.stderr)
                    else:
                        self.assertIn("Stage image_undistorter", result.stdout)
                if target in ("stderr", "stderr_and_file"): self.assertIn("Stage image_undistorter", result.stderr)
                if target == "file": self.assertNotIn("Stage image_undistorter", text)
            quiet = self.run_cli(*common, "--log_target", "stderr", "--log_severity", "2")
            self.assertNotIn("Stage image_undistorter", quiet.stdout + quiet.stderr)

    def test_all_entry_points_handle_logging_initialization_errors(self):
        names = ("colite", "feature_extractor", "exhaustive_matcher", "sequential_matcher",
                 "spatial_matcher", "vocab_tree_matcher", "transitive_matcher", "vocab_tree_builder",
                 "mapper", "compare_colmap", "recon_sfm", "recon_mvs")
        for name in names:
            binary = Path(BINARY).parent / (name + (".exe" if os.name == "nt" else ""))
            self.assertTrue(binary.is_file(), str(binary))
            valid = subprocess.run([str(binary), "--log_target", "stderr", "--log_level", "2", "--help"],
                                   text=True, capture_output=True)
            self.assertEqual(valid.returncode, 0, name + valid.stderr)
            invalid = subprocess.run([str(binary), "--log_level", "bad"], text=True, capture_output=True)
            self.assertEqual(invalid.returncode, 1, name + invalid.stderr)
            self.assertNotIn("terminate called", invalid.stderr)

    def test_invalid_values_exit_cleanly_and_old_names_are_rejected(self):
        for args in (("--log_level", "INFO"), ("--log_level", "1junk"), ("--log_level", "2147483648"),
                     ("--log_severity", "-1"), ("--log_severity", "4"), ("--log_target", "typo"),
                     ("--log_color", "maybe"), ("--log_level",)):
            result = self.run_cli("mapper", *args)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertNotIn("terminate called", result.stderr)
        for old in ("--v", "--log_dir", "--log_to_stderr"):
            result = self.run_cli("mapper", old, "1")
            self.assertEqual(result.returncode, 1)
            self.assertIn("Unknown argument", result.stderr)

if __name__ == "__main__": unittest.main()
