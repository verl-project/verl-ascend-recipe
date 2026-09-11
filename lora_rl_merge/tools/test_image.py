import hashlib
import unittest
from unittest.mock import patch

from check_image import verify


class ImageTest(unittest.TestCase):
    def setUp(self):
        self.layers = '["sha256:example-layer"]\n'
        self.runtime = '["PATH=/usr/bin"]["bash"]null/root\n'
        self.metadata = "2026-01-01T00:00:00Z|arm64|linux|1\n"
        self.reference = {
            "rootfs_layers_sha256": hashlib.sha256(self.layers.encode()).hexdigest(),
            "runtime_config_fields_sha256": hashlib.sha256(self.runtime.encode()).hexdigest(),
            "created": "2026-01-01T00:00:00Z",
            "architecture": "arm64",
            "os": "linux",
            "layer_count": 1,
        }

    @patch("check_image.subprocess.check_output")
    def test_different_image_id_with_matching_content_is_accepted(self, inspect):
        inspect.side_effect = [self.layers, self.runtime, self.metadata]
        verify("sha256:loaded-image", self.reference)

    @patch("check_image.subprocess.check_output")
    def test_changed_layer_or_runtime_is_rejected(self, inspect):
        for values, error in (
            (["[]\n", self.runtime, self.metadata], "RootFS"),
            ([self.layers, "[]\n", self.metadata], "Runtime"),
        ):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                inspect.side_effect = values
                verify("sha256:loaded-image", self.reference)

    @patch("check_image.subprocess.check_output")
    def test_wrong_architecture_is_rejected(self, inspect):
        inspect.side_effect = [self.layers, self.runtime, self.metadata.replace("arm64", "amd64")]
        with self.assertRaisesRegex(ValueError, "architecture"):
            verify("sha256:loaded-image", self.reference)


if __name__ == "__main__":
    unittest.main()
