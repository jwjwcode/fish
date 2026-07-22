import argparse
import json
import tempfile
import unittest
from pathlib import Path

from fish_activity.config import load_config_values, validate_args
from fish_activity.pipeline_v1 import default_output_paths, parse_args


class ConfigValidationTest(unittest.TestCase):
    def _write_config(self, payload: dict[str, object]) -> Path:
        temp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        with temp:
            json.dump(payload, temp)
        return Path(temp.name)

    def test_nested_config_loads_known_keys_and_ignores_metadata(self) -> None:
        path = self._write_config(
            {
                "schema_version": 1,
                "detector": {
                    "type": "unsupervised",
                    "preset": "current",
                    "seg_method": "anomaly",
                    "flow_method": "none",
                },
                "scoring": {"seg_weight": 2.0},
            }
        )
        parser = argparse.ArgumentParser()
        parser.add_argument("--preset", choices=("current", "previous"))
        parser.add_argument("--seg-method", choices=("anomaly", "splash"))
        parser.add_argument("--flow-method", choices=("auto", "none"))
        parser.add_argument("--seg-weight", type=float)

        values = load_config_values(parser, path)

        self.assertEqual(values["preset"], "current")
        self.assertEqual(values["seg_method"], "anomaly")
        self.assertEqual(values["flow_method"], "none")
        self.assertEqual(values["seg_weight"], 2.0)
        self.assertNotIn("type", values)

    def test_unknown_config_key_fails_fast(self) -> None:
        path = self._write_config({"flow_methd": "none"})
        parser = argparse.ArgumentParser()
        parser.add_argument("--flow-method", choices=("auto", "none"))

        with self.assertRaises(SystemExit):
            load_config_values(parser, path)

    def test_cli_override_wins_over_config(self) -> None:
        path = self._write_config({"flow_method": "none", "resize_width": 640})

        args = parse_args(
            [
                "example.mkv",
                "--config",
                str(path),
                "--flow-method",
                "farneback",
            ]
        )

        self.assertEqual(args.flow_method, "farneback")
        self.assertEqual(args.resize_width, 640)

    def test_stream_input_does_not_need_local_path(self) -> None:
        args = parse_args(
            [
                "rtsp://192.168.1.10:554/main",
                "--headless",
                "on",
                "--duration",
                "5",
            ]
        )
        output, csv_output, excel_output = default_output_paths(args)

        self.assertIsNone(output)
        self.assertIsNone(csv_output)
        self.assertEqual(excel_output.name, "192.168.1.10_554_main_activity_v1.xlsx")

    def test_range_validation_rejects_invalid_values(self) -> None:
        args = argparse.Namespace(
            frame_step=0,
            flow_method="auto",
            flow_mask="segmentation",
            flow_weight=0.03,
        )

        with self.assertRaises(SystemExit):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
