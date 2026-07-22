import unittest
from pathlib import Path

from fish_activity.mqtt_io import PondInit
from fish_activity.mqtt_multistream_runtime import (
    parse_args,
    settings_from_args,
    worker_csv_path,
)


class MqttMultistreamRuntimeTest(unittest.TestCase):
    def test_parse_args_keeps_pipeline_args_after_separator(self) -> None:
        args, pipeline_args = parse_args(
            [
                "--mqtt-host",
                "broker.local",
                "--mqtt-init-topic",
                "/AI/+/init",
                "--max-streams",
                "4",
                "--",
                "--config",
                "configs/tune_less_bubble.json",
            ]
        )

        settings = settings_from_args(args)
        self.assertEqual(args.max_streams, 4)
        self.assertEqual(settings.pond_init_topic, "/AI/+/init")
        self.assertEqual(pipeline_args, ["--config", "configs/tune_less_bubble.json"])

    def test_worker_csv_path_uses_slot_sequence_and_pond_label(self) -> None:
        path = worker_csv_path(
            Path("results/mqtt_multistream"),
            3,
            7,
            PondInit(
                pond_id="A4",
                source="rtsp://10.0.0.8:554/live",
                ip="10.0.0.8",
            ),
        )

        self.assertEqual(
            path,
            Path("results/mqtt_multistream/slot03_stream000007_A4_10_0_0_8.csv"),
        )


if __name__ == "__main__":
    unittest.main()
