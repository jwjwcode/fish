import csv
import json
import re
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from fish_activity.pipeline_v1 import parse_args, run_pipeline


class PipelineRuntimeTest(unittest.TestCase):
    def _write_video(self, path: Path, frame_count: int = 6) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            5.0,
            (64, 32),
        )
        self.assertTrue(writer.isOpened())
        try:
            for index in range(frame_count):
                frame = np.zeros((32, 64, 3), dtype=np.uint8)
                cv2.circle(frame, (16 + index * 2, 16), 4, (255, 255, 255), -1)
                writer.write(frame)
        finally:
            writer.release()

    def test_headless_pipeline_writes_csv_metadata_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = temp_path / "input.mp4"
            csv_path = temp_path / "out.csv"
            self._write_video(video_path)

            commands: list[dict[str, object]] = []
            final_scores: list[dict[str, float]] = []
            args = parse_args(
                [
                    str(video_path),
                    "--headless",
                    "on",
                    "--csv",
                    str(csv_path),
                    "--flow-method",
                    "none",
                    "--resize-width",
                    "64",
                    "--progress-interval",
                    "0",
                    "--decision-background-frames",
                    "2",
                ]
            )

            run_pipeline(
                args,
                command_sink=commands.append,
                final_score_sink=final_scores.append,
            )

            metadata_path = csv_path.with_suffix(".metadata.json")
            excel_path = csv_path.with_suffix(".xlsx")
            self.assertTrue(csv_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertTrue(excel_path.exists())
            with csv_path.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertGreaterEqual(len(rows), 2)
            csv_score_avg = sum(float(row["total_activity"]) for row in rows) / float(
                len(rows)
            )
            self.assertIn("run_id", rows[0])
            self.assertEqual(rows[1]["feeding_command"], "start")
            self.assertEqual(commands[0]["command"], "start")
            self.assertEqual(set(commands[0]), {"command"})
            self.assertEqual(len(final_scores), 1)
            self.assertEqual(set(final_scores[0]), {"final_feeding_score"})
            self.assertAlmostEqual(
                final_scores[0]["final_feeding_score"],
                csv_score_avg,
                places=5,
            )
            with metadata_path.open() as metadata_file:
                metadata = json.load(metadata_file)
            self.assertEqual(metadata["csv_path"], str(csv_path))
            self.assertEqual(metadata["excel_path"], str(excel_path))
            with ZipFile(excel_path) as workbook:
                score_sheet = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")
                summary_sheet = workbook.read("xl/worksheets/sheet2.xml").decode("utf-8")
            self.assertIn("current_segmentation_score", score_sheet)
            self.assertIn("current_optical_flow_score", score_sheet)
            self.assertIn("current_total_activity", score_sheet)
            self.assertIn("previous_10_total_activity_average", score_sheet)
            self.assertIn("final_feeding_score", summary_sheet)

    def test_default_run_writes_only_video_and_excel_with_video_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = temp_path / "input.mp4"
            output_path = temp_path / "out.mp4"
            excel_path = temp_path / "out.xlsx"
            self._write_video(video_path, frame_count=10)

            args = parse_args(
                [
                    str(video_path),
                    "-o",
                    str(output_path),
                    "--excel",
                    str(excel_path),
                    "--flow-method",
                    "none",
                    "--resize-width",
                    "64",
                    "--frame-step",
                    "2",
                    "--progress-interval",
                    "0",
                    "--decision-background-frames",
                    "1",
                ]
            )

            run_pipeline(args)

            self.assertTrue(output_path.exists())
            self.assertTrue(excel_path.exists())
            self.assertFalse((temp_path / "out.csv").exists())
            self.assertFalse((temp_path / "out.metadata.json").exists())

            output = cv2.VideoCapture(str(output_path))
            try:
                output_fps = float(output.get(cv2.CAP_PROP_FPS))
                output_frames = int(output.get(cv2.CAP_PROP_FRAME_COUNT))
            finally:
                output.release()
            expected_last_time_s = (output_frames - 1) / output_fps

            with ZipFile(excel_path) as workbook:
                score_sheet = workbook.read("xl/worksheets/sheet1.xml").decode("utf-8")
            numeric_values = [
                float(value) for value in re.findall(r"<v>([-0-9.eE]+)</v>", score_sheet)
            ]
            time_values = numeric_values[0::5]
            self.assertEqual(len(time_values), output_frames)
            self.assertAlmostEqual(time_values[0], 0.0, places=6)
            self.assertAlmostEqual(time_values[-1], expected_last_time_s, places=5)

    def test_stop_after_finish_ends_pipeline_early(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = temp_path / "input.mp4"
            csv_path = temp_path / "out.csv"
            self._write_video(video_path, frame_count=10)

            final_scores: list[dict[str, float]] = []
            args = parse_args(
                [
                    str(video_path),
                    "--headless",
                    "on",
                    "--csv",
                    str(csv_path),
                    "--flow-method",
                    "none",
                    "--resize-width",
                    "64",
                    "--progress-interval",
                    "0",
                    "--decision-background-frames",
                    "1",
                    "--decision-machine-finish-second",
                    "0.2",
                    "--stop-after-finish",
                    "on",
                ]
            )

            run_pipeline(args, final_score_sink=final_scores.append)

            with csv_path.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            csv_score_avg = sum(float(row["total_activity"]) for row in rows) / float(
                len(rows)
            )
            self.assertLess(len(rows), 10)
            self.assertEqual(rows[-1]["feeding_command"], "finish")
            self.assertEqual(len(final_scores), 1)
            self.assertAlmostEqual(
                final_scores[0]["final_feeding_score"],
                csv_score_avg,
                places=5,
            )

    def test_system_finish_stops_without_publishing_finish_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_path = temp_path / "input.mp4"
            csv_path = temp_path / "out.csv"
            self._write_video(video_path, frame_count=10)

            commands: list[dict[str, object]] = []
            final_scores: list[dict[str, float]] = []
            checks = 0

            def system_finish_received() -> bool:
                nonlocal checks
                checks += 1
                return checks >= 3

            args = parse_args(
                [
                    str(video_path),
                    "--headless",
                    "on",
                    "--csv",
                    str(csv_path),
                    "--flow-method",
                    "none",
                    "--resize-width",
                    "64",
                    "--progress-interval",
                    "0",
                    "--decision-background-frames",
                    "1",
                ]
            )

            run_pipeline(
                args,
                command_sink=commands.append,
                final_score_sink=final_scores.append,
                system_finish_checker=system_finish_received,
            )

            with csv_path.open(newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
            self.assertLess(len(rows), 10)
            self.assertEqual(rows[-1]["feeding_command"], "finish")
            self.assertEqual(commands, [{"command": "start"}])
            self.assertEqual(len(final_scores), 1)


if __name__ == "__main__":
    unittest.main()
