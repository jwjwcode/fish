"""MQTT-driven runtime for processing multiple camera streams concurrently."""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import re
import sys
import time
from dataclasses import replace
from pathlib import Path

from fish_activity.mqtt_io import (
    MqttPondInitReceiver,
    MqttPondProtocolClient,
    MqttSettings,
    PondInit,
)
from fish_activity.pipeline_v1 import parse_args as parse_pipeline_args
from fish_activity.pipeline_v1 import run_pipeline, setup_logging
LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description=(
            "Keep up to N MQTT-provided camera streams running concurrently. "
            "Pipeline options go after --."
        )
    )
    parser.add_argument("--mqtt-host", required=True, help="MQTT broker host.")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port.")
    parser.add_argument("--mqtt-username", default="", help="MQTT username.")
    parser.add_argument("--mqtt-password", default="", help="MQTT password.")
    parser.add_argument(
        "--mqtt-client-id",
        default="fish-activity-multistream",
        help="MQTT client id prefix.",
    )
    parser.add_argument(
        "--mqtt-init-topic",
        default="/AI/+/init",
        help="AF -> AI topic filter for pond init messages.",
    )
    parser.add_argument(
        "--mqtt-control-topic-template",
        default="/AI/{pond_id}/control",
        help="AI -> AF command topic template.",
    )
    parser.add_argument(
        "--mqtt-status-topic-template",
        default="/AI/{pond_id}/status",
        help="AF -> AI status topic template.",
    )
    parser.add_argument(
        "--mqtt-score-topic-template",
        default="/AI/{pond_id}/score",
        help="AI -> AF final score topic template.",
    )
    parser.add_argument("--mqtt-qos", type=int, default=1, choices=(0, 1, 2))
    parser.add_argument("--mqtt-connect-timeout", type=float, default=120.0)
    parser.add_argument("--mqtt-command-retries", type=int, default=3)
    parser.add_argument("--mqtt-command-retry-seconds", type=float, default=1.0)
    parser.add_argument(
        "--camera-url-template",
        default="{ip}",
        help="Template used when AF sends IP instead of a full URL. Use {ip} or {host}.",
    )
    parser.add_argument(
        "--max-streams",
        type=int,
        default=4,
        help="Maximum active camera streams.",
    )
    parser.add_argument(
        "--max-total-streams",
        type=int,
        default=0,
        help="Stop after starting this many streams. Use 0 to run forever.",
    )
    parser.add_argument(
        "--worker-output-dir",
        type=Path,
        default=Path("results/mqtt_multistream"),
        help="Directory for per-stream CSV and metadata outputs.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=1.0,
        help="Seconds between worker health checks.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Runtime log verbosity before pipeline options are parsed.",
    )
    args, pipeline_args = parser.parse_known_args(argv)
    if pipeline_args and pipeline_args[0] == "--":
        pipeline_args = pipeline_args[1:]
    return args, pipeline_args


def settings_from_args(args: argparse.Namespace) -> MqttSettings:
    return MqttSettings(
        host=args.mqtt_host,
        port=args.mqtt_port,
        username=args.mqtt_username,
        password=args.mqtt_password,
        client_id=args.mqtt_client_id,
        qos=args.mqtt_qos,
        connect_timeout=args.mqtt_connect_timeout,
        pond_init_topic=args.mqtt_init_topic,
        pond_control_topic_template=args.mqtt_control_topic_template,
        pond_status_topic_template=args.mqtt_status_topic_template,
        pond_score_topic_template=args.mqtt_score_topic_template,
        command_retries=args.mqtt_command_retries,
        command_retry_seconds=args.mqtt_command_retry_seconds,
    )


def worker_csv_path(
    output_dir: Path,
    slot_id: int,
    sequence_id: int,
    pond_init: PondInit,
) -> Path:
    label = f"{pond_init.pond_id}_{pond_init.ip or pond_init.source}"
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", label).strip("_")[:80]
    stem = stem or f"slot_{slot_id}"
    return output_dir / f"slot{slot_id:02d}_stream{sequence_id:06d}_{stem}.csv"


def stream_worker_main(
    settings: MqttSettings,
    pond_init: PondInit,
    pipeline_args: list[str],
    output_dir: str,
    slot_id: int,
    sequence_id: int,
    log_level: str,
) -> int:
    setup_logging(log_level)
    worker_settings = replace(
        settings,
        client_id=f"{settings.client_id}-slot{slot_id}-stream{sequence_id}",
    )
    protocol = MqttPondProtocolClient(worker_settings, pond_init.pond_id)
    csv_path = worker_csv_path(Path(output_dir), slot_id, sequence_id, pond_init)
    worker_pipeline_args = [
        pond_init.source,
        *pipeline_args,
        "--headless",
        "on",
        "--csv",
        str(csv_path),
    ]
    try:
        protocol.connect()
        pipeline = parse_pipeline_args(worker_pipeline_args)
        LOGGER.info(
            "worker slot=%s stream=%s pond_id=%s source=%s csv=%s",
            slot_id,
            sequence_id,
            pond_init.pond_id,
            pond_init.source,
            csv_path,
        )
        return run_pipeline(
            pipeline,
            command_sink=protocol.publish_command,
            final_score_sink=protocol.publish_final_score,
            system_finish_checker=protocol.last_pause_received,
        )
    finally:
        protocol.close()


def start_worker(
    context: mp.context.BaseContext,
    settings: MqttSettings,
    pond_init: PondInit,
    pipeline_args: list[str],
    output_dir: Path,
    slot_id: int,
    sequence_id: int,
    log_level: str,
) -> mp.Process:
    process = context.Process(
        target=stream_worker_main,
        args=(
            settings,
            pond_init,
            list(pipeline_args),
            str(output_dir),
            slot_id,
            sequence_id,
            log_level,
        ),
        name=f"fish-stream-slot{slot_id}-stream{sequence_id}",
    )
    process.start()
    return process


def can_start_more(started_count: int, max_total_streams: int) -> bool:
    return max_total_streams <= 0 or started_count < max_total_streams


def next_available_slot(active: dict[int, tuple[int, int, PondInit, mp.Process]]) -> int:
    slot_id = 1
    while slot_id in active:
        slot_id += 1
    return slot_id


def main(argv: list[str] | None = None) -> int:
    mqtt_args, pipeline_args = parse_args(argv)
    setup_logging(mqtt_args.log_level)
    if mqtt_args.max_streams < 1:
        raise SystemExit("--max-streams must be >= 1")
    if mqtt_args.max_total_streams < 0:
        raise SystemExit("--max-total-streams must be >= 0")
    if mqtt_args.poll_seconds <= 0:
        raise SystemExit("--poll-seconds must be > 0")

    settings = settings_from_args(mqtt_args)
    mqtt_args.worker_output_dir.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    init_receiver = MqttPondInitReceiver(settings, mqtt_args.camera_url_template)
    active: dict[int, tuple[int, int, PondInit, mp.Process]] = {}
    started_count = 0

    try:
        init_receiver.connect()
        while active or can_start_more(started_count, mqtt_args.max_total_streams):
            while len(active) < mqtt_args.max_streams and can_start_more(
                started_count,
                mqtt_args.max_total_streams,
            ):
                slot_id = next_available_slot(active)
                sequence_id = started_count + 1
                timeout = (
                    settings.connect_timeout if not active else mqtt_args.poll_seconds
                )
                try:
                    pond_init = init_receiver.wait_for_pond_init(timeout=timeout)
                except TimeoutError:
                    break
                process = start_worker(
                    context,
                    settings,
                    pond_init,
                    pipeline_args,
                    mqtt_args.worker_output_dir,
                    slot_id,
                    sequence_id,
                    mqtt_args.log_level,
                )
                active[slot_id] = (sequence_id, process.pid or -1, pond_init, process)
                started_count += 1
                LOGGER.info(
                    "started worker slot=%s pid=%s pond_id=%s active=%s/%s",
                    slot_id,
                    process.pid,
                    pond_init.pond_id,
                    len(active),
                    mqtt_args.max_streams,
                )

            for slot_id, (sequence_id, pid, pond_init, process) in list(active.items()):
                if process.is_alive():
                    continue
                process.join()
                LOGGER.info(
                    "worker finished slot=%s pid=%s exit_code=%s pond_id=%s",
                    slot_id,
                    pid,
                    process.exitcode,
                    pond_init.pond_id,
                )
                del active[slot_id]
            if active:
                time.sleep(mqtt_args.poll_seconds)
        return 0
    except KeyboardInterrupt:
        LOGGER.warning("stopping multistream runtime after keyboard interrupt")
        return 130
    finally:
        for _slot_id, (_sequence_id, _pid, _pond_init, process) in active.items():
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        init_receiver.close()


if __name__ == "__main__":
    raise SystemExit(main())
