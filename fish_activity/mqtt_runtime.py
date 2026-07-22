"""MQTT-driven deployment runtime for camera-stream processing."""

from __future__ import annotations

import argparse
import logging
import sys

from fish_activity.mqtt_io import (
    MqttPondInitReceiver,
    MqttPondProtocolClient,
    MqttSettings,
)
from fish_activity.pipeline_v1 import parse_args as parse_pipeline_args
from fish_activity.pipeline_v1 import run_pipeline, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    if argv is None:
        argv = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description=(
            "Wait for AF pond init messages over MQTT, process each camera "
            "stream, then publish score/SCORED/STOP protocol responses. "
            "Pipeline options go after --."
        )
    )
    parser.add_argument("--mqtt-host", required=True, help="MQTT broker host.")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port.")
    parser.add_argument("--mqtt-username", default="", help="MQTT username.")
    parser.add_argument("--mqtt-password", default="", help="MQTT password.")
    parser.add_argument(
        "--mqtt-client-id",
        default="fish-activity-runtime",
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
        "--max-runs",
        type=int,
        default=0,
        help="Number of pond init messages to process. Use 0 to run forever.",
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


def main(argv: list[str] | None = None) -> int:
    mqtt_args, pipeline_args = parse_args(argv)
    setup_logging(mqtt_args.log_level)
    settings = settings_from_args(mqtt_args)

    init_receiver = MqttPondInitReceiver(settings, mqtt_args.camera_url_template)
    init_receiver.connect()
    run_count = 0
    try:
        while mqtt_args.max_runs <= 0 or run_count < mqtt_args.max_runs:
            pond_init = init_receiver.wait_for_pond_init()
            LOGGER.info(
                "starting pipeline pond_id=%s source=%s BM=%s ABW=%s FC=%s",
                pond_init.pond_id,
                pond_init.source,
                pond_init.biomass,
                pond_init.average_body_weight,
                pond_init.feed_cap,
            )
            protocol = MqttPondProtocolClient(settings, pond_init.pond_id)
            try:
                protocol.connect()
                pipeline = parse_pipeline_args([pond_init.source, *pipeline_args])
                if not any(
                    arg == "--headless" or arg.startswith("--headless=")
                    for arg in pipeline_args
                ):
                    pipeline.headless = "on"
                exit_code = run_pipeline(
                    pipeline,
                    command_sink=protocol.publish_command,
                    final_score_sink=protocol.publish_final_score,
                    system_finish_checker=protocol.last_pause_received,
                )
                if exit_code != 0:
                    return exit_code
            finally:
                protocol.close()
            run_count += 1
        return 0
    finally:
        init_receiver.close()


if __name__ == "__main__":
    raise SystemExit(main())
