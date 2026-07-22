import json
import unittest

from fish_activity.mqtt_io import (
    MqttCommandPublisher,
    MqttPondProtocolClient,
    MqttSettings,
    camera_source_from_message,
    mqtt_reason_code_failed,
    parse_af_status_payload,
    parse_camera_message,
    parse_camera_payload,
    parse_pond_init_message,
    parse_system_finish_payload,
)


class FakePublishResult:
    rc = 0

    def wait_for_publish(self) -> None:
        return


class FakeMqttClient:
    def __init__(self) -> None:
        self.published: list[tuple[str, str, int]] = []
        self.on_publish = None

    def publish(self, topic: str, message: str, qos: int) -> FakePublishResult:
        self.published.append((topic, message, qos))
        if self.on_publish is not None:
            self.on_publish(topic, message, qos)
        return FakePublishResult()


class FakeReasonCode:
    def __init__(self, failed: bool) -> None:
        self.failed = failed

    def is_failure(self) -> bool:
        return self.failed


class MqttIoTest(unittest.TestCase):
    def test_plain_ip_payload_uses_template(self) -> None:
        source = camera_source_from_message("192.168.1.20", "rtsp://{ip}:554/main")

        self.assertEqual(source, "rtsp://192.168.1.20:554/main")

    def test_json_ip_payload_uses_template(self) -> None:
        source = camera_source_from_message(
            b'{"camera_ip": "10.0.0.5"}',
            "rtsp://{ip}:8554/live",
        )

        self.assertEqual(source, "rtsp://10.0.0.5:8554/live")

    def test_json_url_payload_uses_url_directly(self) -> None:
        source = camera_source_from_message(
            '{"rtsp_url": "rtsp://camera.local/stream"}',
            "rtsp://{ip}:554/main",
        )

        self.assertEqual(source, "rtsp://camera.local/stream")

    def test_json_payload_can_include_camera_id(self) -> None:
        source = parse_camera_message(
            '{"camera_ip": "10.0.0.8", "camera_id": "pond_08"}',
            "rtsp://{ip}:554/live",
        )

        self.assertEqual(source.source, "rtsp://10.0.0.8:554/live")
        self.assertEqual(source.camera_id, "pond_08")
        self.assertEqual(source.camera_value, "10.0.0.8")

    def test_invalid_json_object_fails(self) -> None:
        with self.assertRaises(ValueError):
            parse_camera_payload('{"unknown": "value"}')

    def test_final_score_publishes_to_configured_topic(self) -> None:
        settings = MqttSettings(
            host="broker.local",
            final_score_topic="fish/activity/final_score",
            qos=1,
        )
        publisher = MqttCommandPublisher(settings)
        fake_client = FakeMqttClient()
        publisher._client = fake_client

        publisher.publish_final_score({"final_feeding_score": 12.5})

        self.assertEqual(len(fake_client.published), 1)
        topic, message, qos = fake_client.published[0]
        self.assertEqual(topic, "fish/activity/final_score")
        self.assertEqual(qos, 1)
        self.assertEqual(json.loads(message), {"final_feeding_score": 12.5})

    def test_mqtt_reason_code_supports_paho_v2_objects(self) -> None:
        self.assertFalse(mqtt_reason_code_failed(FakeReasonCode(False)))
        self.assertTrue(mqtt_reason_code_failed(FakeReasonCode(True)))
        self.assertFalse(mqtt_reason_code_failed(0))
        self.assertTrue(mqtt_reason_code_failed(5))

    def test_system_finish_payload_accepts_simple_formats(self) -> None:
        self.assertTrue(parse_system_finish_payload("system finish"))
        self.assertTrue(parse_system_finish_payload("system_finish"))
        self.assertTrue(parse_system_finish_payload('{"system_finish": true}'))
        self.assertTrue(parse_system_finish_payload('{"event": "system_finish"}'))
        self.assertFalse(parse_system_finish_payload('{"system_finish": false}'))
        self.assertFalse(parse_system_finish_payload("running"))

    def test_pond_init_payload_uses_pond_topic_and_ip(self) -> None:
        pond_init = parse_pond_init_message(
            "/AI/A4/init",
            '{"IP": "192.168.46.24:8080", "BM": 12500, "ABW": 750, "FC": 150}',
            "http://{ip}/video",
        )

        self.assertEqual(pond_init.pond_id, "A4")
        self.assertEqual(pond_init.ip, "192.168.46.24:8080")
        self.assertEqual(pond_init.source, "http://192.168.46.24:8080/video")
        self.assertEqual(pond_init.biomass, 12500.0)
        self.assertEqual(pond_init.average_body_weight, 750.0)
        self.assertEqual(pond_init.feed_cap, 150.0)

    def test_af_status_payload_normalizes_protocol_states(self) -> None:
        self.assertEqual(parse_af_status_payload('{"state": "FEED_STARTED"}'), "FEED_STARTED")
        self.assertEqual(parse_af_status_payload('{"state": "Paused"}'), "PAUSED")
        self.assertEqual(parse_af_status_payload('{"state": "Last_Pause"}'), "LAST_PAUSE")
        self.assertEqual(parse_af_status_payload("SCORED"), "SCORED")

    def test_pond_protocol_maps_commands_score_and_stop_sequence(self) -> None:
        settings = MqttSettings(
            host="broker.local",
            pond_control_topic_template="/AI/{pond_id}/control",
            pond_score_topic_template="/AI/{pond_id}/score",
            connect_timeout=0.1,
            qos=1,
        )
        protocol = MqttPondProtocolClient(settings, "A4")
        fake_client = FakeMqttClient()

        def acknowledge_score(topic: str, _message: str, _qos: int) -> None:
            if topic == "/AI/A4/score":
                protocol._scored_event.set()

        fake_client.on_publish = acknowledge_score
        protocol._client = fake_client

        protocol.publish_command({"command": "start"})
        protocol.publish_command({"command": "pause"})
        protocol.publish_command({"command": "finish"})
        protocol.publish_final_score({"final_feeding_score": 8.2})

        published = [(topic, json.loads(message), qos) for topic, message, qos in fake_client.published]
        self.assertEqual(
            published,
            [
                ("/AI/A4/control", {"command": "START"}, 1),
                ("/AI/A4/control", {"command": "PAUSE"}, 1),
                ("/AI/A4/score", {"score": 8.2}, 1),
                ("/AI/A4/control", {"command": "STOP"}, 1),
            ],
        )


if __name__ == "__main__":
    unittest.main()
