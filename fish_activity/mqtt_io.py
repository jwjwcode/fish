"""MQTT helpers for deployment runtime integration."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event
from typing import Any

LOGGER = logging.getLogger(__name__)


def mqtt_reason_code_failed(reason_code: Any) -> bool:
    is_failure = getattr(reason_code, "is_failure", None)
    if callable(is_failure):
        return bool(is_failure())
    if isinstance(is_failure, bool):
        return is_failure
    value = getattr(reason_code, "value", reason_code)
    try:
        return int(value) != 0
    except (TypeError, ValueError):
        return str(reason_code).lower() not in {"success", "0", "no error"}


def mqtt_client_module() -> Any:
    try:
        import paho.mqtt.client as mqtt
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "Missing MQTT dependency: paho-mqtt\n"
            "Install it with: python3 -m pip install paho-mqtt"
        ) from exc
    return mqtt


@dataclass
class MqttSettings:
    host: str
    port: int = 1883
    username: str = ""
    password: str = ""
    client_id: str = "fish-activity-runtime"
    keepalive: int = 60
    qos: int = 1
    connect_timeout: float = 30.0
    camera_topic: str = "fish/camera/ip"
    camera_request_topic: str = "fish/camera/request"
    command_topic: str = "fish/feeder/command"
    final_score_topic: str = "fish/activity/final_score"
    system_finish_topic: str = "fish/feeder/system_finish"
    pond_init_topic: str = "/AI/+/init"
    pond_control_topic_template: str = "/AI/{pond_id}/control"
    pond_status_topic_template: str = "/AI/{pond_id}/status"
    pond_score_topic_template: str = "/AI/{pond_id}/score"
    status_topic: str = ""
    command_retries: int = 3
    command_retry_seconds: float = 1.0


@dataclass(frozen=True)
class CameraSource:
    source: str
    camera_id: str = ""
    camera_value: str = ""


@dataclass(frozen=True)
class PondInit:
    pond_id: str
    source: str
    ip: str
    biomass: float | None = None
    average_body_weight: float | None = None
    feed_cap: float | None = None


def parse_camera_payload(payload: bytes | str) -> str:
    return parse_camera_message(payload, "{ip}").camera_value


def parse_camera_message(payload: bytes | str, url_template: str) -> CameraSource:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    text = text.strip()
    if not text:
        raise ValueError("empty camera payload")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        camera_value = text
        camera_id = ""
    else:
        if not isinstance(data, dict):
            raise ValueError("camera payload JSON must be an object")
        camera_value = ""
        for key in ("rtsp_url", "url", "camera_url", "camera_ip", "IP", "ip", "host"):
            value = data.get(key)
            if value:
                camera_value = str(value).strip()
                break
        if not camera_value:
            raise ValueError(
                "camera payload must include rtsp_url, url, camera_ip, ip, or host"
            )
        camera_id = str(data.get("camera_id") or data.get("id") or "").strip()

    if camera_value.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        source = camera_value
    else:
        source = url_template.format(ip=camera_value, host=camera_value)
    return CameraSource(source=source, camera_id=camera_id, camera_value=camera_value)


def camera_source_from_message(payload: bytes | str, url_template: str) -> str:
    return parse_camera_message(payload, url_template).source


POND_INIT_TOPIC_RE = re.compile(r"^/AI/([ABC][1-8])/init$")


def parse_pond_id_from_init_topic(topic: str) -> str:
    match = POND_INIT_TOPIC_RE.match(topic)
    if match is None:
        raise ValueError(f"invalid pond init topic: {topic}")
    return match.group(1)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric metadata value: {value}") from exc


def parse_pond_init_message(
    topic: str,
    payload: bytes | str,
    url_template: str,
) -> PondInit:
    pond_id = parse_pond_id_from_init_topic(topic)
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("pond init payload must be JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("pond init payload JSON must be an object")

    ip = str(data.get("IP") or data.get("ip") or data.get("camera_ip") or "").strip()
    if not ip:
        raise ValueError("pond init payload must include IP")
    if ip.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        source = ip
    else:
        source = url_template.format(ip=ip, host=ip)

    return PondInit(
        pond_id=pond_id,
        source=source,
        ip=ip,
        biomass=_optional_float(data.get("BM")),
        average_body_weight=_optional_float(data.get("ABW")),
        feed_cap=_optional_float(data.get("FC")),
    )


def normalize_af_state(value: object) -> str:
    return str(value).strip().upper().replace(" ", "_").replace("-", "_")


def parse_af_status_payload(payload: bytes | str) -> str:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    text = text.strip()
    if not text:
        raise ValueError("empty AF status payload")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return normalize_af_state(text)
    if not isinstance(data, dict):
        raise ValueError("AF status payload JSON must be an object")
    state = data.get("state")
    if not state:
        raise ValueError("AF status payload must include state")
    return normalize_af_state(state)


def parse_system_finish_payload(payload: bytes | str) -> bool:
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    text = text.strip()
    if not text:
        return False

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text.lower().replace("-", "_").replace(" ", "_") in {
            "system_finish",
            "finish",
            "finished",
        }

    if isinstance(data, bool):
        return data
    if not isinstance(data, dict):
        return False
    for key in ("system_finish", "finish", "finished"):
        value = data.get(key)
        if isinstance(value, bool):
            return value
    for key in ("event", "status", "command", "message"):
        value = data.get(key)
        if value is None:
            continue
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {"system_finish", "finish", "finished"}:
            return True
    return False


class MqttCameraIpReceiver:
    def __init__(self, settings: MqttSettings, url_template: str) -> None:
        self.settings = settings
        self.url_template = url_template
        self._event = Event()
        self._source = ""
        self._error = ""

    def wait_for_camera_source(self) -> str:
        mqtt = mqtt_client_module()
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=self.settings.client_id,
        )
        if self.settings.username:
            client.username_pw_set(self.settings.username, self.settings.password)

        def on_connect(client: Any, _userdata: Any, _flags: Any, reason_code: Any, _props: Any) -> None:
            if mqtt_reason_code_failed(reason_code):
                self._error = f"MQTT connection failed: {reason_code}"
                self._event.set()
                return
            LOGGER.info("connected to MQTT broker %s:%s", self.settings.host, self.settings.port)
            client.subscribe(self.settings.camera_topic, qos=self.settings.qos)
            if self.settings.status_topic:
                client.publish(
                    self.settings.status_topic,
                    json.dumps({"status": "waiting_for_camera_ip"}),
                    qos=self.settings.qos,
                )

        def on_message(_client: Any, _userdata: Any, message: Any) -> None:
            try:
                self._source = parse_camera_message(
                    message.payload,
                    self.url_template,
                ).source
            except ValueError as exc:
                LOGGER.warning("ignored invalid camera payload: %s", exc)
                return
            LOGGER.info("received camera source from MQTT topic %s", message.topic)
            self._event.set()

        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(self.settings.host, self.settings.port, self.settings.keepalive)
        client.loop_start()
        try:
            if not self._event.wait(self.settings.connect_timeout):
                raise TimeoutError(
                    f"Timed out waiting for camera IP on topic {self.settings.camera_topic}"
                )
            if self._error:
                raise RuntimeError(self._error)
            return self._source
        finally:
            client.loop_stop()
            client.disconnect()


class MqttPondInitReceiver:
    def __init__(self, settings: MqttSettings, url_template: str) -> None:
        self.settings = settings
        self.url_template = url_template
        self._client: Any | None = None
        self._connected = Event()
        self._queue: Queue[PondInit] = Queue()
        self._error = ""

    def connect(self) -> None:
        if self._client is not None:
            return

        mqtt = mqtt_client_module()
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"{self.settings.client_id}-pond-init",
        )
        if self.settings.username:
            client.username_pw_set(self.settings.username, self.settings.password)

        def on_connect(
            client: Any,
            _userdata: Any,
            _flags: Any,
            reason_code: Any,
            _props: Any,
        ) -> None:
            if mqtt_reason_code_failed(reason_code):
                self._error = f"MQTT connection failed: {reason_code}"
                self._connected.set()
                return
            LOGGER.info(
                "connected to MQTT broker %s:%s",
                self.settings.host,
                self.settings.port,
            )
            client.subscribe(self.settings.pond_init_topic, qos=self.settings.qos)
            self._connected.set()

        def on_message(_client: Any, _userdata: Any, message: Any) -> None:
            try:
                pond_init = parse_pond_init_message(
                    message.topic,
                    message.payload,
                    self.url_template,
                )
            except ValueError as exc:
                LOGGER.warning("ignored invalid pond init payload: %s", exc)
                return
            LOGGER.info(
                "received pond init pond_id=%s ip=%s topic=%s",
                pond_init.pond_id,
                pond_init.ip,
                message.topic,
            )
            self._queue.put(pond_init)

        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(self.settings.host, self.settings.port, self.settings.keepalive)
        client.loop_start()
        self._client = client
        if not self._connected.wait(self.settings.connect_timeout):
            self.close()
            raise TimeoutError(
                f"Timed out connecting to MQTT broker {self.settings.host}:{self.settings.port}"
            )
        if self._error:
            self.close()
            raise RuntimeError(self._error)

    def wait_for_pond_init(self, timeout: float | None = None) -> PondInit:
        if self._client is None:
            self.connect()
        try:
            return self._queue.get(timeout=timeout or self.settings.connect_timeout)
        except Empty as exc:
            raise TimeoutError(
                f"Timed out waiting for pond init on topic {self.settings.pond_init_topic}"
            ) from exc

    def close(self) -> None:
        if self._client is None:
            return
        self._client.loop_stop()
        self._client.disconnect()
        self._client = None


class MqttSystemFinishListener:
    def __init__(self, settings: MqttSettings) -> None:
        self.settings = settings
        self._client: Any | None = None
        self._connected = Event()
        self._event = Event()
        self._error = ""

    def connect(self) -> None:
        if self._client is not None:
            return

        mqtt = mqtt_client_module()
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"{self.settings.client_id}-system-finish",
        )
        if self.settings.username:
            client.username_pw_set(self.settings.username, self.settings.password)

        def on_connect(client: Any, _userdata: Any, _flags: Any, reason_code: Any, _props: Any) -> None:
            if mqtt_reason_code_failed(reason_code):
                self._error = f"MQTT connection failed: {reason_code}"
                self._connected.set()
                return
            LOGGER.info("connected to MQTT broker %s:%s", self.settings.host, self.settings.port)
            client.subscribe(self.settings.system_finish_topic, qos=self.settings.qos)
            self._connected.set()

        def on_message(_client: Any, _userdata: Any, message: Any) -> None:
            if parse_system_finish_payload(message.payload):
                LOGGER.info("received system finish from MQTT topic %s", message.topic)
                self._event.set()
            else:
                LOGGER.warning("ignored invalid system finish payload on %s", message.topic)

        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(self.settings.host, self.settings.port, self.settings.keepalive)
        client.loop_start()
        self._client = client
        if not self._connected.wait(self.settings.connect_timeout):
            self.close()
            raise TimeoutError(
                f"Timed out connecting to MQTT broker {self.settings.host}:{self.settings.port}"
            )
        if self._error:
            self.close()
            raise RuntimeError(self._error)

    def received(self) -> bool:
        return self._event.is_set()

    def close(self) -> None:
        if self._client is None:
            return
        self._client.loop_stop()
        self._client.disconnect()
        self._client = None


class MqttCameraRequestClient:
    def __init__(self, settings: MqttSettings, url_template: str) -> None:
        self.settings = settings
        self.url_template = url_template
        self._client: Any | None = None
        self._connected = Event()
        self._queue: Queue[CameraSource] = Queue()
        self._error = ""

    def connect(self) -> None:
        if self._client is not None:
            return

        mqtt = mqtt_client_module()
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"{self.settings.client_id}-camera-requests",
        )
        if self.settings.username:
            client.username_pw_set(self.settings.username, self.settings.password)

        def on_connect(client: Any, _userdata: Any, _flags: Any, reason_code: Any, _props: Any) -> None:
            if mqtt_reason_code_failed(reason_code):
                self._error = f"MQTT connection failed: {reason_code}"
                self._connected.set()
                return
            LOGGER.info("connected to MQTT broker %s:%s", self.settings.host, self.settings.port)
            client.subscribe(self.settings.camera_topic, qos=self.settings.qos)
            self._connected.set()

        def on_message(_client: Any, _userdata: Any, message: Any) -> None:
            try:
                source = parse_camera_message(message.payload, self.url_template)
            except ValueError as exc:
                LOGGER.warning("ignored invalid camera payload: %s", exc)
                return
            LOGGER.info("received camera source from MQTT topic %s", message.topic)
            self._queue.put(source)

        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(self.settings.host, self.settings.port, self.settings.keepalive)
        client.loop_start()
        self._client = client
        if not self._connected.wait(self.settings.connect_timeout):
            self.close()
            raise TimeoutError(
                f"Timed out connecting to MQTT broker {self.settings.host}:{self.settings.port}"
            )
        if self._error:
            self.close()
            raise RuntimeError(self._error)

    def close(self) -> None:
        if self._client is None:
            return
        self._client.loop_stop()
        self._client.disconnect()
        self._client = None

    def request_next_camera(self, slot_id: int, request_id: str) -> CameraSource:
        if self._client is None:
            self.connect()
        assert self._client is not None

        payload = {
            "request": "next_camera",
            "slot_id": slot_id,
            "request_id": request_id,
        }
        result = self._client.publish(
            self.settings.camera_request_topic,
            json.dumps(payload, sort_keys=True),
            qos=self.settings.qos,
        )
        result.wait_for_publish()
        if result.rc != 0:
            raise RuntimeError(f"Could not publish camera request: rc={result.rc}")
        LOGGER.info(
            "requested next camera slot=%s request_id=%s topic=%s",
            slot_id,
            request_id,
            self.settings.camera_request_topic,
        )
        try:
            return self._queue.get(timeout=self.settings.connect_timeout)
        except Empty as exc:
            raise TimeoutError(
                f"Timed out waiting for camera on topic {self.settings.camera_topic}"
            ) from exc


class MqttCommandPublisher:
    def __init__(self, settings: MqttSettings) -> None:
        self.settings = settings
        self._client: Any | None = None

    def connect(self) -> None:
        mqtt = mqtt_client_module()
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"{self.settings.client_id}-commands",
        )
        if self.settings.username:
            client.username_pw_set(self.settings.username, self.settings.password)
        client.connect(self.settings.host, self.settings.port, self.settings.keepalive)
        client.loop_start()
        self._client = client

    def close(self) -> None:
        if self._client is None:
            return
        self._client.loop_stop()
        self._client.disconnect()
        self._client = None

    def _publish(self, topic: str, payload: dict[str, Any], label: str) -> None:
        if self._client is None:
            self.connect()
        assert self._client is not None

        message = json.dumps(payload, sort_keys=True)
        last_error: Exception | None = None
        for attempt in range(1, self.settings.command_retries + 1):
            try:
                result = self._client.publish(
                    topic,
                    message,
                    qos=self.settings.qos,
                )
                result.wait_for_publish()
                if result.rc == 0:
                    LOGGER.info(
                        "published MQTT %s topic=%s",
                        label,
                        topic,
                    )
                    return
                last_error = RuntimeError(f"MQTT publish rc={result.rc}")
            except Exception as exc:  # pragma: no cover - broker dependent
                last_error = exc
            LOGGER.warning(
                "MQTT %s publish attempt %s/%s failed: %s",
                label,
                attempt,
                self.settings.command_retries,
                last_error,
            )
            time.sleep(self.settings.command_retry_seconds)

        raise RuntimeError(f"Could not publish MQTT {label}: {last_error}")

    def publish_command(self, payload: dict[str, Any]) -> None:
        self._publish(self.settings.command_topic, payload, "command")

    def publish_final_score(self, payload: dict[str, Any]) -> None:
        self._publish(self.settings.final_score_topic, payload, "final_score")


class MqttPondProtocolClient:
    """AF protocol adapter for one pond feeding session."""

    def __init__(self, settings: MqttSettings, pond_id: str) -> None:
        self.settings = settings
        self.pond_id = pond_id
        self.control_topic = settings.pond_control_topic_template.format(
            pond_id=pond_id
        )
        self.status_topic = settings.pond_status_topic_template.format(pond_id=pond_id)
        self.score_topic = settings.pond_score_topic_template.format(pond_id=pond_id)
        self._client: Any | None = None
        self._connected = Event()
        self._last_pause_event = Event()
        self._scored_event = Event()
        self._error = ""

    def connect(self) -> None:
        if self._client is not None:
            return

        mqtt = mqtt_client_module()
        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"{self.settings.client_id}-{self.pond_id.lower()}-protocol",
        )
        if self.settings.username:
            client.username_pw_set(self.settings.username, self.settings.password)

        def on_connect(
            client: Any,
            _userdata: Any,
            _flags: Any,
            reason_code: Any,
            _props: Any,
        ) -> None:
            if mqtt_reason_code_failed(reason_code):
                self._error = f"MQTT connection failed: {reason_code}"
                self._connected.set()
                return
            LOGGER.info(
                "connected to MQTT broker %s:%s for pond %s",
                self.settings.host,
                self.settings.port,
                self.pond_id,
            )
            client.subscribe(self.status_topic, qos=self.settings.qos)
            self._connected.set()

        def on_message(_client: Any, _userdata: Any, message: Any) -> None:
            try:
                state = parse_af_status_payload(message.payload)
            except ValueError as exc:
                LOGGER.warning("ignored invalid AF status payload: %s", exc)
                return
            LOGGER.info(
                "received AF status pond_id=%s state=%s topic=%s",
                self.pond_id,
                state,
                message.topic,
            )
            if state == "LAST_PAUSE":
                self._last_pause_event.set()
            elif state == "SCORED":
                self._scored_event.set()

        client.on_connect = on_connect
        client.on_message = on_message
        client.connect(self.settings.host, self.settings.port, self.settings.keepalive)
        client.loop_start()
        self._client = client
        if not self._connected.wait(self.settings.connect_timeout):
            self.close()
            raise TimeoutError(
                f"Timed out connecting to MQTT broker {self.settings.host}:{self.settings.port}"
            )
        if self._error:
            self.close()
            raise RuntimeError(self._error)

    def close(self) -> None:
        if self._client is None:
            return
        self._client.loop_stop()
        self._client.disconnect()
        self._client = None

    def last_pause_received(self) -> bool:
        return self._last_pause_event.is_set()

    def _publish(self, topic: str, payload: dict[str, Any], label: str) -> None:
        if self._client is None:
            self.connect()
        assert self._client is not None

        message = json.dumps(payload, sort_keys=True)
        last_error: Exception | None = None
        for attempt in range(1, self.settings.command_retries + 1):
            try:
                result = self._client.publish(
                    topic,
                    message,
                    qos=self.settings.qos,
                )
                result.wait_for_publish()
                if result.rc == 0:
                    LOGGER.info(
                        "published MQTT %s pond_id=%s topic=%s",
                        label,
                        self.pond_id,
                        topic,
                    )
                    return
                last_error = RuntimeError(f"MQTT publish rc={result.rc}")
            except Exception as exc:  # pragma: no cover - broker dependent
                last_error = exc
            LOGGER.warning(
                "MQTT %s publish attempt %s/%s failed: %s",
                label,
                attempt,
                self.settings.command_retries,
                last_error,
            )
            time.sleep(self.settings.command_retry_seconds)

        raise RuntimeError(f"Could not publish MQTT {label}: {last_error}")

    def publish_command(self, payload: dict[str, Any]) -> None:
        command = normalize_af_state(payload.get("command", ""))
        if command == "NONE":
            return
        if command == "FINISH":
            LOGGER.info(
                "ignoring local finish command for pond %s; waiting for score/SCORED/STOP protocol",
                self.pond_id,
            )
            return
        if command not in {"START", "PAUSE", "STOP"}:
            raise ValueError(f"unsupported AF command: {command}")
        self._publish(self.control_topic, {"command": command}, "control")

    def publish_final_score(self, payload: dict[str, Any]) -> None:
        score = payload.get("score", payload.get("final_feeding_score"))
        if score is None:
            raise ValueError("final score payload must include final_feeding_score")
        score_value = float(score)
        self._scored_event.clear()
        self._publish(self.score_topic, {"score": score_value}, "score")
        if not self._scored_event.wait(self.settings.connect_timeout):
            raise TimeoutError(
                f"Timed out waiting for AF SCORED on topic {self.status_topic}"
            )
        self._publish(self.control_topic, {"command": "STOP"}, "control")
