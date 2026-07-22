"""Configuration helpers for the feeding activity pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRESETS: dict[str, dict[str, object]] = {
    "current": {
        "seg_method": "anomaly",
        "flow_method": "auto",
        "flow_mask": "segmentation",
        "flow_weight": 0.03,
        "artifact_filter": "on",
    },
    "previous": {
        "seg_method": "splash",
        "flow_method": "farneback",
        "flow_mask": "segmentation",
        "flow_weight": 0.1,
        "artifact_filter": "off",
    },
    "motion_raw": {
        "seg_method": "motion",
        "flow_method": "farneback",
        "flow_mask": "none",
        "flow_weight": 0.1,
        "artifact_filter": "off",
    },
}

CONFIG_METADATA_KEYS = {
    "schema_version",
    "config_name",
    "description",
    "type",
}

CONFIG_VALUE_RANGES: dict[str, tuple[float | None, float | None]] = {
    "resize_width": (0, None),
    "start_second": (0, None),
    "duration": (0, None),
    "frame_step": (1, None),
    "preview_width": (0, None),
    "warmup_frames": (0, None),
    "bg_history": (1, None),
    "bg_var_threshold": (0, None),
    "bg_learning_rate": (-1, 1),
    "diff_percentile": (0, 100),
    "diff_min_threshold": (0, None),
    "bright_value": (0, 255),
    "bright_saturation": (0, 255),
    "splash_min_value": (0, 255),
    "splash_max_saturation": (0, 255),
    "splash_white_score": (-255, 255),
    "splash_texture_threshold": (0, None),
    "splash_edge_threshold": (0, None),
    "anomaly_learning_rate": (0, 1),
    "anomaly_color_z": (0, None),
    "anomaly_texture_z": (0, None),
    "anomaly_flow_z": (0, None),
    "anomaly_min_flow": (0, None),
    "anomaly_texture_flow_flow_z": (0, None),
    "anomaly_texture_flow_min_flow": (0, None),
    "anomaly_texture_flow_min_texture": (0, None),
    "anomaly_texture_flow_min_edge": (0, None),
    "artifact_min_texture_mean": (0, None),
    "artifact_min_edge_density": (0, 1),
    "artifact_min_flow_chaos": (0, None),
    "artifact_persistence_frames": (0, None),
    "artifact_static_new_ratio": (0, 1),
    "artifact_max_bubble_area_pct": (0, 100),
    "artifact_min_reflection_area_pct": (0, 100),
    "artifact_bright_min_value": (0, 255),
    "artifact_bright_min_white_score": (-255, 255),
    "artifact_bright_max_texture_mean": (0, None),
    "artifact_bright_max_edge_density": (0, 1),
    "artifact_bright_min_age": (0, None),
    "artifact_static_max_flow_mean": (0, None),
    "artifact_specular_min_area_pct": (0, 100),
    "artifact_specular_min_value": (0, 255),
    "artifact_specular_max_saturation": (0, 255),
    "artifact_specular_min_white_score": (-255, 255),
    "artifact_specular_max_texture_mean": (0, None),
    "artifact_specular_max_edge_density": (0, 1),
    "artifact_specular_max_texture_or_edge_density": (0, 1),
    "artifact_specular_max_flow_chaos": (0, None),
    "min_component_area": (1, None),
    "flow_percentile": (0, 100),
    "flow_min_mask_pixels": (0, None),
    "seg_weight": (0, None),
    "flow_weight": (0, None),
    "camera_read_fail_limit": (0, None),
    "decision_background_frames": (1, None),
    "decision_window_frames": (1, None),
    "decision_observe_seconds": (0, None),
    "decision_pause_seconds": (0, None),
    "decision_threshold_margin": (0, None),
    "decision_threshold_multiplier": (0, None),
    "decision_max_pauses": (0, None),
    "decision_machine_finish_second": (0, None),
    "progress_interval": (0, None),
}


def provided_cli_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest = {
        option: action.dest
        for action in parser._actions
        for option in action.option_strings
    }
    provided: set[str] = set()
    for token in argv:
        if token == "--":
            break
        option = token.split("=", 1)[0]
        if option in option_to_dest:
            provided.add(option_to_dest[option])
    return provided


def flatten_config(config: object) -> dict[str, object]:
    if not isinstance(config, dict):
        raise SystemExit("Config file must contain a JSON object.")

    flat: dict[str, object] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            nested = flatten_config(value)
            for nested_key, nested_value in nested.items():
                if nested_key in flat:
                    raise SystemExit(f"Duplicate config key: {nested_key}")
                flat[nested_key] = nested_value
        else:
            if key in flat:
                raise SystemExit(f"Duplicate config key: {key}")
            flat[key] = value
    return flat


def load_config_values(
    parser: argparse.ArgumentParser,
    config_path: Path | None,
) -> dict[str, object]:
    if config_path is None:
        return {}
    try:
        with config_path.open() as config_file:
            config = json.load(config_file)
    except OSError as exc:
        raise SystemExit(f"Could not read config file: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON config file: {config_path}: {exc}") from exc

    action_by_dest = {
        action.dest: action
        for action in parser._actions
        if action.option_strings and action.dest not in {"help", "config"}
    }
    values: dict[str, object] = {}
    unknown_keys: list[str] = []
    for dest, value in flatten_config(config).items():
        if dest in CONFIG_METADATA_KEYS:
            continue
        action = action_by_dest.get(dest)
        if action is None:
            unknown_keys.append(dest)
            continue
        if action.type is not None and value is not None:
            try:
                value = action.type(value)
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"Invalid config value for {dest}: {value}") from exc
        if action.choices is not None and value not in action.choices:
            choices = ", ".join(str(choice) for choice in action.choices)
            raise SystemExit(
                f"Invalid config value for {dest}: {value}. Choose one of: {choices}"
            )
        values[dest] = value
    if unknown_keys:
        keys = ", ".join(sorted(unknown_keys))
        raise SystemExit(f"Unknown config key(s): {keys}")
    return values


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    for name, value in PRESETS[args.preset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.flow_method == "none":
        args.flow_mask = "none"
        args.flow_weight = 0.0
    return args


def validate_args(args: argparse.Namespace) -> None:
    for name, (min_value, max_value) in CONFIG_VALUE_RANGES.items():
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if value is None:
            continue
        if not isinstance(value, (int, float)):
            continue
        if min_value is not None and value < min_value:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= {min_value}")
        if max_value is not None and value > max_value:
            raise SystemExit(f"--{name.replace('_', '-')} must be <= {max_value}")

    if args.start_second > 0 and args.duration == 0:
        return


def config_summary(args: argparse.Namespace) -> dict[str, Any]:
    """Small reproducibility snapshot for metadata sidecars/logging."""
    return {
        "preset": args.preset,
        "seg_method": args.seg_method,
        "flow_method": args.flow_method,
        "flow_mask": args.flow_mask,
        "artifact_filter": args.artifact_filter,
        "resize_width": args.resize_width,
        "frame_step": args.frame_step,
        "stop_after_finish": args.stop_after_finish,
        "seg_weight": args.seg_weight,
        "flow_weight": args.flow_weight,
        "decision_mode": args.decision_mode,
    }
