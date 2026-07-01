"""Configuration helpers for the feeding activity pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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

    action_by_dest = {action.dest: action for action in parser._actions}
    values: dict[str, object] = {}
    for dest, value in flatten_config(config).items():
        action = action_by_dest.get(dest)
        if action is None:
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
    return values


def apply_preset(args: argparse.Namespace) -> argparse.Namespace:
    for name, value in PRESETS[args.preset].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.flow_method == "none":
        args.flow_mask = "none"
        args.flow_weight = 0.0
    return args
