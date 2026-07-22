"""Run metadata helpers for reproducible activity outputs."""

from __future__ import annotations

import argparse
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fish_activity.config import config_summary


CODE_VERSION = "feeding_activity_v1"


def git_commit(project_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def build_run_metadata(args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    return {
        "run_id": uuid.uuid4().hex,
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": CODE_VERSION,
        "git_commit": git_commit(project_root),
        "input_path": str(args.input),
        "config_path": str(args.config) if args.config is not None else "",
        "settings": config_summary(args),
    }


def csv_metadata_fields() -> list[str]:
    return [
        "run_id",
        "run_started_utc",
        "code_version",
        "git_commit",
        "input_path",
        "config_path",
    ]


def csv_metadata_values(metadata: dict[str, Any]) -> dict[str, str]:
    return {
        "run_id": str(metadata["run_id"]),
        "run_started_utc": str(metadata["run_started_utc"]),
        "code_version": str(metadata["code_version"]),
        "git_commit": str(metadata["git_commit"]),
        "input_path": str(metadata["input_path"]),
        "config_path": str(metadata["config_path"]),
    }
