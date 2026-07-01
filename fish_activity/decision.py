"""Feeding decision state machine for local V1 experiments."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass


@dataclass
class DecisionConfig:
    mode: str = "on"
    background_frames: int = 10
    window_frames: int = 10
    observe_seconds: float = 45.0
    pause_seconds: float = 120.0
    threshold_margin: float = 0.5
    threshold_multiplier: float = 1.2
    max_pauses: int = 2

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DecisionConfig":
        return cls(
            mode=args.decision_mode,
            background_frames=args.decision_background_frames,
            window_frames=args.decision_window_frames,
            observe_seconds=args.decision_observe_seconds,
            pause_seconds=args.decision_pause_seconds,
            threshold_margin=args.decision_threshold_margin,
            threshold_multiplier=args.decision_threshold_multiplier,
            max_pauses=args.decision_max_pauses,
        )


@dataclass
class DecisionUpdate:
    state: str
    command: str
    last_command: str
    last_command_time_s: float
    background_score: float
    threshold: float
    window_avg: float
    pause_count: int
    process_score: float
    finish_reason: str


class FeedingDecisionEngine:
    """Local control logic before the MQTT integration exists."""

    def __init__(self, config: DecisionConfig) -> None:
        self.config = config
        self.enabled = config.mode == "on"
        self.state = "LEARNING" if self.enabled else "DISABLED"
        self.background_scores: list[float] = []
        self.recent_scores: deque[float] = deque(maxlen=max(1, config.window_frames))
        self.background_score = 0.0
        self.threshold = 0.0
        self.pause_count = 0
        self.first_start_time_s: float | None = None
        self.current_start_time_s: float | None = None
        self.pause_start_time_s: float | None = None
        self.finish_reason = ""
        self.last_command = "none"
        self.last_command_time_s = -1.0
        self.process_score_sum = 0.0
        self.process_score_count = 0
        self.process_score = 0.0

    def update(
        self,
        time_s: float,
        total_activity: float,
        external_command: str | None = None,
    ) -> DecisionUpdate:
        self.recent_scores.append(total_activity)
        window_avg = sum(self.recent_scores) / float(len(self.recent_scores))
        command = "none"

        if not self.enabled:
            return self._result(command, window_avg, time_s)

        if self.state == "LEARNING":
            self.background_scores.append(total_activity)
            if len(self.background_scores) >= max(1, self.config.background_frames):
                self.background_score = sum(self.background_scores) / float(
                    len(self.background_scores)
                )
                self.threshold = max(
                    self.background_score + self.config.threshold_margin,
                    self.background_score * self.config.threshold_multiplier,
                )
                self.state = "FEEDING"
                self.first_start_time_s = time_s
                self.current_start_time_s = time_s
                command = "start"
            return self._result(command, window_avg, time_s)

        if self.state == "FINISHED":
            return self._result(command, window_avg, time_s)

        if self.first_start_time_s is not None:
            self.process_score_sum += total_activity
            self.process_score_count += 1
            self.process_score = self.process_score_sum / float(self.process_score_count)

        if external_command == "finish":
            return self._finish("finish", window_avg, "machine_finish", time_s)

        if self.state == "FEEDING":
            assert self.current_start_time_s is not None
            observe_elapsed = time_s - self.current_start_time_s
            if (
                observe_elapsed >= self.config.observe_seconds
                and len(self.recent_scores) >= max(1, self.config.window_frames)
                and window_avg < self.threshold
            ):
                if self.pause_count >= self.config.max_pauses:
                    return self._finish("finish", window_avg, "max_pauses", time_s)
                self.pause_count += 1
                self.state = "PAUSED"
                self.pause_start_time_s = time_s
                command = "pause"

        elif self.state == "PAUSED":
            assert self.pause_start_time_s is not None
            if time_s - self.pause_start_time_s >= self.config.pause_seconds:
                self.state = "FEEDING"
                self.current_start_time_s = time_s
                self.pause_start_time_s = None
                command = "start"

        return self._result(command, window_avg, time_s)

    def _finish(
        self,
        command: str,
        window_avg: float,
        reason: str,
        time_s: float,
    ) -> DecisionUpdate:
        self.state = "FINISHED"
        self.finish_reason = reason
        return self._result(command, window_avg, time_s)

    def _result(
        self,
        command: str,
        window_avg: float,
        time_s: float,
    ) -> DecisionUpdate:
        if command != "none":
            self.last_command = command
            self.last_command_time_s = time_s
        return DecisionUpdate(
            state=self.state,
            command=command,
            last_command=self.last_command,
            last_command_time_s=self.last_command_time_s,
            background_score=self.background_score,
            threshold=self.threshold,
            window_avg=window_avg,
            pause_count=self.pause_count,
            process_score=self.process_score,
            finish_reason=self.finish_reason,
        )
