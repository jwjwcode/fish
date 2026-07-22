# Feeding Activity Development Plan

This project has moved from a single experimental script into a package-based
V1 activity pipeline. Keep `main` runnable, use short feature branches for code
changes, and keep tuning experiments in configs/results rather than in
long-lived branches.

## Branches

- `main`: stable runnable code only.
- `feature/tune-unsupervised-segmentation`: tune sunlight/reflection/bubble
  suppression before merging into `main`.
- `feature/supervised-detector`: future neural-network detector inference path.
- `feature/hybrid-detector`: future combination of supervised masks with
  image-processing checks and scores.
- `experiment/*`: short-lived branches for risky code experiments. Do not keep
  one branch per threshold set.

## Target Structure

```text
fish_activity/
  pipeline_v1.py          # CLI orchestration for current V1 behavior
  config.py               # config loading, validation against CLI options, presets
  metadata.py             # run id, git commit, config/input traceability
  mqtt_io.py              # MQTT AF/AI pond protocol helpers
  mqtt_runtime.py         # single-stream MQTT pond protocol runtime
  mqtt_multistream_runtime.py
                          # MQTT runtime that keeps up to 4 pond streams active
  video_io.py             # writer adapters
  scoring.py              # activity score calculations
  render.py               # debug video overlays
  decision.py             # local start/pause/finish state machine
  detectors/
    base.py               # common detector result interface
    unsupervised.py       # current anomaly/splash/motion methods
    supervised.py         # trained model inference
    hybrid.py             # combine model output with image-processing checks
scripts/
  feeding_activity_v1.py  # compatibility wrapper
configs/
  *.json                  # reproducible tuning/deployment settings
```

## Detector Contract

Every detector should eventually produce the same shape of result, regardless of
whether it is unsupervised, supervised, or hybrid:

```text
mask
activity_score
confidence
debug_maps
metrics
```

The decision engine consumes detector scores and rolling-window metrics, not raw
OpenCV internals.

## Decision Engine

Use a state machine rather than scattered threshold checks:

```text
LEARNING -> FEEDING -> PAUSED -> FINISHED
```

Current local behavior:

- learn background activity from the first configured number of processed frames
- emit `start` after background learning
- after each start, observe for a configured number of seconds before deciding
- emit `pause` when the rolling activity average falls below the learned threshold
- emit `start` again after the configured pause duration
- emit `finish` after the configured pause limit is reached, or when an external
  machine finish signal is supplied

The command output is written to CSV/debug video and can also be published over
MQTT in deployment runs. A final feeding score is published once per run as the
average `total_activity` across all processed frames.

## Production Baseline

Current industry-hardening baseline:

- strict JSON config validation with unknown-key failures
- numeric range checks for CLI/config parameters
- command-line flags override config values
- structured runtime logging with configurable log level
- run metadata in CSV outputs and sidecar metadata JSON files
- unit tests for decision logic, config behavior, and scoring behavior
- headless runtime mode for deployment runs
- MQTT runtime that waits for AF `/AI/<pond_id>/init` before opening the stream
- MQTT command publishing for AF `START`/`PAUSE` controls
- MQTT final-score publishing to `/AI/<pond_id>/score`
- AF acknowledgement handling for `Last_Pause -> score -> SCORED -> STOP`
- multi-stream MQTT runtime with a configurable 4-stream default limit

Still needed for deployment-grade operation:

- video regression tests against fixed expected CSV summaries
- Jetson performance profiling
- end-to-end MQTT broker test with real AF machine behavior
- fail-safe policy for camera, MQTT, and machine-control failures
- small labeled validation set for sunlight, bubbles, ripples, and feeding states

## Refactor Order

1. Done: preserve current behavior behind a package entrypoint.
2. Done: add config loading and move repeatable tuning into configs.
3. Done: split detector/scoring/render/video-IO modules without changing CSV semantics.
4. Done: add local decision state machine and state columns to CSV.
5. Done: add config validation, structured logging, run metadata, and core tests.
6. Done: add headless mode and MQTT camera-IP runtime baseline.
7. Done: add MQTT final-score publishing and multi-stream runtime baseline.
8. Done: update MQTT runtime to AF/AI pond protocol with score acknowledgement.
9. Next: add video regression tests, Jetson profiling, and real-AF integration test.
10. Future: add stricter fail-safe policy.
11. Future: add supervised detector interface.
12. Future: add hybrid detector.
