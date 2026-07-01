# Feeding Activity Development Plan

This project has moved from a single experimental script into a package-based
V1 activity pipeline. Keep `main` runnable, use short feature branches for code
changes, and keep tuning experiments in configs/results rather than in
long-lived branches.

## Branches

- `main`: stable runnable code only.
- `feature/decision-engine`: future feeding pause/stop/observe state machine.
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

The command output is currently written to CSV/debug video. MQTT publishing and
receiving can reuse the same command/state fields later.

## Refactor Order

1. Done: preserve current behavior behind a package entrypoint.
2. Done: add config loading and move repeatable tuning into configs.
3. Done: split detector/scoring/render/video-IO modules without changing CSV semantics.
4. Done: add local decision state machine and state columns to CSV.
5. Future: add supervised detector interface.
6. Future: add hybrid detector.
