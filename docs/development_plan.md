# Feeding Activity Development Plan

This project is moving from a single experimental script toward a deployable
feeding-control pipeline. Keep `main` runnable, use short feature branches for
code changes, and keep tuning experiments in configs/results rather than in
long-lived branches.

## Branches

- `main`: stable runnable code only.
- `feature/no-flow-mode`: option to disable optical flow.
- `refactor/pipeline-core`: package structure and behavior-preserving cleanup.
- `feature/config-files`: load detector and decision settings from config files.
- `feature/decision-engine`: feeding pause/stop/observe state machine.
- `feature/supervised-detector`: neural-network detector inference path.
- `feature/hybrid-detector`: combine supervised masks with image-processing
  checks and scores.
- `experiment/*`: short-lived branches for risky code experiments. Do not keep
  one branch per threshold set.

## Target Structure

```text
fish_activity/
  pipeline_v1.py          # current behavior, kept intact during first refactor
  config.py               # future config loading and validation
  video_io.py             # capture/writer adapters
  scoring.py              # activity score calculations
  decision.py             # pause/stop/observe state machine
  render.py               # debug video overlays
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

The decision engine should consume detector outputs and rolling-window metrics,
not raw OpenCV internals.

## Decision Engine

Use a state machine rather than scattered threshold checks:

```text
WARMUP -> FEEDING -> OBSERVE -> PAUSE -> STOP
```

Decisions should be based on rolling windows:

- observe for a configured number of seconds before changing state
- pause only after low activity persists
- stop only after very low activity persists longer
- resume only after activity rises above a higher threshold

This avoids unstable pause/resume behavior.

## Refactor Order

1. Preserve current behavior behind a package entrypoint.
2. Add config loading and move detailed tuning values out of CLI defaults.
3. Split detector/scoring/render modules without changing CSV semantics.
4. Add decision state machine and state columns to CSV.
5. Add supervised detector interface.
6. Add hybrid detector.
