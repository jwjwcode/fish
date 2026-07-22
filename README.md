# Fish Feeding Activity V1

Fast first experiment for visualizing feeding activity in a fish-feeding video.

The script uses the middle 70% of the frame width and the full frame height as
the measurement ROI. It tints the non-ROI side bands and draws a ROI border on
the image. The segmentation mask preview and optical-flow magnitude preview use
the full-frame layout, but non-ROI areas are blacked out so only the ROI signal
is visualized. The previews and activity numbers are shown in a compact bottom
strip so they do not cover the video.

`total_activity` is a weighted score:

```text
segmentation_score = seg_weight * segmentation_activity_pct
optical_flow_score = flow_weight * optical_flow_activity
total_activity = segmentation_score + optical_flow_score
```

The annotated video also shows `Prev10 avg`, the average total score from the
previous processed frames, up to 10 frames. The CSV includes matching previous
10-frame averages for segmentation, optical-flow, and total scores.

By default, `segmentation_activity_pct` uses an adaptive unsupervised splash
anomaly model. It learns normal water/ripple appearance and detects abnormal
bright/white, textured, edged, and motion-supported splash pixels. This is
intended to suppress smooth wind ripples.

The current preset also applies component-level artifact filtering after
segmentation. It removes components that look like smooth/coherent ripple,
persistent sunlight reflection, or small persistent white bubbles using
component texture, edge density, flow chaos, temporal persistence, and smooth
bright/white appearance. White regions supported only by flow must also have
texture/edge support by default, which helps reject smooth sunlight reflection.

By default, `optical_flow_activity` is a splash-flow energy score: optical flow
inside the segmentation mask, weighted by mask area. This suppresses ripple-only
optical flow or tiny false masks. The CSV also includes
`optical_flow_raw_activity` for comparison.

The default optical-flow method is `auto`, which uses DIS optical flow when this
OpenCV build provides it and falls back to Farneback otherwise.
Use `--flow-method none` to skip optical-flow calculation entirely. This makes
the output a segmentation-only activity score and sets the flow columns to zero.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Project layout

```text
scripts/feeding_activity_v1.py       # compatibility CLI wrapper
scripts/feeding_activity_mqtt_runtime.py
scripts/feeding_activity_mqtt_multistream_runtime.py
fish_activity/pipeline_v1.py         # V1 CLI orchestration
fish_activity/config.py              # config loading and presets
fish_activity/detectors/base.py      # common detector result contract
fish_activity/detectors/unsupervised.py
fish_activity/scoring.py             # activity score calculations
fish_activity/render.py              # debug video overlays
fish_activity/video_io.py            # video writer helpers
fish_activity/decision.py            # local feeding start/pause/finish logic
fish_activity/mqtt_io.py             # MQTT camera/command/final-score helpers
fish_activity/mqtt_runtime.py        # single-stream MQTT runtime
fish_activity/mqtt_multistream_runtime.py
configs/                             # reproducible tuning/deployment settings
docs/development_plan.md             # branch/refactor roadmap
```

## Run

```bash
.venv/bin/python scripts/feeding_activity_v1.py \
  A6_20260506T161254_abw722_biomass14944_feed12_feedcap12_feedremaining-0_score5.mkv
```

This writes:

- `A6_20260506T161254_abw722_biomass14944_feed12_feedcap12_feedremaining-0_score5_activity_v1.mp4`
- `A6_20260506T161254_abw722_biomass14944_feed12_feedcap12_feedremaining-0_score5_activity_v1.xlsx`

Add `--csv debug.csv` only when you need the full debug CSV and sidecar
metadata JSON.

For a quick short test:

```bash
.venv/bin/python scripts/feeding_activity_v1.py \
  A6_20260506T161254_abw722_biomass14944_feed12_feedcap12_feedremaining-0_score5.mkv \
  --duration 30 \
  --frame-step 2
```

To run from a repeatable config:

```bash
.venv/bin/python scripts/feeding_activity_v1.py \
  A6_20260506T161254_abw722_biomass14944_feed12_feedcap12_feedremaining-0_score5.mkv \
  --config configs/unsup_no_flow.json
```

Config keys use the same names as CLI options, without the leading `--`. Nested
sections are allowed; command-line flags override matching config values.
Unknown config keys fail fast so typos do not silently change an experiment.

## Presets

Current optimized V1:

```bash
.venv/bin/python scripts/feeding_activity_v1.py \
  A6_20260506T161254_abw722_biomass14944_feed12_feedcap12_feedremaining-0_score5.mkv \
  --preset current \
  -o activity_current.mp4 \
  --csv activity_current.csv
```

Previous splash-mask + Farneback version:

```bash
.venv/bin/python scripts/feeding_activity_v1.py \
  A6_20260506T161254_abw722_biomass14944_feed12_feedcap12_feedremaining-0_score5.mkv \
  --preset previous \
  -o activity_previous.mp4 \
  --csv activity_previous.csv
```

Older motion-mask + raw-flow comparison:

```bash
.venv/bin/python scripts/feeding_activity_v1.py \
  A6_20260506T161254_abw722_biomass14944_feed12_feedcap12_feedremaining-0_score5.mkv \
  --preset motion_raw \
  -o activity_motion_raw.mp4 \
  --csv activity_motion_raw.csv
```

Useful tuning flags:

```bash
--preset current         # adaptive anomaly seg + artifact filter + DIS/auto flow
--preset previous        # fixed splash seg + Farneback + masked flow
--preset motion_raw      # older motion seg + Farneback + raw flow
--warmup-frames 0        # show segmentation immediately
--resize-width 0         # keep original 5120x1440 resolution
--diff-min-threshold 12  # make segmentation less sensitive
--bg-var-threshold 36    # make background subtraction less sensitive
--seg-method anomaly     # adaptive water/ripple anomaly segmentation, default
--seg-method splash      # fixed splash/foam heuristic
--seg-method motion      # use the older motion-based segmentation
--flow-method none       # skip optical flow for faster segmentation-only scoring
--flow-method farneback  # compare against the fallback optical-flow method
--flow-mask none         # use raw optical flow instead of segmentation-masked flow
--artifact-filter off    # disable ripple/reflection/bubble component filtering
--anomaly-flow-foam-requires-texture on
--artifact-bright-min-value 165
--artifact-bright-min-white-score 110
--artifact-bright-max-texture-mean 18
--artifact-bright-max-edge-density 0.08
--artifact-bright-min-age 3
--artifact-static-max-flow-mean 0.45
--artifact-specular-min-area-pct 0.05
--artifact-specular-min-value 172
--artifact-specular-max-saturation 105
--artifact-specular-max-texture-or-edge-density 0.52
--anomaly-texture-flow-splash on
--anomaly-texture-flow-min-texture 7.5
--anomaly-texture-flow-min-edge 13
--anomaly-texture-flow-min-flow 0.55
--seg-weight 1           # segmentation contribution to total_activity
--flow-weight 0.03       # optical-flow contribution to total_activity
--log-level INFO         # DEBUG, INFO, WARNING, or ERROR
--progress-interval 100  # log every N processed frames, 0 disables progress logs
```

## Reproducibility

When `--csv` is provided, the run writes reproducibility metadata into each CSV
row:

```text
run_id
run_started_utc
code_version
git_commit
input_path
config_path
```

It also writes a sidecar metadata file next to the CSV:

```text
<csv_stem>.metadata.json
```

This records the run id, git commit, input/output paths, config path, and a
compact settings snapshot.

Each run writes a compact Excel workbook next to the output video by default.
Use `--excel path/to/scores.xlsx` to choose a different path. The `scores` sheet
contains only:

```text
time_s
current_segmentation_score
current_optical_flow_score
current_total_activity
previous_10_total_activity_average
```

The `summary` sheet stores the final feeding score after the full run.

## Tests

Run the test suite with:

```bash
.venv/bin/python -m unittest discover -s tests
```

The tests cover decision logic, config validation/CLI override behavior, and
core scoring behavior.

Artifact tuning configs:

```bash
--config configs/tune_sun_reflection.json
--config configs/tune_less_reflection.json
--config configs/tune_less_bubble.json
--config configs/tune_strict_splash.json
```

Use `tune_sun_reflection` when smooth specular glare is the main false positive,
use `tune_less_reflection` for a milder reflection setting, use
`tune_less_bubble` when persistent white bubbles are over-segmented, and use
`tune_strict_splash` when false positives are more costly than missing weak
splash.

## Feeding decision logic

The V1 pipeline now includes local command logic for control experiments. It
writes command/state fields to the CSV/debug video and can publish commands
through the MQTT runtime. The annotated video shows the current command, the
last command with its
timestamp, and a highlighted bottom-strip banner for a few seconds after each
`start`, `pause`, or `finish` event.

Default behavior:

```text
first 10 processed frames -> learn background activity
start command             -> begin feeding window
after 45 seconds          -> compare last 10-frame average to threshold
low activity              -> pause command
after 120 seconds paused  -> start command
after 2 allowed pauses    -> finish command on the next low-activity decision
```

Decision tuning flags:

```bash
--decision-mode on
--decision-background-frames 10
--decision-window-frames 10
--decision-observe-seconds 45
--decision-pause-seconds 120
--decision-threshold-margin 0.5
--decision-threshold-multiplier 1.2
--decision-max-pauses 2
--decision-machine-finish-second 0
```

The decision threshold is:

```text
max(background_score + decision_threshold_margin,
    background_score * decision_threshold_multiplier)
```

## MQTT runtime

The deployment runtime follows the AF/AI pond protocol. `<pond_id>` is one of
`A1`-`A8`, `B1`-`B8`, or `C1`-`C8`.

| Topic | Direction | Payload |
| --- | --- | --- |
| `/AI/<pond_id>/init` | AF -> AI | `{"IP":"192.168.46.24:8080","BM":12500.0,"ABW":750.0,"FC":150.0}` |
| `/AI/<pond_id>/control` | AI -> AF | `{"command":"START"}`, `{"command":"PAUSE"}`, `{"command":"STOP"}` |
| `/AI/<pond_id>/status` | AF -> AI | `{"state":"FEED_STARTED"}`, `{"state":"Paused"}`, `{"state":"Last_Pause"}`, `{"state":"SCORED"}` |
| `/AI/<pond_id>/score` | AI -> AF | `{"score":8.0}` |

Sequence:

```text
AF init -> AI START/PAUSE decisions -> AF Last_Pause -> AI score
-> AF SCORED -> AI STOP -> AF next init
```

If `IP` is already a full `rtsp://`, `rtmp://`, `http://`, or `https://` URL, it
is used directly. Otherwise the runtime applies `--camera-url-template`.

Example:

```bash
.venv/bin/python scripts/feeding_activity_mqtt_runtime.py \
  --mqtt-host 192.168.1.100 \
  --camera-url-template 'rtsp://{ip}/stream1' \
  -- \
  --config configs/tune_less_bubble.json \
  --headless on \
  --csv results/runtime/latest.csv \
  --duration 0
```

Pipeline options go after `--`. The MQTT runtime defaults to headless mode if
`--headless` is not provided. The final score is the average `total_activity`
across all processed frames in that feeding run.

Manual local MQTT test:

```bash
mosquitto_sub -h localhost -t '/AI/#' -v
```

In another terminal, start one runtime run against a local video path:

```bash
.venv/bin/python scripts/feeding_activity_mqtt_runtime.py \
  --mqtt-host localhost \
  --max-runs 1 \
  --camera-url-template '{ip}' \
  -- \
  --config configs/tune_less_bubble.json \
  --headless on \
  --csv results/mqtt_manual/A4.csv
```

Then simulate AF messages:

```bash
mosquitto_pub -h localhost -t /AI/A4/init -m '{"IP":"videos/example.mp4","BM":12500.0,"ABW":750.0,"FC":150.0}'
mosquitto_pub -h localhost -t /AI/A4/status -m '{"state":"Last_Pause"}'
mosquitto_pub -h localhost -t /AI/A4/status -m '{"state":"SCORED"}'
```

Expected AI messages are `{"command":"START"}` on `/AI/A4/control`, then
`{"score":...}` on `/AI/A4/score`, then `{"command":"STOP"}` on
`/AI/A4/control` after `SCORED`.

Useful runtime safety flags:

```bash
--camera-read-fail-limit 30
--command-error-policy log
```

Use `--command-error-policy stop` if a failed MQTT command publish should stop
the pipeline.

For four-stream deployment, use the multi-stream runtime. It accepts AF init
messages, keeps up to four worker processes active, and starts a replacement
when AF sends the next init after a worker publishes `STOP`.

```bash
.venv/bin/python scripts/feeding_activity_mqtt_multistream_runtime.py \
  --mqtt-host 192.168.1.100 \
  --max-streams 4 \
  --camera-url-template 'rtsp://{ip}/stream1' \
  -- \
  --config configs/tune_less_bubble.json \
  --duration 0
```

For repeatable experiments, prefer saving settings in `configs/` and writing
outputs to a named `results/` folder instead of relying on one-off shell history.
