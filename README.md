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

By default, `segmentation_activity_pct` uses an adaptive unsupervised splash
anomaly model. It learns normal water/ripple appearance and detects abnormal
bright/white, textured, edged, and motion-supported splash pixels. This is
intended to suppress smooth wind ripples.

The current preset also applies component-level artifact filtering after
segmentation. It removes components that look like smooth/coherent ripple,
persistent sunlight reflection, or small persistent white bubbles using
component texture, edge density, flow chaos, and temporal persistence.

By default, `optical_flow_activity` is a splash-flow energy score: optical flow
inside the segmentation mask, weighted by mask area. This suppresses ripple-only
optical flow or tiny false masks. The CSV also includes
`optical_flow_raw_activity` for comparison.

The default optical-flow method is `auto`, which uses DIS optical flow when this
OpenCV build provides it and falls back to Farneback otherwise.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python scripts/feeding_activity_v1.py \
  A6_20260506T161254_abw722_biomass14944_feed12_feedcap12_feedremaining-0_score5.mkv
```

This writes:

- `A6_20260506T161254_abw722_biomass14944_feed12_feedcap12_feedremaining-0_score5_activity_v1.mp4`
- `A6_20260506T161254_abw722_biomass14944_feed12_feedcap12_feedremaining-0_score5_activity_v1.csv`

For a quick short test:

```bash
.venv/bin/python scripts/feeding_activity_v1.py \
  A6_20260506T161254_abw722_biomass14944_feed12_feedcap12_feedremaining-0_score5.mkv \
  --duration 30 \
  --frame-step 2
```

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
--flow-method farneback  # compare against the fallback optical-flow method
--flow-mask none         # use raw optical flow instead of segmentation-masked flow
--artifact-filter off    # disable ripple/reflection/bubble component filtering
--seg-weight 1           # segmentation contribution to total_activity
--flow-weight 0.03       # optical-flow contribution to total_activity
```
