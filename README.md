# Fish Feeding Activity V1

Fast first experiment for visualizing feeding activity in fish-feeding videos.

This previous-method version uses:

- fixed middle-70% width ROI, full frame height
- fixed unsupervised splash/foam segmentation
- Farneback optical flow
- segmentation-masked flow
- weighted score display in a compact bottom strip

`total_activity` is a weighted score:

```text
segmentation_score = seg_weight * segmentation_activity_pct
optical_flow_score = flow_weight * optical_flow_activity
total_activity = segmentation_score + optical_flow_score
```

Default weights:

```text
seg_weight = 1.0
flow_weight = 0.1
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python scripts/feeding_activity_v1.py \
  A6_20260506T161254_abw722_biomass14944_feed12_feedcap12_feedremaining-0_score5.mkv
```

Useful tuning flags:

```bash
--seg-method splash      # fixed splash/foam heuristic, default
--seg-method motion      # older motion-based segmentation
--flow-mask none         # use raw optical flow instead of segmentation-masked flow
--seg-weight 1           # segmentation contribution to total_activity
--flow-weight 0.1        # optical-flow contribution to total_activity
```
