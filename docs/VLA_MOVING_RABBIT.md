# Moving-rabbit VLA pipeline

The Isaac jobs run on `aarav@128.114.152.190:6023` from:

```text
/home/aarav/Documents/g1-bunny-vla-workspace
```

Generation is deliberately separate from robot execution. These scripts never
connect to the physical G1.

## Review and generation gates

The current 12-case visual review is stored under
`artifacts/vla_dataset_review/moving_block_review_v19_locked_yaw_noaa`. Do not start
the policy-ready or 480-episode run until those videos and contact frames are
approved. The older `v10_fxaa` set is retained as failure documentation; it
contains temporal rendering artifacts and must not be used for training.

The generator assigns each episode an independent bunny body yaw and a bounded
straight-line travel heading. A low-discrepancy yaw sequence prevents small
batches from clustering around one orientation, while five balanced heading
bins cover approximately -21 to +21 degrees. The entire path is translated as
needed to keep its original speed/distance while retaining the launch point on
the table and the intercept inside the right-arm workspace. Review/full-batch
audits require at least 25 degrees of heading span and eight 30-degree yaw bins
for any dataset of 12 or more episodes.

The real demonstrations slide the weighted plush without changing its body
orientation. Synthetic episodes therefore sample an independent initial bunny
yaw and lock rotational degrees of freedom while leaving planar translation
dynamic. This avoids a synthetic-only spin cue while retaining full viewpoint
diversity. Rendering uses non-temporal AA mode `0`; temporal DLSS/TAA is
forbidden because it leaves detached dark eye/nose trails on moving frames.

After approval, start a 24-episode policy-ready gate with eight workers (four
per GPU):

```bash
cd /home/aarav/Documents/g1-bunny-vla-workspace
export G1_CONFIRM_DATASET_RUN=moving_block_policy_24
scripts/start_moving_block_dataset.sh moving_block_policy_24 24 8
```

After that audit passes and the dataset is approved, start the 480-episode run:

```bash
export G1_CONFIRM_DATASET_RUN=moving_block_train_480
scripts/start_moving_block_dataset.sh moving_block_train_480 480 8
```

Both runs preserve at least 100 GB of disk, stay in tmux, log per worker, and
stop rather than silently accepting failed physics/contact gates.

## Temporal policy experiments

UniFoLM supports a separate observation stride while keeping the 30 Hz action
chunk dense. Evaluate these in order on identical held-out episodes:

1. `window_size=1`, `observation_stride=1` (single frame).
2. `window_size=5`, `observation_stride=1` (133 ms history).
3. `window_size=5`, `observation_stride=4` (533 ms history).

The GB10 runtime accepts matching causal image windows through
`UnifoLMRuntime.predict_window`. Select a temporal setup from causal held-out
interception error and safety metrics, not training loss alone.

## Real-data calibration

Before the final fine-tune, collect at least three robot-stationary 60 Hz clips
where the rabbit is slid across the real table. Build the blur/motion contract:

```bash
python scripts/build_passive_motion_calibration.py clip1.mp4 clip2.mp4 clip3.mp4 \
  --output artifacts/vla_dataset_review/passive_motion_calibration.json
```

Use only train-split real frames for visual calibration. Validation and test
episodes remain untouched until model selection and final evaluation.
