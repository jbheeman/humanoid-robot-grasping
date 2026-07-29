#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
if [[ "$mode" != "smoke10" && "$mode" != "smoke200" && "$mode" != "full2000" ]]; then
  echo "usage: $0 {smoke10|smoke200|full2000}" >&2
  exit 2
fi

experiment_root="/home/aarav/gr00t-g1-bunny-n17"
repo_root="$experiment_root/Isaac-GR00T"
dataset_root="$experiment_root/datasets"
config_path="$experiment_root/tools/gr00t_g1_bunny_config.py"
run_root="$experiment_root/runs/$mode"

cd "$repo_root"
source .venv/bin/activate
source scripts/activate_spark.sh

python -c 'import gr00t, torch, torchcodec; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
hf auth whoami >/dev/null

if [[ -e "$run_root" ]]; then
  echo "refusing to overwrite existing run directory: $run_root" >&2
  exit 3
fi
mkdir -p "$run_root"

python - <<'PY' > "$run_root/environment.json"
import json
import platform
import subprocess
import torch

print(json.dumps({
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name(0),
    "gr00t_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
}, indent=2, sort_keys=True))
PY

common=(
  gr00t/experiment/launch_finetune.py
  --base-model-path nvidia/GR00T-N1.7-3B
  --embodiment-tag NEW_EMBODIMENT
  --modality-config-path "$config_path"
  --num-gpus 1
  --learning-rate 1e-4
  --state-dropout-prob 0.1
  --episode-sampling-rate 1.0
  --shard-size 512
  --num-shards-per-epoch 2048
  --dataloader-num-workers 4
  --color-jitter-params brightness 0.2 contrast 0.2 saturation 0.2 hue 0.04
  --output-dir "$run_root/checkpoints"
)

case "$mode" in
  smoke10)
    dataset_path="$dataset_root/real_train"
    steps=10
    batch=1
    accumulation=1
    save_steps=10
    save_limit=1
    ;;
  smoke200)
    dataset_path="$dataset_root/real_train"
    steps=200
    batch=2
    accumulation=16
    save_steps=100
    save_limit=2
    ;;
  full2000)
    dataset_path="$dataset_root/real_train:$dataset_root/sim_train"
    steps=2000
    batch=2
    accumulation=16
    save_steps=250
    save_limit=8
    alpha="$(
      python - <<'PY'
import json
from pathlib import Path
report = json.loads(
    Path("/home/aarav/gr00t-g1-bunny-n17/datasets/CONVERSION_REPORT.json").read_text()
)
print(report["ds_weights_alpha"])
PY
    )"
    common+=(--ds-weights-alpha "$alpha")
    ;;
esac

command=(
  python
  "${common[@]}"
  --dataset-path "$dataset_path"
  --experiment-name "g1-bunny-n17-$mode"
  --max-steps "$steps"
  --global-batch-size "$batch"
  --gradient-accumulation-steps "$accumulation"
  --save-steps "$save_steps"
  --save-total-limit "$save_limit"
)

printf '%q ' "${command[@]}" > "$run_root/command.sh"
printf '\n' >> "$run_root/command.sh"
"${command[@]}" 2>&1 | tee "$run_root/train.log"
