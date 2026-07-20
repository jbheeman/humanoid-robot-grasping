#!/usr/bin/env bash
# Prepare and fine-tune UniFoLM-VLA on Unitree's G1_Dex1_Stack_Block dataset.
#
# This is intended to run in tmux.  It is restart-safe: Hugging Face downloads,
# HDF5 episodes, the RLDS build, and accelerate checkpoints are all retained.
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLA_ROOT="${PROJECT_ROOT}/third_party/unifolm-vla"
VENV="${PROJECT_ROOT}/.unifolm-vla-venv"
DATA_ROOT="${PROJECT_ROOT}/data/vla/g1_stack_block"
LEROBOT_ROOT="${DATA_ROOT}/lerobot/G1_Dex1_Stack_Block"
HDF5_ROOT="${DATA_ROOT}/hdf5"
RLDS_ROOT="${DATA_ROOT}/rlds"
MODEL_ROOT="${PROJECT_ROOT}/models/pretrained/UnifoLM-VLM-Base"
RUN_ROOT="${PROJECT_ROOT}/models/unifolm_vla"
RUN_ID="${RUN_ID:-g1_stack_block_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${PROJECT_ROOT}/logs/unifolm_vla"
HF_HOME="${HF_HOME:-${PROJECT_ROOT}/data/cache/huggingface}"

# The upstream code pins CUDA 12.4/PyTorch 2.5.1.  This GB10 already has a
# functional CUDA 13.0/PyTorch 2.13 stack, so we deliberately do not downgrade
# torch.  Set VLA_PYTHON to a separately prepared compatible interpreter if
# needed; otherwise the project environment is used.
VLA_PYTHON="${VLA_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
export HF_HOME TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$DATA_ROOT" "$HDF5_ROOT" "$RLDS_ROOT" "$MODEL_ROOT" "$RUN_ROOT" "$LOG_DIR"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }
die() { log "ERROR: $*"; exit 1; }

[[ -d "$VLA_ROOT/.git" ]] || die "Missing ${VLA_ROOT}; clone the pinned UniFoLM repository first."
[[ -x "$VLA_PYTHON" ]] || die "Missing Python interpreter: ${VLA_PYTHON}"

log "Using UniFoLM revision $(git -C "$VLA_ROOT" rev-parse HEAD)"
log "Run ID: $RUN_ID"

# Install the repository itself without dependency resolution so its historical
# torch==2.5.1 pin cannot replace the GB10's validated CUDA build.
"$VLA_PYTHON" -m pip install --disable-pip-version-check --no-deps -e "$VLA_ROOT"
"$VLA_PYTHON" -m pip install --disable-pip-version-check \
  'huggingface_hub[cli]>=0.34.4' \
  h5py opencv-python tensorflow-datasets==4.9.3 tensorflow-metadata==1.14.0 tensorflow==2.16.1
"$VLA_PYTHON" -m pip install --disable-pip-version-check --no-deps \
  'lerobot @ git+https://github.com/huggingface/lerobot.git@0878c68'
"$VLA_PYTHON" -m pip install --disable-pip-version-check --no-deps \
  'dlimp @ git+https://github.com/kvablack/dlimp@d08da3852c149548aaa8551186d619d87375df08'
"$VLA_PYTHON" -m pip install --disable-pip-version-check \
  transformers==4.52.3 accelerate==1.5.2 tiktoken einops \
  transformers_stream_generator==0.0.4 scipy pillow==11.3.0 tensorboard matplotlib \
  websocket-client==1.8.0 albumentations==1.4.18 \
  pydantic==2.10.6 pyarrow==15.0.1 fastparquet==2024.11.0 av \
  numpydantic==1.6.9 deepspeed==0.16.9 qwen-vl-utils omegaconf wandb==0.16.6 rich \
  diffusers==0.35.1 tyro==0.9.35 fastapi uvicorn json_numpy peft bitsandbytes \
  datasets==3.6.0 jsonlines==4.0.0
"$VLA_PYTHON" -m pip install --disable-pip-version-check \
  protobuf==3.20.3 tensorflow-metadata==1.14.0
"$VLA_PYTHON" -m pip install --disable-pip-version-check --no-deps \
  tensorflow_graphics==2021.12.3

log "Downloading the LeRobot source dataset (resumes if interrupted)"
"${VLA_PYTHON%/python}/hf" download \
  unitreerobotics/G1_Dex1_Stack_Block \
  --repo-type dataset \
  --max-workers 2 \
  --local-dir "$LEROBOT_ROOT"

log "Downloading the VLM initialization checkpoint (resumes if interrupted)"
"${VLA_PYTHON%/python}/hf" download \
  unitreerobotics/UnifoLM-VLM-Base \
  --local-dir "$MODEL_ROOT"

log "Converting LeRobot v2.1 episodes to HDF5"
"$VLA_PYTHON" "$VLA_ROOT/prepare_data/convert_lerobot_to_hdf5.py" \
  --repo-id unitreerobotics/G1_Dex1_Stack_Block \
  --data_path "$LEROBOT_ROOT" \
  --target_path "$HDF5_ROOT" \
  --workers "${CONVERSION_WORKERS:-16}"

TFDS_VERSION_DIR="$RLDS_ROOT/rlds_dataset/1.0.0"
if [[ -f "$TFDS_VERSION_DIR/dataset_info.json" ]] && compgen -G "$TFDS_VERSION_DIR/*.tfrecord*" >/dev/null; then
  log "Reusing validated RLDS training dataset at $TFDS_VERSION_DIR"
else
  log "Building the RLDS training dataset"
  pushd "$VLA_ROOT/prepare_data/hdf5_to_rlds" >/dev/null
  HDF5_ROOT="$HDF5_ROOT" "$VLA_PYTHON" -m tensorflow_datasets.scripts.cli.main build \
    rlds_dataset --data_dir "$RLDS_ROOT" --overwrite
  popd >/dev/null
fi

# UniFoLM expects <data_root>/g1_stack_block/1.0.0.  Keep a stable symlink to
# TFDS's generated version directory without duplicating the converted dataset.
mkdir -p "$RLDS_ROOT/g1_stack_block"
[[ -d "$TFDS_VERSION_DIR" ]] || die "TFDS build did not produce rlds_dataset/1.0.0"
ln -sfn "$TFDS_VERSION_DIR" "$RLDS_ROOT/g1_stack_block/1.0.0"

log "Starting single-GPU fine-tuning; checkpoints save under $RUN_ROOT/$RUN_ID"
cd "$VLA_ROOT"
RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  [[ -f "$RESUME_CHECKPOINT" ]] || die "Resume checkpoint not found: $RESUME_CHECKPOINT"
  [[ -n "${RESUME_STEP:-}" ]] || die "RESUME_STEP is required with RESUME_CHECKPOINT"
  log "Loading model weights from $RESUME_CHECKPOINT at step $RESUME_STEP"
  RESUME_ARGS=(
    --trainer.pretrained_checkpoint "$RESUME_CHECKPOINT"
    --trainer.resume_step "$RESUME_STEP"
  )
fi
"$VLA_PYTHON" -m accelerate.commands.launch \
  --config_file src/unifolm_vla/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 1 \
  src/unifolm_vla/training/train_unifolm_vla.py \
  --config_yaml ./src/unifolm_vla/config/training/unifolm_vla_train.yaml \
  --framework.framework_py unifolm_vla \
  --framework.qwenvl.base_vlm "$MODEL_ROOT" \
  --framework.qwenvl.model_type qwen2_5_vl \
  --framework.qwenvl.attn_implementation sdpa \
  --framework.qwenvl.use_lora True \
  --framework.qwenvl.load_in_4bit True \
  --framework.qwenvl.gradient_checkpointing True \
  --framework.qwenvl.lora_rank "${LORA_RANK:-16}" \
  --framework.qwenvl.lora_alpha "${LORA_ALPHA:-32}" \
  --framework.qwenvl.lora_dropout "${LORA_DROPOUT:-0.05}" \
  --datasets.vla_data.data_root_dir "$RLDS_ROOT" \
  --datasets.vla_data.data_mix g1_stack_block \
  --datasets.vla_data.window_size 1 \
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE:-1}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS:-30000}" \
  --trainer.shuffle_buffer_size 10000 \
  --trainer.save_interval "${SAVE_INTERVAL:-500}" \
  --trainer.use_wrist_image True \
  --trainer.use_proprio True \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 30001 \
  --trainer.learning_rate.base 2e-4 \
  "${RESUME_ARGS[@]}" \
  --run_root_dir "$RUN_ROOT" \
  --run_id "$RUN_ID" \
  --wandb_project "${WANDB_PROJECT:-unifolm_vla}" \
  --wandb_entity "${WANDB_ENTITY:-}"
