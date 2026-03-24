#!/usr/bin/env bash
set -euo pipefail

log() {
  printf '[colab] %s\n' "$*"
}

usage() {
  cat <<'EOF'
Usage:
  bash colab_bootstrap.sh --repo-url <git_url> [options]

Options:
  --repo-url URL           Git URL to clone when the repo is not present.
  --repo-dir PATH          Clone/work directory. Default: /content/endingengineering
  --branch NAME            Git branch or tag to checkout. Default: main
  --mode NAME              One of:
                           smoke
                           teacher
                           cifar_baseline
                           cifar_adaptive
                           cifar_kd
                           custom
                           Default: smoke
  --epochs N               Epoch count for training modes. Default: 5
  --batch-size N           Batch size for training modes. Default: 128
  --mount-drive            Mount Google Drive before running.
  --drive-sync-dir PATH    Where to copy saved artifacts. Default:
                           /content/drive/MyDrive/endingengineering
  --teacher-path PATH      Teacher checkpoint for cifar_kd mode.
  --skip-install           Skip pip installs.
  --help                   Show this help.

Pass extra arguments to the underlying training command after `--`.

Examples:
  bash colab_bootstrap.sh \
    --repo-url https://github.com/YOUR_USERNAME/endingengineering.git \
    --mode smoke

  bash colab_bootstrap.sh \
    --repo-url https://github.com/YOUR_USERNAME/endingengineering.git \
    --mode cifar_adaptive \
    --epochs 50 \
    --batch-size 256 \
    --mount-drive \
    -- -ew 0.001
EOF
}

REPO_URL="${REPO_URL:-}"
REPO_DIR="${REPO_DIR:-/content/endingengineering}"
BRANCH="${BRANCH:-main}"
MODE="${MODE:-smoke}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MOUNT_DRIVE="${MOUNT_DRIVE:-0}"
DRIVE_SYNC_DIR="${DRIVE_SYNC_DIR:-/content/drive/MyDrive/endingengineering}"
INSTALL_EXTRAS="${INSTALL_EXTRAS:-1}"
TARGET_SPARSITY="${TARGET_SPARSITY:-0.15}"
SPARSITY_WEIGHT="${SPARSITY_WEIGHT:-0.01}"
KD_TEMPERATURE="${KD_TEMPERATURE:-4.0}"
KD_ALPHA="${KD_ALPHA:-0.7}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
CIFAR_DATA_DIR="${CIFAR_DATA_DIR:-/content/data/cifar10}"
TEACHER_PATH="${TEACHER_PATH:-}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url)
      REPO_URL="$2"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --mount-drive)
      MOUNT_DRIVE=1
      shift
      ;;
    --drive-sync-dir)
      DRIVE_SYNC_DIR="$2"
      shift 2
      ;;
    --teacher-path)
      TEACHER_PATH="$2"
      shift 2
      ;;
    --skip-install)
      INSTALL_EXTRAS=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mount_drive() {
  if [[ "$MOUNT_DRIVE" != "1" ]]; then
    return
  fi

  log "Mounting Google Drive"
  python3 - <<'PY'
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
PY
  mkdir -p "$DRIVE_SYNC_DIR"
}

ensure_repo() {
  if [[ -d "$REPO_DIR/.git" ]]; then
    log "Using existing repo at $REPO_DIR"
    git -C "$REPO_DIR" fetch origin "$BRANCH" --depth 1 || true
    git -C "$REPO_DIR" checkout "$BRANCH" || true
    git -C "$REPO_DIR" pull --ff-only origin "$BRANCH" || true
    return
  fi

  if [[ -d .git ]]; then
    REPO_DIR="$(pwd)"
    log "Using current repo at $REPO_DIR"
    return
  fi

  if [[ -z "$REPO_URL" ]]; then
    printf 'Missing --repo-url and no local git repo found.\n' >&2
    exit 1
  fi

  if [[ -e "$REPO_DIR" ]]; then
    printf 'Target repo dir already exists and is not a git repo: %s\n' "$REPO_DIR" >&2
    exit 1
  fi

  log "Cloning $REPO_URL into $REPO_DIR"
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
}

install_deps() {
  if [[ "$INSTALL_EXTRAS" != "1" ]]; then
    log "Skipping dependency install"
    return
  fi

  log "Installing Python dependencies"
  python3 -m pip install -q --upgrade pip
  python3 -m pip install -q tqdm seaborn matplotlib pillow
}

print_environment() {
  log "Environment summary"
  python3 - <<'PY'
import os
import platform
import torch
import torchvision

print(f"Python: {platform.python_version()}")
print(f"Torch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Working dir: {os.getcwd()}")
PY

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi || true
  fi
}

sync_artifacts() {
  if [[ "$MOUNT_DRIVE" != "1" ]]; then
    return
  fi

  log "Syncing artifacts to $DRIVE_SYNC_DIR"
  mkdir -p "$DRIVE_SYNC_DIR"

  for dir_name in save_CIFAR10_model save_ImageNet_model save_teacher_model; do
    if [[ -d "$REPO_DIR/$dir_name" ]]; then
      rm -rf "$DRIVE_SYNC_DIR/$dir_name"
      cp -R "$REPO_DIR/$dir_name" "$DRIVE_SYNC_DIR/$dir_name"
    fi
  done
}

run_mode() {
  cd "$REPO_DIR"

  local -a cmd
  case "$MODE" in
    smoke)
      log "Running repo smoke tests"
      python3 test_models.py
      python3 test_imagenet_models.py
      python3 test_encoder_fix.py
      ;;
    teacher)
      mkdir -p "$CIFAR_DATA_DIR"
      cmd=(
        python3 train_teacher.py
        -e "$EPOCHS"
        -b "$BATCH_SIZE"
        -d "$CIFAR_DATA_DIR"
        -s
      )
      cmd+=("${EXTRA_ARGS[@]}")
      log "Running: ${cmd[*]}"
      "${cmd[@]}"
      ;;
    cifar_baseline)
      mkdir -p "$CIFAR_DATA_DIR"
      cmd=(
        python3 cifar10.py
        -id 0
        -e "$EPOCHS"
        -b "$BATCH_SIZE"
        -d "$CIFAR_DATA_DIR"
        -s
      )
      cmd+=("${EXTRA_ARGS[@]}")
      log "Running: ${cmd[*]}"
      "${cmd[@]}"
      ;;
    cifar_adaptive)
      mkdir -p "$CIFAR_DATA_DIR"
      cmd=(
        python3 cifar10.py
        -id 1
        -e "$EPOCHS"
        -b "$BATCH_SIZE"
        -d "$CIFAR_DATA_DIR"
        -ts "$TARGET_SPARSITY"
        -sw "$SPARSITY_WEIGHT"
        --label_smoothing "$LABEL_SMOOTHING"
        -s
      )
      cmd+=("${EXTRA_ARGS[@]}")
      log "Running: ${cmd[*]}"
      "${cmd[@]}"
      ;;
    cifar_kd)
      mkdir -p "$CIFAR_DATA_DIR"
      if [[ -z "$TEACHER_PATH" ]]; then
        if [[ -f "$REPO_DIR/save_teacher_model/model_teacher.pt" ]]; then
          TEACHER_PATH="$REPO_DIR/save_teacher_model/model_teacher.pt"
        else
          printf 'cifar_kd mode requires --teacher-path or an existing save_teacher_model/model_teacher.pt\n' >&2
          exit 1
        fi
      fi
      cmd=(
        python3 cifar10.py
        -id 2
        -e "$EPOCHS"
        -b "$BATCH_SIZE"
        -d "$CIFAR_DATA_DIR"
        -ts "$TARGET_SPARSITY"
        -sw "$SPARSITY_WEIGHT"
        -temp "$KD_TEMPERATURE"
        -alpha "$KD_ALPHA"
        --label_smoothing "$LABEL_SMOOTHING"
        -tp "$TEACHER_PATH"
        -s
      )
      cmd+=("${EXTRA_ARGS[@]}")
      log "Running: ${cmd[*]}"
      "${cmd[@]}"
      ;;
    custom)
      if [[ "${#EXTRA_ARGS[@]}" -eq 0 ]]; then
        printf 'custom mode requires a command after --\n' >&2
        exit 1
      fi
      log "Running custom command: ${EXTRA_ARGS[*]}"
      "${EXTRA_ARGS[@]}"
      ;;
    *)
      printf 'Unsupported mode: %s\n' "$MODE" >&2
      exit 1
      ;;
  esac
}

main() {
  mount_drive
  ensure_repo
  install_deps
  print_environment
  run_mode
  sync_artifacts
  log "Done"
}

main "$@"
