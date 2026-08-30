#!/usr/bin/env bash
# Stage 2 validation driver: runs each model's validator sequentially on ONE pinned GPU.
# Sequential (not parallel) so the single validation GPU never OOMs. Stage 1 is untouched.
set -u
export TMPDIR="${TMPDIR:-/tmp}"
: "${HF_HOME:?set HF_HOME to your HuggingFace cache}"; export HF_HOME
GPU="${1:-3}"
export CUDA_VISIBLE_DEVICES="$GPU"
DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/out/run_all.log"
echo "=== Stage 2 validation on GPU $GPU  $(date) ===" | tee "$LOG"

run() {
  local name="$1"; local py="$2"; local script="$3"
  echo | tee -a "$LOG"
  echo ">>> $name  $(date +%H:%M:%S)" | tee -a "$LOG"
  "$py" "$script" >>"$LOG" 2>&1
  echo "    exit=$? (see out/${name}.json)" | tee -a "$LOG"
}

run qwen2_audio     ${CONDA_ROOT}/envs/qwen2-audio-env/bin/python "$DIR/validate_qwen2_audio.py"
run audio_flamingo3 ${CONDA_ROOT}/envs/audio-flamingo/bin/python   "$DIR/validate_audio_flamingo3.py"
run voxtral_mini    ${CONDA3_ROOT}/envs/voxtral/bin/python         "$DIR/validate_voxtral_mini.py"
run phi4_multimodal ${CONDA_ROOT}/envs/phi4-env/bin/python         "$DIR/validate_phi4_multimodal.py"
run audio_reasoner  ${CONDA_ROOT}/envs/audio-reasoner/bin/python   "$DIR/validate_audio_reasoner.py"

echo | tee -a "$LOG"
echo "=== ALL VALIDATORS DONE  $(date) ===" | tee -a "$LOG"
