#!/usr/bin/env bash

set -uo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

if [[ -n ${PYTHON_BIN:-} ]]; then
  python_bin=$PYTHON_BIN
elif [[ -x /opt/anaconda3/envs/FPVLoc/bin/python ]]; then
  python_bin=/opt/anaconda3/envs/FPVLoc/bin/python
else
  python_bin=python
fi

output_base=${OUTPUT_BASE:-/media/amax/PortableSSD/0908/Yaw-only}
config_file=${CONFIG_FILE:-configs/feicuiwan_m4t.yaml}
sample_num=${SAMPLE_NUM:-500}
seed=${RANDOM_SEED:-0}
requested_run_id=${RUN_ID:-}
run_id=${requested_run_id:-run_$(date +%Y%m%d_%H%M%S)_$$}
run_root="$output_base/$run_id"

if ! "$python_bin" -c "import matplotlib, numpy, torch" >/dev/null 2>&1; then
  echo "Python environment is missing torch/numpy/matplotlib: $python_bin" >&2
  exit 1
fi

mkdir -p "$output_base"
if [[ -e "$run_root" ]]; then
  if [[ -z "$requested_run_id" ]]; then
    echo "Refusing to overwrite existing run: $run_root" >&2
    exit 1
  fi
  echo "Resuming existing run without overwriting outputs: $run_root"
else
  mkdir "$run_root"
fi

python_prefix=$("$python_bin" -c "import sys; print(sys.prefix)")
torch_lib=$(
  "$python_bin" -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
)
export LD_LIBRARY_PATH="${python_prefix}/lib:${torch_lib}:${LD_LIBRARY_PATH:-}"

if [[ $# -gt 0 ]]; then
  sequences=("$@")
else
  sequences=("DJI_20250612182732_0001_V")
fi

yaw_values=(0 5 -5 10 -10 15 -15)

yaw_label() {
  local value=$1
  if (( value > 0 )); then
    printf "yawp%02ddeg" "$value"
  elif (( value < 0 )); then
    printf "yawn%02ddeg" "$((-value))"
  else
    printf "yaw000deg"
  fi
}

manifest="$run_root/manifest.csv"
if [[ ! -e "$manifest" ]]; then
  echo "yaw_error_deg,yaw_magnitude_deg,sample_num,seed,sequence,status,output_folder" > "$manifest"
fi

for yaw_deg in "${yaw_values[@]}"; do
  condition=$(yaw_label "$yaw_deg")
  output_folder="$run_root/$condition/sample${sample_num}/seed${seed}"
  mkdir -p "$output_folder"
  magnitude_deg=${yaw_deg#-}

  for sequence in "${sequences[@]}"; do
    log_file="$output_folder/${sequence}.log"
    status_file="$output_folder/${sequence}_status.json"
    sequence_output="$output_folder/$sequence"
    if [[ -f "$status_file" ]] && "$python_bin" -c \
        "import json, sys; raise SystemExit(0 if json.load(open(sys.argv[1]))['status'] == 'complete' else 1)" \
        "$status_file"; then
      echo "Skipping completed run: $condition $sequence"
      continue
    fi
    if [[ -e "$sequence_output" || -e "$output_folder/${sequence}_config.json" || -e "$log_file" ]]; then
      echo "Preserving failed/interrupted run; use a new RUN_ID to retry: $condition $sequence"
      continue
    fi

    echo "Running $condition $sequence"
    if PYTHONHASHSEED="$seed" "$python_bin" main.py \
        --config "$config_file" \
        --name "$sequence" \
        --sample_num "$sample_num" \
        --output_folder "$output_folder" \
        --yaw_sensitivity \
        --prior_yaw_deg "$yaw_deg" \
        --random_seed "$seed" > "$log_file" 2>&1; then
      status="complete"
    else
      status="failed"
    fi
    echo "$yaw_deg,$magnitude_deg,$sample_num,$seed,$sequence,$status,$output_folder" >> "$manifest"
  done
done

if ! "$python_bin" summarize_yaw_prior.py "$run_root"; then
  echo "Failed to summarize experiment: $run_root" >&2
  exit 1
fi

echo "Yaw-only experiment saved to: $run_root"
