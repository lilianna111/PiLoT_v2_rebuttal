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
output_base=${OUTPUT_BASE:-/media/amax/PortableSSD/0908/Translation-only}
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

sequences=(
  "DJI_20250612182732_0001_V"
  "DJI_20250612183852_0005_V"
  "DJI_20250612194903_0021_V"
  "DJI_20250612193930_0012_V"
  "DJI_20250612194150_0014_V"
  "DJI_20250612194622_0018_V"
)
magnitudes=(0 5 10 15)
directions=(0 90 180 270)

manifest="$run_root/manifest.csv"
if [[ ! -e "$manifest" ]]; then
  echo "magnitude_m,direction_deg,sample_num,seed,sequence,status,output_folder" > "$manifest"
fi

for magnitude_m in "${magnitudes[@]}"; do
  if [[ "$magnitude_m" -eq 0 ]]; then
    run_directions=(0)
  else
    run_directions=("${directions[@]}")
  fi

  for direction_deg in "${run_directions[@]}"; do
    condition=$(printf "r%02dm_dir%03ddeg" "$magnitude_m" "$direction_deg")
    output_folder="$run_root/$condition/sample${sample_num}/seed${seed}"
    mkdir -p "$output_folder"

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
          --translation_sensitivity \
          --prior_translation_m "$magnitude_m" \
          --prior_translation_direction_deg "$direction_deg" \
          --random_seed "$seed" > "$log_file" 2>&1; then
        status="complete"
      else
        status="failed"
      fi
      echo "$magnitude_m,$direction_deg,$sample_num,$seed,$sequence,$status,$output_folder" >> "$manifest"
    done
  done
done

if ! "$python_bin" summarize_translation_prior.py "$run_root"; then
  echo "Failed to summarize experiment: $run_root" >&2
  exit 1
fi
echo "Translation-only experiment saved to: $run_root"
