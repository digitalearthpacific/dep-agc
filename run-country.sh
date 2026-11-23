#!/usr/bin/env bash
# Explicit serial Docker command sequence, not a scheduler or task generator.
# Inputs/results are one approved country tree; historical comparisons stay out.
set -euo pipefail
if [[ $# -ne 8 ]]; then
  echo 'Usage: bash run-country.sh IMAGE_ID COUNTRY YEAR COUNTRY_ROOT MODEL_ROOT SOURCE_ROOT RESULTS_ROOT RUN_PREFIX' >&2
  exit 2
fi
image=$1 country=$2 year=$3 data=$4 model=$5 source=$6 results=$7 prefix=$8
[[ "$image" =~ ^sha256:[a-f0-9]{64}$ ]] || { echo 'Pinned local image ID required' >&2; exit 2; }
for path in "$data" "$model" "$source"; do
  [[ "$path" = /* && -d "$path" ]] || { echo 'Explicit existing absolute artifact directory required' >&2; exit 2; }
done
[[ -f "$data/countries.json" && "$results" = /* && "$prefix" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || exit 2
if [[ ! -e "$results" ]]; then mkdir -m 1777 "$results"; fi
[[ -d "$results" ]] || exit 2
ids=$(docker run --pull never --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --user 65534:65534 --memory 1g --cpus 1 \
  --mount "type=bind,src=$data,dst=/data,readonly" \
  --entrypoint /opt/adapter-venv/bin/python "$image" -B -m spc_agb_dep.print_tasks \
  --catalog /data/countries.json --country "$country" --year "$year" --pending-only --ids)
if [[ -z "$ids" ]]; then echo 'All planned jobs verified complete; no dispatch'; exit 0; fi
mapfile -t windows <<<"$ids"
max_jobs=${AGC_MAX_JOBS:-${#windows[@]}}
[[ "$max_jobs" =~ ^[0-9]+$ && ${#windows[@]} -le "$max_jobs" ]] || { echo 'Explicit job allowance exceeded before dispatch' >&2; exit 2; }
for window in "${windows[@]}"; do
  safe=${window//+/p}
  name="$prefix-$country-$safe"
  if docker container inspect "$name" >/dev/null 2>&1; then
    echo "Existing container $name preserved; do not restart or retry it." >&2
    exit 1
  fi
  # Failed/incomplete output or an existing container name is a hard stop.
  # No --rm, automatic retry, restart, old height mount, or host Python mount.
  if ! timeout 720 docker run --name "$name" --pull never --runtime=nvidia --gpus all \
      --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
      --user 65534:65534 --memory 6g --memory-swap 6g --cpus 8 \
      --tmpfs /tmp:rw,noexec,nosuid,size=268435456 \
      --mount "type=bind,src=$data,dst=/data,readonly" \
      --mount "type=bind,src=$model,dst=/model,readonly" \
      --mount "type=bind,src=$source,dst=/source,readonly" \
      --mount "type=bind,src=$results,dst=/work" \
      "$image" --catalog /data/countries.json --country "$country" --window "$window" \
      --year "$year" --source-version 0.4.0 --model /model --source /source \
      --output "/work/$window" --worker-timeout 600 --skip-completed; then
    docker stop --time 0 "$name" || true
    echo "Stopped at $window; container/logs/partial output retained. No retry." >&2
    exit 1
  fi
done
