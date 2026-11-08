#!/usr/bin/env bash
# GPU confirmation runs on SkyPilot (cloud/gpu-baselines.yaml). Reruns the
# GPU-dependent part of the planner-versus-direct study on a Lambda GPU,
# follows it and copies the results back into results/direct/remote/.
# Run from a checkout with `sky check` passing. The cluster installs the lazyqc
# release pinned in the script lockfiles from PyPI. SkyPilot comes from a
# standalone install if one is on PATH, otherwise from the `remote` group.
#
#   scripts/sky_gpu.sh dryrun                     instance and price, nothing launched
#   scripts/sky_gpu.sh launch [sky launch flags]  e.g. GPU=A100:1 scripts/sky_gpu.sh launch
#   scripts/sky_gpu.sh exec                       re-sync and queue the task on the running cluster
#   scripts/sky_gpu.sh logs | status | collect | watch | down
#
# GPU selects the accelerator (default H100:1). The cluster tears itself down
# AUTODOWN_MINUTES after the job ends (default 30), which leaves time to
# collect; `watch` collects every WATCH_MINUTES (default 2) until it is gone.
# Record the instance-hours and cost after each run.
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTER=${CLUSTER:-lazyqc-gpu}
GPU=${GPU:-H100:1}
AUTODOWN_MINUTES=${AUTODOWN_MINUTES:-30}
YAML=cloud/gpu-baselines.yaml
OUT=results/direct/remote
STAGE=results/direct/.remote-staging

sky() {
  if command -v sky > /dev/null; then
    command sky "$@"
  else
    uv run --locked --group remote sky "$@"
  fi
}

preflight() {
  for runner in run_direct_baselines run_gradient_baselines; do
    [ -f "benchmarks/$runner.py.lock" ] || {
      echo "benchmarks/$runner.py.lock is missing: run uv lock --script benchmarks/$runner.py" >&2
      exit 1
    }
  done
  if git rev-parse --git-dir > /dev/null 2>&1 &&
    { ! git diff --quiet HEAD -- benchmarks scripts cloud ||
      [ -n "$(git ls-files --others --exclude-standard -- benchmarks scripts cloud)" ]; }; then
    echo "Note: uncommitted files are synced as they are;" \
      "every result file records the hash of the sources it ran." >&2
  fi
}

collect() {
  # Pull into a staging copy, then promote only complete files, so that a JSON
  # file read while the cluster was rewriting it never replaces a good copy.
  # Nothing local is ever deleted: a cluster that vanishes takes nothing with it.
  mkdir -p "$STAGE" "$OUT"
  rsync -az --exclude '*.tmp' "$CLUSTER:sky_workdir/$OUT/" "$STAGE/" || return 1
  (cd "$STAGE" && find . -type f) | while read -r file; do
    if [[ $file == *.json ]] &&
      ! python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "$STAGE/$file" 2> /dev/null; then
      echo "$(date -u +%FT%TZ) kept the previous copy of $file (incomplete read)" >&2
      continue
    fi
    if ! cmp -s "$STAGE/$file" "$OUT/$file"; then
      mkdir -p "$(dirname "$OUT/$file")"
      cp -p "$STAGE/$file" "$OUT/$file.tmp" && mv "$OUT/$file.tmp" "$OUT/$file"
    fi
  done
}

case "${1:-}" in
  dryrun)
    sky launch -c "$CLUSTER" "$YAML" --gpus "$GPU" --dryrun
    ;;
  launch)
    shift
    preflight
    mkdir -p "$OUT"  # uploaded to the cluster so that a relaunch resumes
    sky launch -c "$CLUSTER" "$YAML" --gpus "$GPU" \
      --idle-minutes-to-autostop "$AUTODOWN_MINUTES" --down --detach-run --yes "$@"
    echo "Submitted. Follow with: $0 logs"
    ;;
  exec)
    # Re-sync the code to a running cluster and queue the task again; steps
    # with a .done marker on the cluster are skipped, new steps run.
    shift
    preflight
    sky exec "$CLUSTER" "$YAML" --detach-run "$@"
    ;;
  logs)
    sky logs "$CLUSTER"
    ;;
  status)
    sky status "$CLUSTER"
    sky queue "$CLUSTER"
    ;;
  collect)
    collect
    find "$OUT" -name complete.txt -exec echo "complete: {}" \;
    echo "Summarize with: uv run --locked --group notebooks python benchmarks/summarize_direct.py"
    ;;
  watch)
    # Collect every WATCH_MINUTES until the cluster has been gone for three
    # checks in a row; one failed call (network, API) never ends the watch.
    misses=0
    while [ "$misses" -lt 3 ]; do
      if collect; then
        misses=0
        echo "$(date -u +%FT%TZ) collected; done steps: $(find "$OUT" -name '*.done' | wc -l)"
      elif sky status "$CLUSTER" 2> /dev/null | grep -q "^$CLUSTER "; then
        echo "$(date -u +%FT%TZ) collect failed; cluster still listed, retrying" >&2
      else
        misses=$((misses + 1))
      fi
      sleep $((${WATCH_MINUTES:-2} * 60))
    done
    echo "Cluster gone; every completed case collected so far is in $OUT."
    ;;
  down)
    sky down --yes "$CLUSTER"
    ;;
  *)
    sed -n '2,16p' "$0"
    exit 1
    ;;
esac
