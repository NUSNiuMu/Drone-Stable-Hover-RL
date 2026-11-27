#!/usr/bin/env bash
# TD3 v2 (CTBR) re-tuning sweep: short runs, seed 0. Usage: bash hover_rl/td3_sweep.sh TAG '{"json":"hp"}' [TAG HP ...]
# Runs every config given in parallel (call with at most 3 configs: GPU memory), then evaluates them.
cd "$(dirname "$0")/.."
STEPS=${STEPS:-25000000}
OUT=runs_v2_td3sweep
mkdir -p $OUT runs_v2/logs
tags=()
while [ $# -gt 0 ]; do
  tag=$1; hp=$2; shift 2; tags+=("$tag")
  hover_rl/run.sh hover_rl/train.py --algo td3 --preset v2_ctbr --name "td3sw_$tag" --seed 0 --env_steps $STEPS \
    --out $OUT --hp "$hp" > runs_v2/logs/td3sw_$tag.log 2>&1 &
done
wait
for tag in "${tags[@]}"; do grep -q "^saved" runs_v2/logs/td3sw_$tag.log && echo "td3sw $tag ok" || echo "td3sw $tag FAILED"; done
