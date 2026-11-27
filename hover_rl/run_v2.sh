#!/usr/bin/env bash
# v2 experiment, PPO streams: per-motor commands, CTBR, and the privileged RMA teacher (3 seeds each).
#   bash hover_rl/run_v2.sh        (from the repo root)
set -u
cd "$(dirname "$0")/.."
STEPS=${STEPS:-100000000}
SEEDS=${SEEDS:-"0 1 2"}
mkdir -p runs_v2/logs
for spec in "ppo_motor v2_motor" "ppo_ctbr v2_ctbr" "teacher_ctbr v2_ctbr_priv"; do
  set -- $spec
  (
    cd "$(dirname "$0")/.." 2>/dev/null || true
    for s in $SEEDS; do
      hover_rl/run.sh hover_rl/train.py --algo ppo --preset $2 --name $1 --seed $s --env_steps $STEPS --out runs_v2 \
        > runs_v2/logs/${1}_s${s}.log 2>&1
      echo "$1 s$s exit=$?" >> runs_v2/logs/status.txt
    done
  ) &
done
wait
