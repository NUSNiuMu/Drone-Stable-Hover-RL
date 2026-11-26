#!/usr/bin/env bash
# Full experiment: 3 algorithms x 3 seeds x 50M env steps, then evaluation + report.
#   bash hover_rl/run_all.sh            (from the repo root)
set -u
cd "$(dirname "$0")/.."
STEPS=${STEPS:-50000000}
SEEDS=${SEEDS:-"0 1 2"}
mkdir -p runs/logs
for algo in ppo td3 dqn; do
  (
    for s in $SEEDS; do
      hover_rl/run.sh hover_rl/train.py --algo $algo --seed $s --env_steps $STEPS > runs/logs/${algo}_s${s}.log 2>&1
      echo "$algo s$s exit=$?" >> runs/logs/status.txt
    done
  ) &
done
wait
hover_rl/run.sh hover_rl/evaluate.py > runs/logs/eval.log 2>&1; echo "eval exit=$?" >> runs/logs/status.txt
VENV=${VENV:-$HOME/workspace/isaaclab-env}
env -u PYTHONPATH "$VENV/bin/python" hover_rl/report.py > runs/logs/report.log 2>&1; echo "report exit=$?" >> runs/logs/status.txt
