#!/usr/bin/env bash
# TD3 with CTBR actions on v2 (3 seeds, 100M steps like the v2 PPO runs); waits for the PPO streams to free the GPU.
set -u
cd "$(dirname "$0")/.."
STEPS=${STEPS:-100000000}
until [ "$(grep -cE '^(ppo|teacher)' runs_v2/logs/status.txt 2>/dev/null)" -ge 9 ]; do sleep 60; done
# one seed at a time: only 3 Isaac Sim processes fit into the 12 GB GPU together
for s in 0 1 2; do
  hover_rl/run.sh hover_rl/train.py --algo td3 --preset v2_ctbr --name td3_ctbr --seed $s --env_steps $STEPS \
    --out runs_v2 > runs_v2/logs/td3_ctbr_s$s.log 2>&1
  grep -q "^saved" runs_v2/logs/td3_ctbr_s$s.log && r=ok || r=FAILED
  echo "td3_ctbr s$s $r" >> runs_v2/logs/status.txt
done
