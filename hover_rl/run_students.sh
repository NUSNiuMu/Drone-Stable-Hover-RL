#!/usr/bin/env bash
# RMA phase 2 for each teacher seed (waits for the teacher run to finish when chained after run_v2.sh).
set -u
cd "$(dirname "$0")/.."
STEPS=${STEPS:-30000000}
for s in ${SEEDS:-0 1 2}; do
  # wait for all PPO/teacher runs (GPU memory only fits 3 Isaac processes next to each other)
  until [ "$(grep -cE '^(ppo|teacher)' runs_v2/logs/status.txt 2>/dev/null)" -ge 9 ]; do sleep 60; done
  [ -f runs_v2/student_ctbr_s$s/student.pt ] && continue
  hover_rl/run.sh hover_rl/train_student.py --teacher runs_v2/teacher_ctbr_s$s/model.pt --preset v2_ctbr_priv \
    --seed $s --env_steps $STEPS --ring 24 --out runs_v2 > runs_v2/logs/student_ctbr_s$s.log 2>&1
  # Isaac Sim's shutdown hides the python exit code -> judge success by the output file
  [ -f runs_v2/student_ctbr_s$s/student.pt ] && r=ok || r=FAILED
  echo "student_ctbr s$s $r" >> runs_v2/logs/status.txt
done
