#!/usr/bin/env bash
# Tune the classical baselines, 3 tuning seeds each (never the evaluation seeds):
#   cascaded PID on v1 (motor RPM out) and v2 (CTBR out with the onboard rate loop, and per-motor out),
#   LQR and LQI on v1 and on v2 with per-motor output.
set -u
cd "$(dirname "$0")/.."
mkdir -p baselines/logs
JOBS=${JOBS:-"pid:v1 pid:v2_ctbr pid:v2_motor lqr:v1 lqi:v1 lqr:v2_motor lqi:v2_motor"}
for job in $JOBS; do
  ctrl=${job%%:*}; preset=${job#*:}
  for s in 11 12 13; do
    out=baselines/${ctrl}_${preset}_s${s}.json
    [ -f "$out" ] && continue
    hover_rl/run.sh hover_rl/tune_pid.py --ctrl $ctrl --preset $preset --seed $s --out $out \
      > baselines/logs/${ctrl}_${preset}_s${s}.log 2>&1
    [ -f "$out" ] && r=ok || r=FAILED
    echo "$ctrl $preset s$s $r" >> baselines/logs/status.txt
  done
done
