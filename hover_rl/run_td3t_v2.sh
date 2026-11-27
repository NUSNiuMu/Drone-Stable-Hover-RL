#!/usr/bin/env bash
# Re-tuned TD3 (CTBR, v2): 3 seeds x 100 M steps with the configuration chosen by the sweep
# (results_v2/td3_tuning/sweep_configs.json, "chosen"): exploration/target noise 0.1 (clip 0.25), 3-step returns, lr 1e-4.
cd "$(dirname "$0")/.."
HP='{"expl_noise":0.1,"target_noise":0.1,"noise_clip":0.25,"n_step":3,"lr":1e-4}'
for s in 0 1 2; do
  ( hover_rl/run.sh hover_rl/train.py --algo td3 --preset v2_ctbr --name td3t_ctbr --seed $s --env_steps 100000000 \
      --out runs_v2 --hp "$HP" > runs_v2/logs/td3t_ctbr_s$s.log 2>&1
    grep -q "^saved" runs_v2/logs/td3t_ctbr_s$s.log && r=ok || r=FAILED; echo "td3t_ctbr s$s $r" >> runs_v2/logs/status.txt ) &
done
wait
