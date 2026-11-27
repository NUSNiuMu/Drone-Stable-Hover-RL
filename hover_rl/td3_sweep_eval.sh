#!/usr/bin/env bash
# Evaluate TD3 sweep runs (v2 protocol + 30/50 % wind) and diagnose their static hover bias.
# Usage: bash hover_rl/td3_sweep_eval.sh TAG [TAG ...]
cd "$(dirname "$0")/.."
only=$(printf "td3sw_%s," "$@"); only=${only%,}
EMPTY=$(mktemp -d)
hover_rl/run.sh hover_rl/evaluate.py --preset v2_ctbr_priv --runs runs_v2_td3sweep --baselines "$EMPTY" \
  --out results_v2/td3_tuning --only "$only" --sweep 0.3,0.5 > runs_v2/logs/td3sw_eval.log 2>&1
grep -E "^td3sw_" runs_v2/logs/td3sw_eval.log
for tag in "$@"; do
  hp=$(python3 -c "import json;print(json.dumps(json.load(open('runs_v2_td3sweep/td3sw_${tag}_s0/config.json')).get('hp',{})))")
  hover_rl/run.sh hover_rl/td3_diag.py --model runs_v2_td3sweep/td3sw_${tag}_s0/model.pt --algo td3 --hp "$hp" \
    --winds 0,0.3 --out results_v2/td3_tuning/diag_${tag}.json > runs_v2/logs/td3sw_diag_${tag}.log 2>&1
  echo "diag $tag: $(python3 -c "import json;d=json.load(open('results_v2/td3_tuning/diag_${tag}.json'))['per_wind'];print({w:round(v['bias_norm_cm'],2) for w,v in d.items()})")"
done
rmdir "$EMPTY"
