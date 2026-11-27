"""Unified evaluation protocol for every controller (RL policies, cascaded PID, RMA student).

    python hover_rl/evaluate.py --preset v1                       -> results/eval_per_model.json
    python hover_rl/evaluate.py --preset v2_ctbr_priv  --out results_v2   (all CTBR controllers)
    python hover_rl/evaluate.py --preset v2_motor_priv --out results_v2   (per-motor-command controllers)

Protocol (identical for every controller / seed, deterministic policies, fixed evaluation seeds):
  * 512 parallel episodes of 12 s (600 control steps at 50 Hz), start within +-0.2 m of (0, 0, 1) m.
  * 0-6 s  : gusty wind, base strength ~ U[wind_min, wind_max] (training range), random direction.
  * 6 s    : gust pulse on top of the base wind for 0.5 s in a random horizontal direction
             (v1: +0.10 N = 36 % of the weight; v2: +40 % of the nominal weight). Random training gusts are off.
  * Position error  : mean TRUE ||p - p*|| (no sensor noise) over t in [2, 6) s.
  * Gust peak error : max error after the gust onset.
  * Recovery time   : time from gust onset until the error is < 0.05 m and stays there for 0.5 s (6 s window).
                      Episodes that never recover (or crash) are counted in `recovered_rate` and excluded from the mean.
  * Success         : no crash/out-of-bounds during the whole 12 s AND mean error over the last 2 s < 0.10 m.
  * Classical baselines (cascaded PID, LQR, LQI) are tuned on their own seeds (11-13) by tune_pid.py.
  * Wind sweep      : same episode (same gust pulse) with the base wind strength fixed to s for each s.
v2 envs are created with the privileged observation appended; controllers that were not trained with it only see the
first `obs_dim` entries (Sliced), so all controllers of one action mode run on the same episodes.
"""

import argparse
import glob
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="v1")
parser.add_argument("--runs", default=None, help="default: runs (v1) / runs_v2 (v2)")
parser.add_argument("--baselines", default=os.path.join(os.path.dirname(__file__), "..", "baselines"),
                    help="tuned classical controllers: <ctrl>_<preset>_s<seed>.json (ctrl = pid | lqr | lqi)")
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--eval_seed", type=int, default=1000)
parser.add_argument("--sweep", type=str, default=None)
parser.add_argument("--only", type=str, default=None, help="comma-separated run groups to (re-)evaluate")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from hover_rl.control import load_controller  # noqa: E402
from hover_rl.env import HoverWindEnv  # noqa: E402
from hover_rl.presets import make_cfg  # noqa: E402

TOL, REC_TOL, HOLD_S, GUST_S, EP_S = 0.10, 0.05, 0.5, 6.0, 12.0


@torch.no_grad()
def rollout(env, ctrl, seed, wind_min, wind_max, gust):
    env.cfg.wind_min, env.cfg.wind_max, env.cfg.gust_strength = wind_min, wind_max, gust
    obs, _ = env.reset(seed=seed)
    obs = obs["policy"]
    ctrl.reset()
    n, dt = env.num_envs, env.step_dt
    T = int(round(EP_S / dt)) - 1  # time-out fires at max_episode_length - 1
    err = torch.zeros(T, n, device=env.device)
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    alive_hist = torch.zeros(T, n, dtype=torch.bool, device=env.device)
    for t in range(T):
        obs, _, term, trunc, _ = env.step(ctrl.act(obs, deterministic=True))
        obs = obs["policy"]
        alive &= ~term
        err[t] = env.terminal_err
        alive_hist[t] = alive
    err, alive_hist = err.cpu().numpy(), alive_hist.cpu().numpy()
    survived = alive_hist[-1]
    times = (np.arange(T) + 1) * dt  # row t corresponds to time (t + 1) * dt
    steady = (times >= 2.0) & (times < GUST_S)
    ok_steady = alive_hist[steady].all(0)
    pos_err = err[steady][:, ok_steady].mean(0)
    last = times >= EP_S - 2.0
    success = survived & (err[last].mean(0) < TOL)
    g = int(np.argmax(times >= GUST_S))
    hold = int(round(HOLD_S / dt))
    rec = np.full(n, np.nan)
    peak = np.full(n, np.nan)
    for i in range(n):
        if not survived[i]:
            continue
        inside = err[g:, i] < REC_TOL
        peak[i] = err[g:, i].max()
        for k in range(len(inside) - hold + 1):
            if inside[k:k + hold].all():
                rec[i] = k * dt
                break
    return {
        "pos_err_m": float(np.mean(pos_err)) if len(pos_err) else float("nan"),
        "recovery_s": float(np.nanmean(rec)) if np.isfinite(rec).any() else float("nan"),
        "recovered_rate": float(np.isfinite(rec).mean()),
        "gust_peak_err_m": float(np.nanmean(peak)) if np.isfinite(peak).any() else float("nan"),
        "success_rate": float(success.mean()),
        "crash_rate": float(1.0 - survived.mean()),
    }


def discover(preset, runs_dir):
    """-> list of (name, group, kind, path, obs_dim)."""
    items = []
    if preset == "v1":
        for path in sorted(glob.glob(os.path.join(runs_dir, "*_s*", "model.pt"))):
            name = os.path.basename(os.path.dirname(path))
            algo = name.split("_s")[0]
            items.append((name, algo, algo, path, 18))
        for path in sorted(glob.glob(os.path.join(args.baselines, "*_v1_s*.json"))):
            ctrl, seed = os.path.basename(path).split("_")[0], os.path.basename(path)[:-5].split("_")[-1]
            items.append((f"{ctrl}_{seed}", ctrl, ctrl, path, 0))
        return items
    mode = "ctbr" if "ctbr" in preset else "motor"
    for path in sorted(glob.glob(os.path.join(runs_dir, "*_s*", "model.pt"))):
        cfgj = json.load(open(os.path.join(os.path.dirname(path), "config.json")))
        if cfgj.get("preset", "").replace("_priv", "") != f"v2_{mode}":
            continue
        name = os.path.basename(os.path.dirname(path))
        items.append((name, name.rsplit("_s", 1)[0], cfgj["algo"], path, cfgj["obs_dim"]))
    for path in sorted(glob.glob(os.path.join(runs_dir, "*_s*", "student.pt"))):
        cfgj = json.load(open(os.path.join(os.path.dirname(path), "config.json")))
        if cfgj["preset"].replace("_priv", "") != f"v2_{mode}":
            continue
        name = os.path.basename(os.path.dirname(path))
        items.append((name, name.rsplit("_s", 1)[0], "student", path, 0))
    for path in sorted(glob.glob(os.path.join(args.baselines, f"*_v2_{mode}_s*.json"))):
        ctrl, seed = os.path.basename(path).split("_")[0], os.path.basename(path)[:-5].split("_")[-1]
        items.append((f"{ctrl}_{mode}_{seed}", f"{ctrl}_{mode}", ctrl, path, 0))
    return items


def main():
    v1 = args.preset == "v1"
    cfg = make_cfg(args.preset)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.eval_seed
    cfg.episode_length_s = EP_S
    cfg.gust_time_s = GUST_S
    cfg.randomize_episode_start = False
    cfg.random_gust_rate_hz = 0.0
    train_wind = (cfg.wind_min, cfg.wind_max)
    env = HoverWindEnv(cfg)
    os.makedirs(args.out, exist_ok=True)
    sweep_str = args.sweep or ("0,0.02,0.04,0.06,0.08,0.10,0.12,0.14" if v1 else "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    sweep = [float(s) for s in sweep_str.split(",")]
    runs_dir = args.runs or os.path.join(ROOT, "runs" if v1 else "runs_v2")
    items = discover(args.preset, runs_dir)
    if args.only:
        groups = tuple(args.only.split(","))
        items = [it for it in items if it[1] in groups]
    out_path = os.path.join(args.out, "eval_per_model.json" if v1 else f"eval_{args.preset.replace('_priv', '')}.json")
    per_model = json.load(open(out_path))["models"] if (args.only and os.path.exists(out_path)) else {}
    for name, group, kind, path, obs_dim in items:
        ctrl = load_controller(kind, path, env, obs_dim)
        main_res = rollout(env, ctrl, args.eval_seed, train_wind[0], train_wind[1], cfg.gust_strength)
        sw = {str(s): rollout(env, ctrl, args.eval_seed + 1 + j, s, s, cfg.gust_strength) for j, s in enumerate(sweep)}
        per_model[name] = {"algo": group, "kind": kind, "main": main_res, "sweep": sw}
        print(name, json.dumps(main_res), flush=True)
        del ctrl
        torch.cuda.empty_cache()

    unit = 1.0 if v1 else env.W0
    with open(out_path, "w") as f:
        json.dump({"protocol": {"preset": args.preset, "num_envs": args.num_envs, "eval_seed": args.eval_seed,
                                "episode_s": EP_S, "gust_time_s": GUST_S, "gust_N": cfg.gust_strength * unit,
                                "gust_duration_s": cfg.gust_duration_s, "tol_m": TOL, "recovery_tol_m": REC_TOL,
                                "hold_s": HOLD_S, "train_wind_N": [w * unit for w in train_wind],
                                "robot_weight_N": env.W0, "sweep": sweep, "sweep_unit": unit,
                                "wind_unit": "N" if v1 else "fraction of nominal weight"},
                   "models": per_model}, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
