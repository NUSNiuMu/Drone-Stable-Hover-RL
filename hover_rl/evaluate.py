"""Unified evaluation protocol for all trained policies.

    python hover_rl/evaluate.py --runs runs --out results

Protocol (identical for every algorithm / seed, deterministic policy, fixed evaluation seeds):
  * 512 parallel episodes of 12 s (600 control steps at 50 Hz), start within +-0.2 m of (0, 0, 1) m.
  * 0-6 s  : gusty wind, base strength ~ U[wind_min, wind_max] N, random direction (same model as training).
  * 6 s    : gust pulse - an additional 0.10 N (~36 % of the drone's weight) in a random horizontal direction
             for 0.5 s on top of the base wind (random training gusts are switched off during evaluation).
  * Position error  : mean ||p - p*|| over t in [2, 6) s (steady wind, initial transient excluded).
  * Recovery time   : time from gust onset until the error is < 0.05 m and stays there for 0.5 s.
                      Measured on a 6 s window, so values are NOT capped at 3 s. Episodes that never recover
                      (or crash) are counted in `recovered_rate` and excluded from the mean.
  * Success         : no crash/out-of-bounds during the whole 12 s AND mean error over the last 2 s < 0.10 m.
  * Wind sweep      : same episode (same 0.10 N gust pulse) with base wind strength fixed to s for each s.
Results are averaged over episodes per model, then mean +- std across training seeds.
"""

import argparse
import glob
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--runs", default=os.path.join(os.path.dirname(__file__), "..", "runs"))
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--eval_seed", type=int, default=1000)
parser.add_argument("--sweep", type=str, default="0,0.02,0.04,0.06,0.08,0.10,0.12,0.14")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hover_rl.algos import ALGOS  # noqa: E402
from hover_rl.env import HoverWindEnv, HoverWindEnvCfg  # noqa: E402

TOL, REC_TOL, HOLD_S, GUST_S, EP_S = 0.10, 0.05, 0.5, 6.0, 12.0


@torch.no_grad()
def rollout(env, agent, seed, wind_min, wind_max, gust):
    env.cfg.wind_min, env.cfg.wind_max, env.cfg.gust_strength = wind_min, wind_max, gust
    obs, _ = env.reset(seed=seed)
    obs = obs["policy"]
    n, dt = env.num_envs, env.step_dt
    T = int(round(EP_S / dt)) - 1  # time-out fires at max_episode_length - 1
    err = torch.zeros(T, n, device=env.device)
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    alive_hist = torch.zeros(T, n, dtype=torch.bool, device=env.device)
    for t in range(T):
        act = agent.act(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(act)
        obs = obs["policy"]
        e = env.terminal_obs[:, 15:18].norm(dim=-1)
        alive &= ~term
        err[t] = e
        alive_hist[t] = alive
    err, alive_hist = err.cpu().numpy(), alive_hist.cpu().numpy()
    survived = alive_hist[-1]
    # steps are 1-based in time: row t corresponds to time (t + 1) * dt
    times = (np.arange(T) + 1) * dt
    steady = (times >= 2.0) & (times < GUST_S)
    ok_steady = alive_hist[steady].all(0)
    pos_err = err[steady][:, ok_steady].mean(0)  # per-episode mean error
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


def main():
    cfg = HoverWindEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.episode_length_s = EP_S
    cfg.gust_time_s = GUST_S
    cfg.randomize_episode_start = False
    cfg.random_gust_rate_hz = 0.0
    train_wind = (cfg.wind_min, cfg.wind_max)
    env = HoverWindEnv(cfg)
    os.makedirs(args.out, exist_ok=True)
    sweep = [float(s) for s in args.sweep.split(",")]

    per_model = {}
    for path in sorted(glob.glob(os.path.join(args.runs, "*_s*", "model.pt"))):
        name = os.path.basename(os.path.dirname(path))
        algo = name.split("_s")[0]
        agent = ALGOS[algo](18, 4, env.num_envs, env.device)
        agent.load_state_dict(torch.load(path, map_location=env.device))
        agent.norm.frozen = True
        main_res = rollout(env, agent, args.eval_seed, train_wind[0], train_wind[1], cfg.gust_strength)
        sw = {s: rollout(env, agent, args.eval_seed + 1 + j, s, s, cfg.gust_strength) for j, s in enumerate(sweep)}
        per_model[name] = {"algo": algo, "main": main_res, "sweep": sw}
        print(name, json.dumps(main_res), flush=True)

    with open(os.path.join(args.out, "eval_per_model.json"), "w") as f:
        json.dump({"protocol": {"num_envs": args.num_envs, "eval_seed": args.eval_seed, "episode_s": EP_S,
                                "gust_time_s": GUST_S, "gust_N": cfg.gust_strength, "gust_duration_s": cfg.gust_duration_s, "tol_m": TOL, "recovery_tol_m": REC_TOL, "hold_s": HOLD_S, "train_wind_N": train_wind,
                                "robot_weight_N": env.robot_weight, "sweep_N": sweep},
                   "models": per_model}, f, indent=2)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
