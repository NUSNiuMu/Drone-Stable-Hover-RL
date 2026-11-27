"""Disturbance-response traces for the paper figure (F5): position error vs time in one scripted v2 episode.

    hover_rl/run.sh hover_rl/time_response.py --preset v2_ctbr_priv  --groups pid_ctbr,ppo_ctbr,td3t_ctbr,student_ctbr,teacher_ctbr --out results_v2/time_response_ctbr.npz
    hover_rl/run.sh hover_rl/time_response.py --preset v2_motor_priv --groups pid_motor,lqi_motor,lqr_motor,ppo_motor --out results_v2/time_response_motor.npz

Scripted episode (identical for every controller): base wind fixed at 30 % of the weight, +40 % gust for 0.5 s at
t = 6 s, 12 s episode, evaluation seed 1000, `--num_envs` parallel copies (they differ only in start position, wind
direction, motor lag/delay and mass/thrust randomisation). For each controller and seed the true position-error
trace of every copy is stored; results are merged into results_v2/time_response_<mode>.npz via --out (key <group>_s<seed>).
"""

import argparse
import glob
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="v2_ctbr_priv")
parser.add_argument("--groups", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--wind", type=float, default=0.30)
parser.add_argument("--out", default="results_v2/time_response.npz")
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

EP_S, GUST_S = 12.0, 6.0


def items(mode, groups):
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "runs_v2", "*_s*", "*.pt"))):
        if os.path.basename(path) not in ("model.pt", "student.pt"):
            continue
        name = os.path.basename(os.path.dirname(path))
        g = name.rsplit("_s", 1)[0]
        if g not in groups:
            continue
        cfgj = json.load(open(os.path.join(os.path.dirname(path), "config.json")))
        if cfgj.get("preset", "").replace("_priv", "") != f"v2_{mode}":
            continue
        kind = "student" if path.endswith("student.pt") else cfgj["algo"]
        out.append((name, kind, path, cfgj.get("obs_dim", 0) if kind != "student" else 0))
    for path in sorted(glob.glob(os.path.join(ROOT, "baselines", f"*_v2_{mode}_s*.json"))):
        ctrl, seed = os.path.basename(path).split("_")[0], os.path.basename(path)[:-5].split("_")[-1]
        if f"{ctrl}_{mode}" in groups:
            out.append((f"{ctrl}_{mode}_{seed}", ctrl, path, 0))
    return out


@torch.no_grad()
def main():
    mode = "ctbr" if "ctbr" in args.preset else "motor"
    cfg = make_cfg(args.preset)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 1000
    cfg.episode_length_s = EP_S
    cfg.gust_time_s = GUST_S
    cfg.randomize_episode_start = False
    cfg.random_gust_rate_hz = 0.0
    cfg.wind_min = cfg.wind_max = args.wind
    env = HoverWindEnv(cfg)
    T = int(round(EP_S / env.step_dt)) - 1
    data = dict(np.load(args.out)) if os.path.exists(args.out) else {}
    data["t"] = (np.arange(T) + 1) * env.step_dt
    for name, kind, path, obs_dim in items(mode, set(args.groups.split(","))):
        ctrl = load_controller(kind, path, env, obs_dim)
        obs, _ = env.reset(seed=1000)
        obs = obs["policy"]
        ctrl.reset()
        err = torch.zeros(T, env.num_envs, device=env.device)
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for t in range(T):
            obs, _, term, _, _ = env.step(ctrl.act(obs, deterministic=True))
            obs = obs["policy"]
            alive &= ~term
            err[t] = torch.where(alive, env.terminal_err, torch.full_like(env.terminal_err, float("nan")))
        data[name] = err.cpu().numpy()
        print(name, "peak", float(np.nanmax(data[name][int(GUST_S / env.step_dt):], 0).mean()), flush=True)
        del ctrl
        torch.cuda.empty_cache()
    np.savez_compressed(args.out, **data)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
    os._exit(0)
