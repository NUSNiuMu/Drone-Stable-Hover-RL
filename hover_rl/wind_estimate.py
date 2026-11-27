"""Log the RMA student's online wind estimate against the true wind (evaluation episodes, eval seed).

    hover_rl/run.sh hover_rl/wind_estimate.py --students runs_v2/student_ctbr_s*/student.pt --out results_v2

For base wind w in {0.3 (inside the training range), 0.6 (outside)} x nominal weight, 512 episodes of 12 s with the
evaluation gust (+0.40 W at 6 s for 0.5 s). The privileged vector of the env gives the true wind force in the body
frame; both are rotated to the world frame with the true attitude for plotting. Saves
<out>/wind_estimate.npz (one example episode per case + per-step RMSE over episodes) and wind_estimate.json.
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--students", nargs="+", required=True)
parser.add_argument("--preset", default="v2_ctbr_priv")
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results_v2"))
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--eval_seed", type=int, default=1000)
parser.add_argument("--cases", default="0.3,0.6")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hover_rl.control import Student  # noqa: E402
from hover_rl.env import HoverWindEnv  # noqa: E402
from hover_rl.presets import make_cfg  # noqa: E402
from isaaclab.utils.math import quat_rotate  # noqa: E402

EP_S, GUST_S = 12.0, 6.0


@torch.no_grad()
def run(env, stu, seed, w):
    env.cfg.wind_min = env.cfg.wind_max = w
    obs, _ = env.reset(seed=seed)
    obs = obs["policy"]
    stu.reset()
    D = stu.base_dim
    T = int(round(EP_S / env.step_dt)) - 1
    tru, est = np.zeros((T, env.num_envs, 3), np.float32), np.zeros((T, env.num_envs, 3), np.float32)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for t in range(T):
        a = stu.act(obs)
        q = env._robot.data.root_quat_w
        est[t] = quat_rotate(q, stu.e_hat[:, :3]).cpu().numpy()
        tru[t] = quat_rotate(q, obs[:, D:D + 3]).cpu().numpy()
        obs, _, term, _, _ = env.step(a)
        obs = obs["policy"]
        alive &= ~term
    return tru, est, alive.cpu().numpy()


def main():
    cfg = make_cfg(args.preset)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.eval_seed
    cfg.episode_length_s = EP_S
    cfg.gust_time_s = GUST_S
    cfg.randomize_episode_start = False
    cfg.random_gust_rate_hz = 0.0
    env = HoverWindEnv(cfg)
    cases = [float(c) for c in args.cases.split(",")]
    times = (np.arange(int(round(EP_S / env.step_dt)) - 1)) * env.step_dt
    save, summary = {"t": times}, {}
    for si, path in enumerate(args.students):
        stu = Student.load(path, env.num_envs, env.device)
        name = os.path.basename(os.path.dirname(path))
        for ci, w in enumerate(cases):
            tru, est, alive = run(env, stu, args.eval_seed + 100 + ci, w)
            e = np.linalg.norm(est - tru, axis=-1)[:, alive]  # (T, n_alive), fraction of W0
            mag_t = np.linalg.norm(tru[:, alive, :2], axis=-1)
            mag_e = np.linalg.norm(est[:, alive, :2], axis=-1)
            steady = (times >= 2.0) & (times < GUST_S)
            gust = (times >= GUST_S) & (times < GUST_S + 0.5)
            summary[f"{name}@{w}"] = {
                "rmse_all": float(np.sqrt((e[times >= 1.0] ** 2).mean())),
                "rmse_steady": float(np.sqrt((e[steady] ** 2).mean())),
                "rmse_gust": float(np.sqrt((e[gust] ** 2).mean())),
                "magnitude_bias_steady": float((mag_e[steady] - mag_t[steady]).mean()),
                "corr_xy": float(np.corrcoef(est[:, alive, :2].ravel(), tru[:, alive, :2].ravel())[0, 1]),
                "alive": float(alive.mean())}
            print(name, w, json.dumps(summary[f"{name}@{w}"]), flush=True)
            k = int(np.argmax(alive))
            save[f"s{si}_w{w}_true"] = tru[:, k]
            save[f"s{si}_w{w}_est"] = est[:, k]
            save[f"s{si}_w{w}_rmse_t"] = np.sqrt((e**2).mean(1))
    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "wind_estimate.npz"), **save)
    json.dump({"weight_N": env.W0, "cases": cases, "students": args.students, "summary": summary},
              open(os.path.join(args.out, "wind_estimate.json"), "w"), indent=2)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
