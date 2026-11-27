"""Train one algorithm with one seed on the unified wind-hover task.

    python hover_rl/train.py --algo ppo --seed 0 --env_steps 20000000 [--preset v2_ctbr] [--name ppo_ctbr]

Writes <out>/<name or algo>_s<seed>/{model.pt, train_log.csv, config.json}. The FINAL policy is saved (no best-checkpoint
selection) so that evaluation is not biased by picking the luckiest snapshot.
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=["ppo", "td3", "dqn"], required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--env_steps", type=int, default=20_000_000, help="total environment transitions")
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "runs"))
parser.add_argument("--preset", default="v1", help="v1 | v2_motor | v2_ctbr | v2_motor_priv | v2_ctbr_priv")
parser.add_argument("--name", default=None, help="run name prefix (default: algo)")
parser.add_argument("--hp", default="{}", help='JSON of agent constructor overrides, e.g. {"expl_noise": 0.1}')
parser.add_argument("--log_every", type=int, default=32, help="vector steps between log rows")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hover_rl.algos import ALGOS  # noqa: E402
from hover_rl.env import HoverWindEnv  # noqa: E402
from hover_rl.presets import make_cfg  # noqa: E402


def main():
    torch.backends.cudnn.deterministic = True
    cfg = make_cfg(args.preset)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = HoverWindEnv(cfg)
    dev = env.device
    hp = json.loads(args.hp)
    agent = ALGOS[args.algo](env.obs_dim, 4, env.num_envs, dev, **hp)

    run_dir = os.path.abspath(os.path.join(args.out, f"{args.name or args.algo}_s{args.seed}"))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"algo": args.algo, "preset": args.preset, "obs_dim": env.obs_dim, "seed": args.seed, "num_envs": args.num_envs, "env_steps": args.env_steps, "hp": hp,
                   "robot_mass_kg": env.robot_mass, "robot_weight_N": env.robot_weight, "k_f": env.k_f,
                   "step_dt": env.step_dt, "episode_length_s": cfg.episode_length_s,
                   "wind_N": [cfg.wind_min, cfg.wind_max]}, f, indent=2)
    log = open(os.path.join(run_dir, "train_log.csv"), "w")
    log.write("env_steps,episodes,mean_return,crash_rate,mean_pos_err,wall_s,extra\n")

    obs, _ = env.reset(seed=args.seed)
    obs = obs["policy"]
    n = env.num_envs
    ep_ret = torch.zeros(n, device=dev)
    fresh = torch.zeros(n, dtype=torch.bool, device=dev)  # first (randomly truncated) episodes are not logged
    done_ret, done_crash = [], []
    err_acc, err_cnt = 0.0, 0
    episodes, vec_steps, t0, info = 0, 0, time.time(), None
    total_vec = args.env_steps // n

    while vec_steps < total_vec:
        act = agent.act(obs)
        nobs, rew, term, trunc, _ = env.step(act)
        nobs = nobs["policy"]
        out = agent.observe(nobs, rew, term, trunc, env.terminal_obs)
        info = out or info
        ep_ret += rew
        err_acc += env.terminal_err.mean().item()
        err_cnt += 1
        ended = term | trunc
        if ended.any():
            ids = ended & fresh
            if ids.any():
                done_ret.append(ep_ret[ids])
                done_crash.append(term[ids].float())
                episodes += int(ids.sum())
            ep_ret[ended] = 0.0
            fresh |= ended
        obs = nobs
        vec_steps += 1
        if vec_steps % max(1, 5_000_000 // n) == 0:  # periodic checkpoints (model.pt is overwritten by the final model)
            torch.save(agent.state_dict(), os.path.join(run_dir, "model.pt"))
            os.makedirs(os.path.join(run_dir, "snapshots"), exist_ok=True)
            torch.save(agent.state_dict(), os.path.join(run_dir, "snapshots", f"step_{vec_steps * n // 1_000_000:03d}M.pt"))
        if vec_steps % args.log_every == 0 or vec_steps == total_vec:
            if done_ret:
                r = torch.cat(done_ret)
                c = torch.cat(done_crash)
                mr, cr = r.mean().item(), c.mean().item()
            else:
                mr, cr = float("nan"), float("nan")
            extra = ";".join(f"{k}={v:.4g}" for k, v in (info or {}).items())
            log.write(f"{vec_steps * n},{episodes},{mr:.3f},{cr:.4f},"
                      f"{err_acc / max(1, err_cnt):.4f},{time.time() - t0:.1f},{extra}\n")
            log.flush()
            print(f"[{args.algo} s{args.seed}] steps {vec_steps * n:>10d} | eps {episodes:>7d} | "
                  f"ret {mr:8.2f} | crash {cr:.3f} | err {err_acc / max(1, err_cnt):.3f} | {extra}", flush=True)
            done_ret, done_crash = [], []
            err_acc, err_cnt = 0.0, 0

    torch.save(agent.state_dict(), os.path.join(run_dir, "model.pt"))
    print(f"saved {run_dir}/model.pt  wall {time.time() - t0:.0f}s", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
