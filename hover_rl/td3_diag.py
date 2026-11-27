"""Diagnose a TD3 (or PPO) v2 policy's steady hover offset.

    hover_rl/run.sh hover_rl/td3_diag.py --model runs_v2/td3_ctbr_s0/model.pt --algo td3 --out results_v2/td3_diag_v1hp.json

For several fixed base winds (no gust) it reports, over t in [2, 6) s (TRUE state, deterministic policy):
  * the mean position-error vector decomposed into along-wind / cross-wind / vertical components (cm),
  * the RMS of the error around that mean (cm) = jitter vs. bias,
  * the mean commanded collective thrust and body rates,
and for TD3 the critic's estimate Q(s, pi(s)) at t = 2 s vs. the realised discounted return of the deterministic
policy from that state (over-estimation check).
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--algo", default="td3")
parser.add_argument("--preset", default="v2_ctbr")
parser.add_argument("--out", required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--winds", default="0,0.15,0.3")
parser.add_argument("--hp", default="{}", help="JSON of constructor kwargs the model was trained with")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hover_rl.algos import ALGOS  # noqa: E402
from hover_rl.env import HoverWindEnv  # noqa: E402
from hover_rl.presets import make_cfg  # noqa: E402

EP_S = 12.0


@torch.no_grad()
def main():
    cfg = make_cfg(args.preset)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 1000
    cfg.episode_length_s = EP_S
    cfg.gust_time_s = 100.0  # no scripted gust
    cfg.randomize_episode_start = False
    cfg.random_gust_rate_hz = 0.0
    env = HoverWindEnv(cfg)
    hp = json.loads(args.hp)
    hp.pop("n_step", None)
    kw = {"capacity": 1, **hp} if args.algo == "td3" else {}
    agent = ALGOS[args.algo](env.obs_dim, 4, env.num_envs, env.device, **kw)
    agent.load_state_dict(torch.load(args.model, map_location=env.device))
    agent.norm.frozen = True
    gamma = getattr(agent, "gamma", 0.99)
    dt = env.step_dt
    T = int(round(EP_S / dt)) - 1
    res = {}
    for w in [float(x) for x in args.winds.split(",")]:
        env.cfg.wind_min = env.cfg.wind_max = w
        env.cfg.gust_strength = 0.0
        obs, _ = env.reset(seed=1000)
        obs = obs["policy"]
        errs, acts, rews, alive_l, yaw_l = [], [], [], [], []
        q0 = None
        alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for t in range(T):
            a = agent.act(obs, deterministic=True)
            if args.algo == "td3" and t == int(2.0 / dt):
                no = agent.norm(obs)
                x = torch.cat([no, a], -1)
                q0 = torch.min(agent.q1(x), agent.q2(x)).squeeze(-1)
            obs, r, term, _, _ = env.step(a)
            obs = obs["policy"]
            alive &= ~term
            pos = env._robot.data.root_pos_w - env.scene.env_origins
            errs.append(pos - env.target_local)
            acts.append(a)
            yaw_l.append(env._robot.data.root_ang_vel_b[:, 2].clone())
            rews.append(r)
            alive_l.append(alive.clone())
        E, A, R, AL = (torch.stack(x) for x in (errs, acts, rews, alive_l))
        times = (torch.arange(T, device=env.device) + 1) * dt
        m = (times >= 2.0) & (times < 6.0)
        ok = AL[m].all(0)
        wd = env.wind_force_w[:, :2]
        wd = wd / wd.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        Em = E[m][:, ok]
        mean_vec = Em.mean(0)  # (n_ok, 3)
        wdo = wd[ok]
        along = (mean_vec[:, :2] * wdo).sum(-1)
        cross = (mean_vec[:, 0] * -wdo[:, 1] + mean_vec[:, 1] * wdo[:, 0]).abs()
        jitter = (Em - mean_vec).norm(dim=-1).pow(2).mean(0).sqrt()
        out = {"n_ok": int(ok.sum()), "along_wind_cm": float(along.mean() * 100), "cross_wind_abs_cm": float(cross.mean() * 100),
               "vertical_cm": float(mean_vec[:, 2].mean() * 100), "bias_norm_cm": float(mean_vec.norm(dim=-1).mean() * 100),
               "jitter_rms_cm": float(jitter.mean() * 100), "mean_err_cm": float(Em.norm(dim=-1).mean() * 100),
               "thrust_cmd": float(A[m][:, ok, 0].mean()), "rate_cmd_abs": [float(v) for v in A[m][:, ok, 1:].abs().mean((0, 1))],
               "rate_cmd_signed_abs_mean": [float(v) for v in A[m][:, ok, 1:].mean(0).abs().mean(0)],
               "yaw_rate_abs_mean_rad_s": float(torch.stack(yaw_l)[m][:, ok].mean(0).abs().mean()),
               "rate_cmd_step_change": float((A[1:] - A[:-1])[m[1:]][:, ok, 1:].abs().mean())}
        if q0 is not None:
            k0 = int(2.0 / dt)
            disc = gamma ** torch.arange(T - k0, device=env.device).float()
            G = (R[k0:] * disc.unsqueeze(-1)).sum(0)  # truncated at episode end (~10 s => gamma^500 ~ 0.007)
            okq = AL[-1]
            out["Q_pred"] = float(q0[okq].mean())
            out["G_real"] = float(G[okq].mean())
            out["Q_overestimate_pct"] = float((q0[okq].mean() / G[okq].mean() - 1) * 100)
        res[str(w)] = out
        print(w, json.dumps(out), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"model": args.model, "algo": args.algo, "hp": hp, "per_wind": res}, open(args.out, "w"), indent=2)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
    os._exit(0)
