"""Tune a classical controller (cascaded PID, LQR or LQI) by vectorised search (one parameter set per parallel env).

    hover_rl/run.sh hover_rl/tune_pid.py --ctrl pid --preset v1 --seed 11 --out baselines/pid_v1_s11.json
    hover_rl/run.sh hover_rl/tune_pid.py --ctrl lqi --preset v2_motor --seed 11 --out baselines/lqi_v2_motor_s11.json

PID parameters: the 8 gains in control.GAIN_NAMES. LQR/LQI parameters: Bryson's-rule maxima (control.LQR_NAMES /
LQI_NAMES) that define the diagonal Q and R of the Riccati problem.

Tuning uses its OWN seeds (11, 12, 13 -> three independently tuned controllers, the analogue of 3 training seeds)
and a shorter episode than the evaluation (8 s, gust at 4 s); the evaluation seed (1000+) is never used here.
Score per episode (lower is better):
    mean error over [1.5, 4) s + mean error over [4, 8) s + 0.3 * peak error after the gust + 2 m if it crashed
Search: cross-entropy method in log-parameter space within the ranges in control.py (PID: start mid-range, 20
iterations; LQR/LQI: start at the Bryson guess control.LQR_GUESS with std e^1, 30 iterations). Each iteration
evaluates 128 candidates x 8 episodes (fresh episode seed per iteration); the 16 best set the next mean/std.
Final selection: the final mean + the 31 best candidates seen are re-evaluated on 32 new episodes each.
"""

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ctrl", default="pid", choices=["pid", "lqr", "lqi"])
parser.add_argument("--preset", default="v1")
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--out", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--iters", type=int, default=None, help="CEM iterations (default: 20 PID, 30 LQR/LQI)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hover_rl.control import (GAIN_NAMES, GAIN_RANGES, LQI_NAMES, LQI_RANGES, LQR_GUESS, LQR_NAMES,  # noqa: E402
                              LQR_RANGES, make_classical)
from hover_rl.env import HoverWindEnv  # noqa: E402
from hover_rl.presets import make_cfg  # noqa: E402

EP_S, GUST_S = 8.0, 4.0


@torch.no_grad()
def score(env, gains, seed):
    pid = make_classical(args.ctrl, env, gains)
    env.reset(seed=seed)
    pid.reset()
    n, dt = env.num_envs, env.step_dt
    T = int(round(EP_S / dt)) - 1
    err = torch.zeros(T, n, device=env.device)
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    for t in range(T):
        _, _, term, _, _ = env.step(pid.act())
        alive &= ~term
        err[t] = torch.where(alive, env.terminal_err, torch.full_like(env.terminal_err, 1.0))
    times = (torch.arange(T, device=env.device) + 1) * dt
    a = err[(times >= 1.5) & (times < GUST_S)].mean(0)
    b = err[times >= GUST_S].mean(0)
    pk = err[times >= GUST_S].max(0).values
    return a + b + 0.3 * pk + 2.0 * (~alive).float(), a, pk, ~alive


def main():
    cfg = make_cfg(args.preset)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.episode_length_s = EP_S
    cfg.gust_time_s = GUST_S
    cfg.randomize_episode_start = False
    cfg.random_gust_rate_hz = 0.0
    env = HoverWindEnv(cfg)
    n, dev = env.num_envs, env.device
    NAMES, RANGES = {"pid": (GAIN_NAMES, GAIN_RANGES), "lqr": (LQR_NAMES, LQR_RANGES),
                     "lqi": (LQI_NAMES, LQI_RANGES)}[args.ctrl]
    lo = torch.tensor([math.log(RANGES[k][0]) for k in NAMES], device=dev)
    hi = torch.tensor([math.log(RANGES[k][1]) for k in NAMES], device=dev)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    K, POP, ELITE = len(NAMES), 128, 16
    reps = n // POP
    if args.ctrl == "pid":
        mu, sig = (lo + hi) / 2, (hi - lo) / 4
    else:  # LQR/LQI: start at a Bryson guess (the mid-range start converged poorly in 12-D), +-e^1 spread
        mu = torch.tensor([math.log(LQR_GUESS[k]) for k in NAMES], device=dev)
        sig = torch.ones_like(mu)
    iters = args.iters or (20 if args.ctrl == "pid" else 30)
    seen_g, seen_s, history = [], [], []
    for it in range(iters):
        z = mu + sig * torch.randn(POP, K, device=dev, generator=gen)
        z[0] = mu
        z = torch.minimum(torch.maximum(z, lo), hi)
        sc, a, pk, cr = score(env, z.exp().repeat_interleave(reps, 0), args.seed * 1000 + it)
        sc, a, pk, cr = (x.float().reshape(POP, reps).mean(1) for x in (sc, a, pk, cr))
        order = torch.argsort(sc)
        el = z[order[:ELITE]]
        mu = 0.7 * el.mean(0) + 0.3 * mu
        sig = torch.clamp(0.7 * el.std(0) + 0.3 * sig, min=0.03)
        seen_g.append(z[order[:8]]), seen_s.append(sc[order[:8]])
        i = int(order[0])
        history.append({"iter": it, "best": float(sc[i]), "steady": float(a[i]), "peak": float(pk[i]), "crash": float(cr[i]),
                        "mean_gains": {k: float(v) for k, v in zip(NAMES, mu.exp())}})
        print(f"iter {it} best {float(sc[i]):.4f} steady {float(a[i]):.4f} peak {float(pk[i]):.4f} crash {float(cr[i]):.3f}"
              f" | mean {[float(f'{float(v):.3g}') for v in mu.exp()]}", flush=True)
    G = torch.cat(seen_g)
    S = torch.cat(seen_s)
    C = torch.cat([mu.unsqueeze(0), G[torch.argsort(S)[:31]]]).exp()
    reps = n // 32
    sc, a, pk, cr = score(env, C.repeat_interleave(reps, 0), args.seed * 1000 + 999)
    sc, a, pk, cr = (x.float().reshape(32, reps).mean(1) for x in (sc, a, pk, cr))
    i = int(torch.argmin(sc))
    best = C[i]
    history.append({"final": True, "score": float(sc[i]), "steady": float(a[i]), "peak": float(pk[i]), "crash": float(cr[i])})
    print(f"final score {float(sc[i]):.4f} steady {float(a[i]):.4f} peak {float(pk[i]):.4f} crash {float(cr[i]):.3f}", flush=True)
    out = {"ctrl": args.ctrl, "preset": args.preset, "tuning_seed": args.seed,
           "gains": {k: float(v) for k, v in zip(NAMES, best)},
           "history": history}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print("GAINS", json.dumps(out["gains"]), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
