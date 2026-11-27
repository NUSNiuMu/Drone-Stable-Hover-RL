"""RMA phase 2: train the adaptation module of a student policy.

    hover_rl/run.sh hover_rl/train_student.py --teacher runs_v2/teacher_ctbr_s0/model.pt --preset v2_ctbr_priv --seed 0

The teacher (PPO trained with the 7-D privileged vector e = [wind force in body frame / W0, mass scale - 1,
k_f scale - 1, motor tau (norm.), delay (norm.)]) is frozen. The adaptation module phi maps the last H = 50
observations (1 s, the same 26-D observation the non-adaptive PPO gets, incl. the last 2 actions) to an estimate
e_hat; the student acts with teacher(obs, e_hat). Following RMA, phi is trained by regression on states visited by
the STUDENT itself (on-policy, DAgger-style), loss = MSE in the teacher's normalised privileged space.
Writes <out>/<name>_s<seed>/{student.pt, train_log.csv, config.json}.
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--teacher", required=True)
parser.add_argument("--preset", default="v2_ctbr_priv")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--env_steps", type=int, default=30_000_000)
parser.add_argument("--hist", type=int, default=50)
parser.add_argument("--ring", type=int, default=48, help="vector steps kept for minibatch sampling")
parser.add_argument("--batch", type=int, default=4096)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--name", default="student_ctbr")
parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "runs_v2"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hover_rl.algos import PPO  # noqa: E402
from hover_rl.control import Student  # noqa: E402
from hover_rl.env import HoverWindEnv  # noqa: E402
from hover_rl.presets import make_cfg  # noqa: E402


def main():
    cfg = make_cfg(args.preset)
    assert cfg.privileged_obs
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    env = HoverWindEnv(cfg)
    dev, n = env.device, env.num_envs
    D, P = env.obs_dim - env.priv_dim, env.priv_dim
    teacher = PPO(D + P, 4, n, dev)
    teacher.load_state_dict(torch.load(args.teacher, map_location=dev))
    teacher.norm.frozen = True
    stu = Student(teacher, D, P, args.hist)
    opt = torch.optim.Adam(stu.adapter.parameters(), lr=args.lr)
    total = args.env_steps // n
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=args.lr * 0.05)

    run_dir = os.path.abspath(os.path.join(args.out, f"{args.name}_s{args.seed}"))
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(args) | {"obs_dim": D, "priv_dim": P}, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    log = open(os.path.join(run_dir, "train_log.csv"), "w")
    log.write("env_steps,loss,wind_rmse_frac_w,crash_rate,mean_pos_err,wall_s\n")

    ring_h = torch.zeros(args.ring, n, args.hist, D, device=dev)
    ring_e = torch.zeros(args.ring, n, P, device=dev)
    filled = 0
    obs, _ = env.reset(seed=args.seed)
    obs = obs["policy"]
    t0 = time.time()
    acc = {"loss": 0.0, "wind": 0.0, "crash": 0.0, "err": 0.0, "k": 0}
    for step in range(total):
        hist = stu.push(obs)
        e_true = obs[:, D:]
        with torch.no_grad():
            e_hat = stu.estimate(hist)
            act = teacher.act(torch.cat([obs[:, :D], e_hat], -1), deterministic=True)
        slot = step % args.ring
        ring_h[slot], ring_e[slot] = hist, e_true
        filled = min(filled + 1, args.ring)
        # one gradient step on a random minibatch from the last `ring` vector steps
        i = torch.randint(0, filled, (args.batch,), device=dev)
        j = torch.randint(0, n, (args.batch,), device=dev)
        pred = stu.estimate(ring_h[i, j])
        loss = F.mse_loss(stu.norm_priv(pred), stu.norm_priv(ring_e[i, j]))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        obs, _, term, trunc, _ = env.step(act)
        obs = obs["policy"]
        stu.reset(term | trunc)
        acc["loss"] += loss.item()
        acc["wind"] += ((e_hat[:, :3] - e_true[:, :3]) ** 2).sum(-1).mean().sqrt().item()
        acc["crash"] += term.float().sum().item() / max(1.0, (term | trunc).float().sum().item())
        acc["err"] += env.terminal_err.mean().item()
        acc["k"] += 1
        if (step + 1) % 64 == 0 or step == total - 1:
            k = acc["k"]
            log.write(f"{(step + 1) * n},{acc['loss'] / k:.5f},{acc['wind'] / k:.5f},{acc['crash'] / k:.4f},"
                      f"{acc['err'] / k:.4f},{time.time() - t0:.1f}\n")
            log.flush()
            print(f"[student s{args.seed}] steps {(step + 1) * n:>10d} | loss {acc['loss'] / k:.4f} | wind rmse "
                  f"{acc['wind'] / k:.4f} W | err {acc['err'] / k:.4f}", flush=True)
            acc = {"loss": 0.0, "wind": 0.0, "crash": 0.0, "err": 0.0, "k": 0}
    stu.save(os.path.join(run_dir, "student.pt"), {"teacher": os.path.abspath(args.teacher), "preset": args.preset})
    print(f"saved {run_dir}/student.pt wall {time.time() - t0:.0f}s", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
