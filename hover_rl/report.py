"""Aggregate training logs + evaluation JSON into tables and figures (pure numpy/matplotlib, no Isaac Sim).

    python hover_rl/report.py --runs runs --results results --fig figures
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
p = argparse.ArgumentParser()
p.add_argument("--runs", default=os.path.join(ROOT, "runs"))
p.add_argument("--results", default=os.path.join(ROOT, "results"))
p.add_argument("--fig", default=os.path.join(ROOT, "figures"))
args = p.parse_args()

ORDER = ["ppo", "td3", "dqn"]
LABEL = {"ppo": "PPO", "td3": "TD3", "dqn": "DQN"}
COLOR = {"ppo": "#2a78d6", "td3": "#eb6834", "dqn": "#1baf7a"}  # validated categorical slots 1-3
MARK = {"ppo": "o", "td3": "s", "dqn": "^"}
LS = {"ppo": "-", "td3": "--", "dqn": "-."}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "lines.linewidth": 2})
os.makedirs(args.fig, exist_ok=True)


def load_logs():
    logs = {}
    for f in sorted(glob.glob(os.path.join(args.runs, "*_s*", "train_log.csv"))):
        name = os.path.basename(os.path.dirname(f))
        algo = name.split("_s")[0]
        d = np.genfromtxt(f, delimiter=",", names=True, usecols=(0, 1, 2, 3, 4), dtype=float)
        logs.setdefault(algo, []).append(d)
    return logs


def smooth(y, k=5):
    y = np.asarray(y, float)
    out = np.copy(y)
    for i in range(len(y)):
        w = y[max(0, i - k + 1):i + 1]
        w = w[np.isfinite(w)]
        out[i] = w.mean() if len(w) else np.nan
    return out


def training_curves(logs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for algo in [a for a in ORDER if a in logs]:
        runs = logs[algo]
        n = min(len(r) for r in runs)
        x = runs[0]["env_steps"][:n] / 1e6
        for ax, key, ylab in ((axes[0], "mean_return", "Episode return (10 s episode)"),
                              (axes[1], "crash_rate", "Crash / out-of-bounds rate")):
            Y = np.stack([smooth(r[key][:n]) for r in runs])
            m, s = np.nanmean(Y, 0), np.nanstd(Y, 0)
            ax.plot(x, m, color=COLOR[algo], ls=LS[algo], label=f"{LABEL[algo]} (n={len(runs)} seeds)")
            ax.fill_between(x, m - s, m + s, color=COLOR[algo], alpha=0.15, lw=0)
            ax.set_xlabel("Environment steps (millions)")
            ax.set_ylabel(ylab)
    axes[0].set_title("Training return (mean ± std over seeds)", color=INK, loc="left")
    axes[1].set_title("Training crash rate", color=INK, loc="left")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.fig, "training_curves.png"), dpi=160)
    plt.close(fig)


def aggregate(ev):
    by_algo = {}
    for name, m in ev["models"].items():
        by_algo.setdefault(m["algo"], []).append(m)
    table = {}
    for algo in [a for a in ORDER if a in by_algo]:
        ms = by_algo[algo]
        row = {}
        for k in ms[0]["main"]:
            v = np.array([x["main"][k] for x in ms], float)
            row[k] = (float(np.nanmean(v)), float(np.nanstd(v)), len(v))
        sweep = {}
        for s in ms[0]["sweep"]:
            sweep[s] = {k: (float(np.nanmean([x["sweep"][s][k] for x in ms])),
                            float(np.nanstd([x["sweep"][s][k] for x in ms]))) for k in ms[0]["sweep"][s]}
        table[algo] = {"main": row, "sweep": sweep}
    return table


def _pm(r, k, scale=1.0, fmt="{:.3f}"):
    return (fmt + " ± " + fmt).format(scale * r[k][0], scale * r[k][1])


def write_tables(table, ev):
    prot = ev["protocol"]
    lines = [
        f"Protocol: {prot['num_envs']} episodes/model, eval seed {prot['eval_seed']}, {prot['episode_s']:.0f} s episodes, "
        f"base wind U[{prot['train_wind_N'][0]:.2f}, {prot['train_wind_N'][1]:.2f}] N "
        f"(drone weight {prot['robot_weight_N']:.3f} N), gust pulse +{prot['gust_N']:.2f} N for {prot['gust_duration_s']:.1f} s at "
        f"{prot['gust_time_s']:.0f} s. Recovery band {100 * prot.get('recovery_tol_m', prot['tol_m']):.0f} cm held "
        f"{prot['hold_s']:.1f} s; success = no crash and last-2 s error < {100 * prot['tol_m']:.0f} cm. "
        "Mean ± std across 3 training seeds.",
        "",
        "| Algorithm | Position error (cm) | Gust peak error (cm) | Recovery time (s) | Success rate | Crash rate |",
        "|---|---|---|---|---|---|",
    ]
    csv = ["algo,seeds,pos_err_mean,pos_err_std,gust_peak_mean,gust_peak_std,recovery_mean,recovery_std,"
           "recovered_mean,success_mean,success_std,crash_mean"]
    for algo, t in table.items():
        r = t["main"]
        lines.append(
            f"| {LABEL[algo]} | {_pm(r, 'pos_err_m', 100, '{:.2f}')} | {_pm(r, 'gust_peak_err_m', 100, '{:.1f}')} | "
            f"{_pm(r, 'recovery_s', 1, '{:.2f}')} | {_pm(r, 'success_rate', 100, '{:.1f}')}% | "
            f"{100 * r['crash_rate'][0]:.1f}% |")
        csv.append(",".join([algo, str(r["pos_err_m"][2])] + [f"{r[k][i]:.4f}" for k in
                   ("pos_err_m", "gust_peak_err_m", "recovery_s") for i in (0, 1)] +
                   [f"{r['recovered_rate'][0]:.4f}", f"{r['success_rate'][0]:.4f}", f"{r['success_rate'][1]:.4f}",
                    f"{r['crash_rate'][0]:.4f}"]))
    # robustness beyond the training range
    sw = sorted(next(iter(table.values()))["sweep"], key=float)
    hi = [s for s in sw if float(s) > prot["train_wind_N"][1] + 1e-9]
    if hi:
        lines += ["", "Beyond the training range (base wind fixed, same gust pulse) — success rate / crash rate:", "",
                  "| Algorithm | " + " | ".join(f"{float(s):.2f} N ({100 * float(s) / prot['robot_weight_N']:.0f} % weight)" for s in hi) + " |",
                  "|---|" + "---|" * len(hi)]
        for algo, t in table.items():
            lines.append(f"| {LABEL[algo]} | " + " | ".join(
                f"{100 * t['sweep'][s]['success_rate'][0]:.1f}% / {100 * t['sweep'][s]['crash_rate'][0]:.1f}%" for s in hi) + " |")
    if "ppo" in table:
        pr = table["ppo"]["main"]
        lines.append("")
        for other in ("dqn", "td3"):
            if other in table:
                o = table[other]["main"]
                f = lambda k: 100 * (pr[k][0] / o[k][0] - 1)  # noqa: E731
                lines.append(f"- PPO relative to {LABEL[other]}: position error {f('pos_err_m'):+.0f}%, "
                             f"gust peak error {f('gust_peak_err_m'):+.0f}%, recovery time {f('recovery_s'):+.0f}% "
                             f"(negative = PPO better)")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(args.results, "summary.md"), "w") as fh:
        fh.write(md)
    with open(os.path.join(args.results, "summary.csv"), "w") as fh:
        fh.write("\n".join(csv) + "\n")
    print(md)


def wind_plot(table, ev):
    w = ev["protocol"]["robot_weight_N"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    specs = (("pos_err_m", "Position error in steady wind (cm)", 100.0),
             ("gust_peak_err_m", "Peak error after +0.10 N gust (cm)", 100.0),
             ("success_rate", "Success rate (%)", 100.0))
    for ax, (k, ylab, sc) in zip(axes, specs):
        for algo, t in table.items():
            xs = sorted(t["sweep"], key=float)
            x = np.array([float(s) for s in xs])
            m = np.array([t["sweep"][s][k][0] for s in xs]) * sc
            s = np.array([t["sweep"][s][k][1] for s in xs]) * sc
            ax.plot(x, m, color=COLOR[algo], ls=LS[algo], marker=MARK[algo], ms=8, mec="#fcfcfb", mew=2,
                    label=LABEL[algo])
            ax.fill_between(x, m - s, m + s, color=COLOR[algo], alpha=0.12, lw=0)
        lo, hi = ev["protocol"]["train_wind_N"]
        ax.axvspan(lo, hi, color=GRID, alpha=0.6, lw=0)
        ax.set_xlabel("Wind force (N)")
        ax.set_ylabel(ylab)
        sec = ax.secondary_xaxis("top", functions=(lambda f: f / w * 100, lambda p_: p_ * w / 100))
        sec.set_xlabel("% of drone weight", color=INK2)
    axes[0].legend(frameon=False)
    axes[0].text(0.5 * (lo + hi), axes[0].get_ylim()[1], "training range", ha="center", va="top", color=INK2, fontsize=8)
    fig.suptitle("Wind robustness sweep: base wind fixed per point, +0.10 N gust at 6 s (mean ± std over 3 seeds; error panels use surviving episodes)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.fig, "wind_adaptation.png"), dpi=160)
    plt.close(fig)


def main():
    logs = load_logs()
    if logs:
        training_curves(logs)
    ev_path = os.path.join(args.results, "eval_per_model.json")
    if os.path.exists(ev_path):
        ev = json.load(open(ev_path))
        table = aggregate(ev)
        write_tables(table, ev)
        wind_plot(table, ev)


if __name__ == "__main__":
    main()
