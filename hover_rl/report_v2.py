"""v2 report: tables + figures for the realistic environment (pure numpy/matplotlib, no Isaac Sim).

    python hover_rl/report_v2.py
Reads results_v2/eval_v2_ctbr.json, results_v2/eval_v2_motor.json, results_v2/wind_estimate.{json,npz},
runs_v2/*/train_log.csv. Writes results_v2/summary.md|csv and figures/v2_*.png.
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES, FIG, RUNS = (os.path.join(ROOT, d) for d in ("results_v2", "figures", "runs_v2"))
ORDER = ["pid_ctbr", "pid_motor", "lqr_motor", "lqi_motor", "ppo_motor", "ppo_ctbr", "td3_ctbr", "td3t_ctbr",
         "student_ctbr", "teacher_ctbr"]
LABEL = {"pid_ctbr": "Cascaded PID (CTBR, onboard rate loop)", "pid_motor": "Cascaded PID (motor commands)",
         "lqr_motor": "LQR (motor commands)", "lqi_motor": "LQI (motor commands)", "ppo_motor": "PPO, motor commands", "ppo_ctbr": "PPO, CTBR",
         "td3_ctbr": "TD3 (v1 hparams), CTBR", "td3t_ctbr": "TD3 (tuned), CTBR", "student_ctbr": "RMA student (CTBR, estimates wind)",
         "teacher_ctbr": "RMA teacher (sees true wind; upper bound)"}
SHORT = {"pid_ctbr": "PID-CTBR", "pid_motor": "PID-motor", "lqr_motor": "LQR", "lqi_motor": "LQI", "ppo_motor": "PPO-motor", "ppo_ctbr": "PPO-CTBR", "td3_ctbr": "TD3 (v1 hparams)", "td3t_ctbr": "TD3 (tuned)",
         "student_ctbr": "RMA student", "teacher_ctbr": "Teacher (privileged)"}
# Only the first three categorical slots (blue, orange, aqua) pass the ALL-pairs CVD check, so the line charts use
# exactly those three hues for the learned controllers and neutral inks for the two references: PID (dark ink) and the
# privileged teacher (light ink, dotted, upper bound). TD3 is reported in the tables only.
COLOR = {"ppo_ctbr": "#2a78d6", "ppo_motor": "#eb6834", "student_ctbr": "#1baf7a", "pid_ctbr": "#52514e",
         "teacher_ctbr": "#9a9993", "td3_ctbr": "#52514e", "td3t_ctbr": "#52514e", "pid_motor": "#52514e", "lqr_motor": "#52514e",
         "lqi_motor": "#52514e"}
MARK = {"ppo_ctbr": "o", "ppo_motor": "s", "student_ctbr": "^", "teacher_ctbr": "v", "pid_ctbr": "D", "td3_ctbr": "P", "td3t_ctbr": "X",
        "pid_motor": "D", "lqr_motor": "s", "lqi_motor": "v"}
LS = {"ppo_ctbr": "-", "ppo_motor": "--", "student_ctbr": "-", "teacher_ctbr": ":", "pid_ctbr": "-.", "td3_ctbr": ":", "td3t_ctbr": ":",
      "pid_motor": ":", "lqr_motor": "--", "lqi_motor": "-."}
PLOT = ["pid_ctbr", "ppo_motor", "ppo_ctbr", "student_ctbr", "teacher_ctbr"]
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "lines.linewidth": 2})


def load_eval():
    models, prot = {}, None
    for f in ("eval_v2_ctbr.json", "eval_v2_motor.json"):
        p = os.path.join(RES, f)
        if os.path.exists(p):
            d = json.load(open(p))
            prot = prot or d["protocol"]
            models.update(d["models"])
    return prot, models


def aggregate(models):
    by = {}
    for m in models.values():
        by.setdefault(m["algo"], []).append(m)
    table = {}
    for g in [g for g in ORDER if g in by]:
        ms = by[g]
        row = {k: (float(np.nanmean([x["main"][k] for x in ms])), float(np.nanstd([x["main"][k] for x in ms])), len(ms))
               for k in ms[0]["main"]}
        sweep = {s: {k: (float(np.nanmean([x["sweep"][s][k] for x in ms])), float(np.nanstd([x["sweep"][s][k] for x in ms])))
                     for k in ms[0]["sweep"][s]} for s in ms[0]["sweep"]}
        table[g] = {"main": row, "sweep": sweep}
    return table


def pm(r, k, sc, fmt):
    return (fmt + " ± " + fmt).format(sc * r[k][0], sc * r[k][1])


def rel(a, b):
    return 100.0 * (a / b - 1.0)


def write_tables(prot, table):
    w = prot["robot_weight_N"]
    tw = [x / w for x in prot["train_wind_N"]]
    L = [f"v2 protocol: {prot['num_envs']} episodes/model, eval seed {prot['eval_seed']}, 12 s episodes, base wind "
         f"U[{100 * tw[0]:.0f}, {100 * tw[1]:.0f}] % of the nominal weight ({w:.3f} N), gust +{100 * prot['gust_N'] / w:.0f} % "
         f"for {prot['gust_duration_s']:.1f} s at 6 s; motor lag 20-50 ms, 1-2 step (20-40 ms) delay, mass/thrust ±10 %, "
         "sensor noise; errors are TRUE position errors. Mean ± std over 3 seeds (RL: training seeds; PID: tuning seeds).",
         "", "| Controller | Position error (cm) | Gust peak error (cm) | Recovery time (s) | Recovered | Success | Crash |",
         "|---|---|---|---|---|---|---|"]
    csv = ["group,n,pos_err_cm,pos_err_std,gust_peak_cm,gust_peak_std,recovery_s,recovery_std,recovered,success,success_std,crash"]
    for g, t in table.items():
        r = t["main"]
        L.append(f"| {LABEL[g]} | {pm(r, 'pos_err_m', 100, '{:.2f}')} | {pm(r, 'gust_peak_err_m', 100, '{:.1f}')} | "
                 f"{pm(r, 'recovery_s', 1, '{:.2f}')} | {100 * r['recovered_rate'][0]:.1f}% | "
                 f"{pm(r, 'success_rate', 100, '{:.1f}')}% | {100 * r['crash_rate'][0]:.1f}% |")
        csv.append(",".join([g, str(r["pos_err_m"][2])] + [f"{100 * r[k][i]:.3f}" for k in ("pos_err_m", "gust_peak_err_m") for i in (0, 1)]
                            + [f"{r['recovery_s'][i]:.3f}" for i in (0, 1)]
                            + [f"{r['recovered_rate'][0]:.4f}", f"{r['success_rate'][0]:.4f}", f"{r['success_rate'][1]:.4f}",
                               f"{r['crash_rate'][0]:.4f}"]))
    sw = sorted(next(iter(table.values()))["sweep"], key=float)
    hi = [s for s in sw if float(s) > tw[1] + 1e-9]
    L += ["", "Success rate / crash rate with the base wind fixed beyond the training range (same +40 % gust):", "",
          "| Controller | " + " | ".join(f"{100 * float(s):.0f} % wt" for s in hi) + " |", "|---|" + "---|" * len(hi)]
    for g, t in table.items():
        L.append(f"| {SHORT[g]} | " + " | ".join(f"{100 * t['sweep'][s]['success_rate'][0]:.1f}% / "
                                                  f"{100 * t['sweep'][s]['crash_rate'][0]:.1f}%" for s in hi) + " |")
    L.append("")

    def cmp(a, b):
        if a in table and b in table:
            ra, rb = table[a]["main"], table[b]["main"]
            L.append(f"- {SHORT[a]} vs {SHORT[b]}: position error {rel(ra['pos_err_m'][0], rb['pos_err_m'][0]):+.0f}%, "
                     f"gust peak {rel(ra['gust_peak_err_m'][0], rb['gust_peak_err_m'][0]):+.0f}%, recovery time "
                     f"{rel(ra['recovery_s'][0], rb['recovery_s'][0]):+.0f}% (negative = {SHORT[a]} better); success beyond "
                     "range " + ", ".join(f"{100 * float(s):.0f}%: {100 * table[a]['sweep'][s]['success_rate'][0]:.1f} vs "
                                          f"{100 * table[b]['sweep'][s]['success_rate'][0]:.1f}" for s in hi))
    for a, b in (("ppo_ctbr", "ppo_motor"), ("ppo_ctbr", "pid_ctbr"), ("student_ctbr", "ppo_ctbr"),
                 ("student_ctbr", "pid_ctbr"), ("teacher_ctbr", "student_ctbr"), ("td3_ctbr", "ppo_ctbr"), ("td3t_ctbr", "ppo_ctbr"), ("td3t_ctbr", "td3_ctbr"),
                 ("lqr_motor", "pid_motor"), ("lqi_motor", "pid_motor"), ("lqi_motor", "lqr_motor"),
                 ("pid_ctbr", "pid_motor"), ("ppo_motor", "pid_motor"), ("ppo_motor", "lqi_motor")):
        cmp(a, b)
    we = os.path.join(RES, "wind_estimate.json")
    if os.path.exists(we):
        d = json.load(open(we))["summary"]
        L += ["", "RMA wind estimate vs true wind (horizontal+vertical force, % of weight; mean over students):", "",
              "| Base wind | RMSE all (t ≥ 1 s) | RMSE steady (2-6 s) | RMSE during gust | magnitude bias (steady) | corr (x,y) |",
              "|---|---|---|---|---|---|"]
        for w_ in sorted({k.split("@")[1] for k in d}, key=float):
            v = [x for k, x in d.items() if k.endswith("@" + w_)]
            f = lambda k: 100 * np.mean([x[k] for x in v])  # noqa: E731
            L.append(f"| {100 * float(w_):.0f} % wt | {f('rmse_all'):.1f} | {f('rmse_steady'):.1f} | {f('rmse_gust'):.1f} | "
                     f"{f('magnitude_bias_steady'):+.1f} | {np.mean([x['corr_xy'] for x in v]):.3f} |")
    md = "\n".join(L) + "\n"
    open(os.path.join(RES, "summary.md"), "w").write(md)
    open(os.path.join(RES, "summary.csv"), "w").write("\n".join(csv) + "\n")
    print(md)


def sweep_plot(prot, table):
    w = prot["robot_weight_N"]
    tw = [x / w for x in prot["train_wind_N"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    specs = (("pos_err_m", "Position error before the gust (cm)", 100.0),
             ("gust_peak_err_m", "Peak error after the +40 % gust (cm)", 100.0),
             ("success_rate", "Success rate (%)", 100.0))
    for ax, (k, ylab, sc) in zip(axes, specs):
        for g, t in ((g, table[g]) for g in PLOT if g in table):
            xs = sorted(t["sweep"], key=float)
            x = np.array([100 * float(s) for s in xs])
            m = np.array([t["sweep"][s][k][0] for s in xs]) * sc
            sd = np.array([t["sweep"][s][k][1] for s in xs]) * sc
            ax.plot(x, m, color=COLOR[g], ls=LS[g], marker=MARK[g], ms=8, mec=BG, mew=2, label=SHORT[g])
            ax.fill_between(x, m - sd, m + sd, color=COLOR[g], alpha=0.12, lw=0)
        ax.axvspan(100 * tw[0], 100 * tw[1], color=GRID, alpha=0.6, lw=0)
        ax.set_xlabel("Base wind force (% of drone weight)")
        ax.set_ylabel(ylab)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].legend(frameon=False, loc="lower left")
    axes[2].text(100 * np.mean(tw), 50, "training\nrange", ha="center", va="center", color=INK2, fontsize=8)
    fig.suptitle("v2 (motor lag, delay, randomised mass/thrust, sensor noise): wind sweep with a +40 % gust at 6 s. "
                 "Mean ± std over 3 seeds; error panels use surviving episodes", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_wind_sweep.png"), dpi=160)
    plt.close(fig)


def classical_plot(prot, table):
    """figures/classical_vs_rl.png: row 1 = v1 (ideal actuators), row 2 = v2 (realistic). Classical controllers in
    neutral ink told apart by marker + line style; RL in the categorical hues."""
    p1 = os.path.join(ROOT, "results", "eval_per_model.json")
    if not os.path.exists(p1):
        return
    ev1 = json.load(open(p1))
    t1 = {}
    for m in ev1["models"].values():
        t1.setdefault(m["algo"], []).append(m)
    w1 = ev1["protocol"]["robot_weight_N"]
    rows = [("v1: ideal motors, no delay, RPM 4500-5500 (0.81-1.21 x weight)", w1, ev1["protocol"]["train_wind_N"],
             [("ppo", "PPO", "#2a78d6", "o", "-"), ("pid", "Cascaded PID", INK2, "D", ":"),
              ("lqr", "LQR", INK2, "s", "--"), ("lqi", "LQI", INK2, "v", "-.")], t1, 1.0),
            ("v2: motor lag 20-50 ms, 20-40 ms delay, mass/thrust ±10 %, sensor noise, thrust/weight 1.9",
             prot["robot_weight_N"], prot["train_wind_N"],
             [("ppo_ctbr", "PPO-CTBR", "#2a78d6", "o", "-"), ("ppo_motor", "PPO-motor", "#eb6834", "s", "--"),
              ("pid_motor", "PID (motor)", INK2, "D", ":"), ("lqr_motor", "LQR (motor)", INK2, "s", "--"),
              ("lqi_motor", "LQI (motor)", INK2, "v", "-.")], None, prot["robot_weight_N"])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    for ri, (title, w, trw, series, tv1, unit) in enumerate(rows):
        for ci, (k, ylab, sc) in enumerate((("pos_err_m", "Position error before gust (cm)", 100.0),
                                            ("gust_peak_err_m", "Peak error after gust (cm)", 100.0),
                                            ("success_rate", "Success rate (%)", 100.0))):
            ax = axes[ri, ci]
            for g, lab, col, mk, ls in series:
                if tv1 is not None:
                    if g not in tv1:
                        continue
                    ms = tv1[g]
                    xs = sorted(ms[0]["sweep"], key=float)
                    vals = np.array([[x["sweep"][s][k] for s in xs] for x in ms])
                    xx = np.array([100 * float(s) / w for s in xs])
                else:
                    if g not in table:
                        continue
                    xs = sorted(table[g]["sweep"], key=float)
                    vals = None
                    xx = np.array([100 * float(s) for s in xs])
                if vals is not None:
                    m, sd = np.nanmean(vals, 0) * sc, np.nanstd(vals, 0) * sc
                else:
                    m = np.array([table[g]["sweep"][s][k][0] for s in xs]) * sc
                    sd = np.array([table[g]["sweep"][s][k][1] for s in xs]) * sc
                ax.plot(xx, m, color=col, ls=ls, marker=mk, ms=7, mec=BG, mew=1.5, label=lab)
                ax.fill_between(xx, m - sd, m + sd, color=col, alpha=0.10, lw=0)
            lo, hi = (100 * x / w for x in trw)
            ax.axvspan(lo, hi, color=GRID, alpha=0.6, lw=0)
            ax.set_ylabel(ylab)
            if k != "success_rate":
                ax.set_yscale("log")
            if ri == 1:
                ax.set_xlabel("Base wind force (% of drone weight)")
            if ci == 0:
                ax.set_title(title, loc="left", color=INK, fontsize=10)
        axes[ri, 2].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle("Classical control vs RL: wind sweep with gust pulse at 6 s (grey band = training wind range; "
                 "mean ± std over 3 seeds; error panels use surviving episodes)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "classical_vs_rl.png"), dpi=160)
    plt.close(fig)


def training_curves():
    groups = {}
    for f in sorted(glob.glob(os.path.join(RUNS, "*_s*", "train_log.csv"))):
        g = os.path.basename(os.path.dirname(f)).rsplit("_s", 1)[0]
        if g.startswith("student"):
            continue
        groups.setdefault(g, []).append(np.genfromtxt(f, delimiter=",", names=True, usecols=(0, 1, 2, 3, 4), dtype=float))
    if not groups:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for g in [g for g in PLOT if g in groups]:
        runs = groups[g]
        n = min(len(r) for r in runs)
        x = runs[0]["env_steps"][:n] / 1e6
        for ax, key in ((axes[0], "mean_return"), (axes[1], "mean_pos_err")):
            Y = np.stack([np.convolve(np.nan_to_num(r[key][:n]), np.ones(10) / 10, mode="same") for r in runs])
            m, s = Y.mean(0), Y.std(0)
            ax.plot(x[5:-5], m[5:-5], color=COLOR[g], ls=LS[g], label=f"{SHORT[g]} (n={len(runs)})")
            ax.fill_between(x[5:-5], (m - s)[5:-5], (m + s)[5:-5], color=COLOR[g], alpha=0.15, lw=0)
            ax.set_xlabel("Environment steps (millions)")
    axes[0].set_ylabel("Episode return (10 s)")
    axes[1].set_ylabel("Mean position error during training (m)")
    axes[1].set_yscale("log")
    axes[0].set_title("v2 training return (mean ± std over seeds)", loc="left", color=INK)
    axes[1].set_title("Training position error (training wind + random gusts)", loc="left", color=INK)
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_training_curves.png"), dpi=160)
    plt.close(fig)


def wind_estimate_plot():
    p = os.path.join(RES, "wind_estimate.npz")
    if not os.path.exists(p):
        return
    d = np.load(p)
    meta = json.load(open(os.path.join(RES, "wind_estimate.json")))
    t = d["t"]
    cases = meta["cases"]
    fig, axes = plt.subplots(2, len(cases), figsize=(6.2 * len(cases), 7), sharex=True)
    for ci, w in enumerate(cases):
        ax = axes[0, ci]
        tru, est = 100 * d[f"s0_w{w}_true"], 100 * d[f"s0_w{w}_est"]
        for j, (axn, col) in enumerate((("x", "#2a78d6"), ("y", "#eb6834"))):
            ax.plot(t, tru[:, j], color=col, lw=2, label=f"true F{axn}")
            ax.plot(t, est[:, j], color=col, lw=1.2, ls="--", label=f"estimated F{axn}")
        ax.axvspan(6.0, 6.5, color=GRID, alpha=0.8, lw=0)
        ax.set_ylabel("Wind force (% of weight), world frame")
        inside = "inside" if w <= 0.35 + 1e-9 else "OUTSIDE"
        ax.set_title(f"One episode, base wind {100 * w:.0f} % ({inside} training range), gust at 6 s", loc="left",
                     color=INK, fontsize=10)
        if ci == 0:
            ax.legend(frameon=False, ncol=2, fontsize=8)
        ax2 = axes[1, ci]
        keys = sorted(k for k in d.files if k.endswith(f"_w{w}_rmse_t"))
        R = np.stack([100 * d[k] for k in keys])
        ax2.plot(t, R.mean(0), color="#1baf7a", lw=2, label=f"RMSE over 512 episodes (mean of {len(keys)} students)")
        ax2.fill_between(t, R.min(0), R.max(0), color="#1baf7a", alpha=0.15, lw=0)
        ax2.axvspan(6.0, 6.5, color=GRID, alpha=0.8, lw=0)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Estimation error |F̂ - F| (% of weight)")
        ax2.legend(frameon=False, fontsize=8)
    fig.suptitle("RMA adaptation module: online wind-force estimate from 1 s of state/action history", color=INK,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "v2_wind_estimate.png"), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    training_curves()
    prot, models = load_eval()
    if models:
        table = aggregate(models)
        write_tables(prot, table)
        sweep_plot(prot, table)
        classical_plot(prot, table)
    wind_estimate_plot()


if __name__ == "__main__":
    main()
