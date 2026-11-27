"""Publication figures (F1-F8) and final tables from the saved results (pure numpy/matplotlib, no Isaac Sim).

    python hover_rl/paper_figs.py            # -> figures/paper/F*.{png,svg,pdf}, results_v2/paper_tables.md

Inputs: results/eval_per_model.json (v1), results_v2/eval_v2_{ctbr,motor}.json (v2), results_v2/time_response_{ctbr,motor}.npz,
results_v2/wind_estimate.{json,npz}, runs_v2/*/train_log.csv, results_v2/td3_tuning/ (sweep evaluation + diagnosis),
runs_v2_td3sweep/*/{config.json,train_log.csv}.

Style (identical in every figure): one identity per controller = colour of its family + marker / line style / hatch of
the variant. Families: classical (PID dark ink; LQR/LQI mid ink), PPO aqua, TD3 orange, RMA blue. Only those three
hues are used because they are the only categorical slots that stay distinguishable for all pairs under colour-vision
deficiency (validated); variants are separated by marker, dash and hatch, never by colour alone. Sans-serif font,
mean +- std over 3 seeds (RL) / 3 independent tunings (classical), panel labels (a)(b)(c), no in-image titles.
PNG at 200 dpi (2x of the 100-dpi layout; 2000-2200 px wide) + SVG + PDF.
"""

import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES1, RES2, RUNS2 = (os.path.join(ROOT, d) for d in ("results", "results_v2", "runs_v2"))
SWEEP_RUNS = os.path.join(ROOT, "runs_v2_td3sweep")
TUNE = os.path.join(RES2, "td3_tuning")
OUT = os.path.join(ROOT, "figures", "paper")

INK, INK2, INK3, GRID, BG = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# controller -> (label, colour, marker, linestyle, hatch, filled)
STYLE = {
    "pid_ctbr": ("PID (CTBR)", INK2, "D", "-", "", True),
    "pid_motor": ("PID (motor)", INK2, "D", "--", "////", False),
    "lqr_motor": ("LQR", INK3, "s", ":", "....", False),
    "lqi_motor": ("LQI", INK3, "v", "-.", "xxxx", True),
    "ppo_motor": ("PPO (motor)", AQUA, "s", "--", "////", False),
    "ppo_ctbr": ("PPO (CTBR)", AQUA, "o", "-", "", True),
    "td3_ctbr": ("TD3 (v1 hparams)", ORANGE, "X", "--", "////", False),
    "td3t_ctbr": ("TD3", ORANGE, "P", "-", "", True),
    "student_ctbr": ("RMA (student)", BLUE, "^", "-", "", True),
    "teacher_ctbr": ("RMA teacher (privileged)", BLUE, "v", ":", "....", False),
}
# The paper shows the final (tuned) TD3 only; the default-hyper-parameter TD3 appears in the tuning study (F8, Table 4).
V2_ORDER = ["pid_ctbr", "pid_motor", "lqr_motor", "lqi_motor", "ppo_motor", "ppo_ctbr", "td3t_ctbr",
            "student_ctbr", "teacher_ctbr"]
V1_MAP = {"pid": "pid_motor", "lqr": "lqr_motor", "lqi": "lqi_motor", "ppo": "ppo_motor", "td3": "td3_ctbr"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Liberation Sans", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "lines.linewidth": 2, "lines.markersize": 6.5, "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    "hatch.linewidth": 0.8,
})


def plain_log(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext, kw in (("png", {"dpi": 200}), ("svg", {}), ("pdf", {})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.05, **kw)
    plt.close(fig)
    print("wrote", os.path.join("figures", "paper", name + ".{png,svg,pdf}"))


def panel(ax, k, x=-0.12, y=1.04):
    ax.text(x, y, f"({'abcdefgh'[k]})", transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK,
            va="bottom", ha="left")


def line_kw(g, **kw):
    lab, col, mk, ls, _, filled = STYLE[g]
    return dict(color=col, marker=mk, ls=ls, label=lab, mfc=col if filled else BG, mec=col, mew=1.4, **kw)


def legend_handles(groups):
    return [Line2D([], [], **line_kw(g)) for g in groups]


# ---------------------------------------------------------------- data
def load_v2():
    models = {}
    for f in ("eval_v2_ctbr.json", "eval_v2_motor.json"):
        p = os.path.join(RES2, f)
        if os.path.exists(p):
            models.update(json.load(open(p))["models"])
    return models


def load_v1():
    d = json.load(open(os.path.join(RES1, "eval_per_model.json")))["models"]
    return {k: {**v, "algo": V1_MAP.get(v["algo"], v["algo"])} for k, v in d.items() if v["algo"] in V1_MAP}


def by_group(models):
    out = {}
    for m in models.values():
        out.setdefault(m["algo"], []).append(m)
    return out


def stat(ms, key, sweep=None):
    vals = np.array([(m["sweep"][sweep] if sweep else m["main"])[key] for m in ms], dtype=float)
    return np.nanmean(vals), np.nanstd(vals), vals


def read_log(path):
    rows = list(csv.DictReader(open(path)))
    x = np.array([float(r["env_steps"]) for r in rows])
    cols = {k: np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows]) for k in rows[0]
            if k not in ("extra",)}
    return x, cols


def smooth(y, k=25):
    y = np.asarray(y, float)
    m = np.isfinite(y)
    if m.sum() < k:
        return y
    yy = np.interp(np.arange(len(y)), np.flatnonzero(m), y[m])
    return np.convolve(np.pad(yy, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def curves(group, key, root=RUNS2):
    xs, ys = None, []
    for p in sorted(glob.glob(os.path.join(root, f"{group}_s*", "train_log.csv"))):
        x, c = read_log(p)
        y = smooth(c[key])
        if xs is None or len(x) < len(xs):
            xs = x
        ys.append(y)
    if not ys:
        return None, None
    n = len(xs)
    return xs, np.stack([y[:n] for y in ys])


# ---------------------------------------------------------------- F1 diagram
def box(ax, x, y, w, h, text, fc=BG, ec=INK2, lw=1.2, fs=8.6, bold=False, color=INK, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p, q, color=INK2, ls="-", lw=1.2, rad=0.0, text=None, tpos=0.5, toff=(0, 0.012), fs=7.8):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))
    if text:
        ax.text(p[0] + (q[0] - p[0]) * tpos + toff[0], p[1] + (q[1] - p[1]) * tpos + toff[1], text, fontsize=fs,
                color=INK2, ha="center", va="bottom")


def tint(c, a):
    c = np.array(matplotlib.colors.to_rgb(c))
    return tuple(1 - a * (1 - c))


def fig1():
    fig = plt.figure(figsize=(11, 5.6))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1), ax.axis("off")
    # (a) simulator
    ax.text(0.01, 0.965, "(a)", fontsize=11, fontweight="bold", color=INK)
    ax.text(0.05, 0.965, "v2 simulator (Isaac Lab, physics 100 Hz, control 50 Hz)", fontsize=9.5, color=INK2)
    ax.add_patch(FancyBboxPatch((0.01, 0.53), 0.98, 0.405, boxstyle="round,pad=0,rounding_size=0.01", fc="#f6f6f4",
                                ec=GRID, lw=1))
    box(ax, 0.03, 0.74, 0.13, 0.14, "Command\nqueue\n1-2 steps (20-40 ms)")
    box(ax, 0.20, 0.74, 0.15, 0.14, "Onboard rate loop\n100 Hz P, mixer\n(CTBR only)", ls="--")
    box(ax, 0.39, 0.74, 0.14, 0.14, "Motors\n1st-order lag\nτ ~ U[20, 50] ms\nT/W = 1.9")
    box(ax, 0.57, 0.70, 0.17, 0.22, "Rigid body (28 g)\nmass, inertia ×U[0.9, 1.1]\nthrust coeff. ×U[0.9, 1.1]",
        fc=BG, ec=INK, lw=1.5, bold=False)
    box(ax, 0.78, 0.74, 0.19, 0.14, "Sensors\n+ noise: 5 mm, 2 cm/s,\n0.01 rad, 0.05 rad/s")
    box(ax, 0.57, 0.555, 0.17, 0.10, "Wind: base 0-35 % W\n+ gusts 10-40 % W", fc=tint(ORANGE, 0.12), ec=ORANGE)
    arrow(ax, (0.16, 0.81), (0.20, 0.81))
    arrow(ax, (0.35, 0.81), (0.39, 0.81))
    arrow(ax, (0.53, 0.81), (0.57, 0.81))
    arrow(ax, (0.74, 0.81), (0.78, 0.81))
    arrow(ax, (0.655, 0.655), (0.655, 0.70), color=ORANGE, text="force", tpos=0.4, toff=(0.025, -0.005))
    arrow(ax, (0.065, 0.74), (0.46, 0.74), ls="--", color=INK3, rad=0.22)
    ax.text(0.30, 0.632, "motor-command policies bypass the rate loop", fontsize=7.6, color=INK3, ha="center")
    arrow(ax, (0.875, 0.74), (0.875, 0.435), color=INK, lw=1.4, text="o_t", tpos=0.75, toff=(0.015, 0))
    arrow(ax, (0.12, 0.435), (0.12, 0.74), color=INK, lw=1.4, text="a_t", tpos=0.45, toff=(0.016, 0))
    ax.text(0.20, 0.603, "observation o_t (26-D): body-frame state, position error,\nlast 2 actions; teacher also gets "
            "7-D privileged e_t\n(wind, mass, thrust coeff., motor lag, delay)", fontsize=7.6, color=INK2, va="top")
    # (b) controller stack
    ax.text(0.01, 0.475, "(b)", fontsize=11, fontweight="bold", color=INK)
    ax.text(0.14, 0.475, "Controllers (all output a_t to the command queue)", fontsize=9.5, color=INK2)
    y0, h = 0.04, 0.39
    cols = [(0.01, 0.225, INK2, "Cascaded PID", "position PID → thrust vector\n(tilt ≤ 40°)\n→ attitude P → rate P\n→ mixer\n\n8 gains, CEM-tuned\non seeds 11-13"),
            (0.25, 0.225, INK3, "LQR / LQI", "11-state hover model (+3\nintegral states for LQI)\nZOH 20 ms, DARE\nu = −K x, error clip\n\nQ, R CEM-tuned\n(Bryson's rule)"),
            (0.49, 0.235, AQUA, "PPO / TD3 policy", "MLP 256-256\no_t → a_t\na_t = collective thrust +\nbody rates (CTBR)\nor 4 motor commands\n\n100 M steps × 3 seeds"),
            (0.74, 0.25, BLUE, "RMA (teacher-student)", "")]
    for x, w, c, title, body in cols:
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.01", fc=tint(c, 0.08), ec=c, lw=1.3))
        ax.text(x + 0.012, y0 + h - 0.035, title, fontsize=9.5, fontweight="bold", color=INK)
        if body:
            ax.text(x + 0.012, y0 + h - 0.07, body, fontsize=8, color=INK2, va="top", linespacing=1.4)
    ax.text(0.623, y0 + 0.012, "TD3: same actor, off-policy, 3-step returns", fontsize=7.2, color=INK3, ha="center")
    # RMA internals
    x0 = 0.75
    box(ax, x0 + 0.005, 0.29, 0.095, 0.07, "history\no_{t-49..t}", fs=7.8)
    box(ax, x0 + 0.12, 0.29, 0.11, 0.07, "adaptation\nmodule ψ", fs=7.8, ec=BLUE, lw=1.4)
    box(ax, x0 + 0.12, 0.175, 0.11, 0.07, "policy π\n(o_t, ê_t) → a_t", fs=7.8, ec=BLUE, lw=1.4)
    box(ax, x0 + 0.005, 0.175, 0.095, 0.07, "true e_t\n(teacher only)", fs=7.8, ls=":", ec=INK3)
    arrow(ax, (x0 + 0.10, 0.325), (x0 + 0.12, 0.325))
    arrow(ax, (x0 + 0.175, 0.29), (x0 + 0.175, 0.245), text="ê_t", tpos=0.5, toff=(0.015, -0.01))
    arrow(ax, (x0 + 0.10, 0.21), (x0 + 0.12, 0.21), ls=":", color=INK3)
    ax.text(x0 + 0.005, 0.145, "Phase 1: teacher π(o, e) with PPO.\nPhase 2: ψ regresses e from history\n"
            "(MSE, 30 M steps); π frozen.", fontsize=7.4, color=INK2, va="top", linespacing=1.35)
    save(fig, "F1_system_diagram")


# ---------------------------------------------------------------- F2 in-range bars
def fig2(v2g):
    groups = [g for g in V2_ORDER if g in v2g]
    metrics = [("pos_err_m", 100, "Steady position error (cm)"), ("gust_peak_err_m", 100, "Gust peak error (cm)"),
               ("recovery_s", 1, "Recovery time (s)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
    y = np.arange(len(groups))[::-1]
    for k, (ax, (key, sc, xl)) in enumerate(zip(axes, metrics)):
        for yi, g in zip(y, groups):
            lab, col, mk, ls, hatch, filled = STYLE[g]
            mu, sd, vals = stat(v2g[g], key)
            ax.barh(yi, sc * mu, height=0.68, color=col if filled else tint(col, 0.25), edgecolor=col, lw=1.0,
                    hatch=hatch)
            ax.errorbar(sc * mu, yi, xerr=sc * sd, fmt="none", ecolor=INK, elinewidth=1.1, capsize=3)
            ax.plot(sc * vals, yi + np.array([-0.18, 0, 0.18])[:len(vals)], "o", ms=3, color=INK, mfc=BG, mew=0.9,
                    zorder=5)
            if key == "recovery_s":
                rr = 100 * stat(v2g[g], "recovered_rate")[0]
                if rr < 90:
                    ax.text(sc * max(mu + sd, np.nanmax(vals)) + 0.06, yi, f"{rr:.0f} % recover", va="center", fontsize=7.6, color=INK2)
        ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(left=0)
        panel(ax, k, x=-0.02 if k else -0.42)
    axes[0].set_yticks(y, [STYLE[g][0] for g in groups])
    axes[0].tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=1.5)
    save(fig, "F2_v2_in_range")


# ---------------------------------------------------------------- F3 wind sweep
def fig3(v2g):
    groups = [g for g in V2_ORDER if g in v2g]
    sweep = sorted(v2g[groups[0]][0]["sweep"], key=float)
    xs = 100 * np.array([float(s) for s in sweep])
    metrics = [("success_rate", 100, "Success rate (%)", False), ("pos_err_m", 100, "Steady position error (cm)", True),
               ("gust_peak_err_m", 100, "Gust peak error (cm)", True)]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    for k, (ax, (key, sc, yl, logy)) in enumerate(zip(axes, metrics)):
        ax.axvspan(0, 35, color="#f1f1ee", lw=0, zorder=0)
        for g in groups:
            mu = np.array([stat(v2g[g], key, s)[0] for s in sweep]) * sc
            sd = np.array([stat(v2g[g], key, s)[1] for s in sweep]) * sc
            ax.plot(xs, mu, ms=4.5, lw=1.6, **line_kw(g))
            ax.fill_between(xs, mu - sd, mu + sd, color=STYLE[g][1], alpha=0.12, lw=0)
        ax.set_xlabel("Steady wind (% of weight), + 40 % gust")
        ax.set_ylabel(yl)
        ax.set_xticks(xs[::2])
        if logy:
            ax.set_yscale("log")
            plain_log(ax)
        else:
            ax.set_ylim(-3, 103)
        panel(ax, k, x=-0.2)
    for ax in axes:
        ax.text(17.5, 1.0, "training range", ha="center", va="bottom", fontsize=7.8, color=INK3,
                transform=ax.get_xaxis_transform())
    fig.legend(handles=legend_handles(groups), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.13), fontsize=8.6,
               handlelength=2.6, columnspacing=1.4)
    fig.tight_layout(w_pad=2)
    save(fig, "F3_v2_wind_sweep")


# ---------------------------------------------------------------- F4 sim gap
def fig4(v1g, v2g):
    pairs = [("pid_motor", "pid_motor", "PID"), ("lqr_motor", "lqr_motor", "LQR"), ("lqi_motor", "lqi_motor", "LQI"),
             ("ppo_motor", "ppo_motor", "PPO (motor)"), ("ppo_motor", "ppo_ctbr", "PPO (motor → CTBR)"),
             ("td3_ctbr", "td3t_ctbr", "TD3 (motor → CTBR)")]
    pairs = [p for p in pairs if p[0] in v1g and p[1] in v2g]
    metrics = [("pos_err_m", 100, "Steady position error (cm)", True), ("gust_peak_err_m", 100, "Gust peak error (cm)", False),
               ("success_rate", 100, "In-range success (%)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    for k, (ax, (key, sc, yl, logy)) in enumerate(zip(axes, metrics)):
        for g1, g2, lab in pairs:
            a, sa, _ = stat(v1g[g1], key)
            b, sb, _ = stat(v2g[g2], key)
            kw = line_kw(g2)
            kw["label"] = lab
            ax.errorbar([0, 1], [sc * a, sc * b], yerr=[sc * sa, sc * sb], capsize=3, elinewidth=1, lw=1.6, ms=6, **kw)
        ax.set_xticks([0, 1], ["v1 idealised", "v2 realistic"])
        ax.set_xlim(-0.25, 1.25)
        ax.grid(axis="x", visible=False)
        ax.set_ylabel(yl)
        if logy:
            ax.set_yscale("log")
            plain_log(ax)
        panel(ax, k, x=-0.2)
    h = []
    for g1, g2, lab in pairs:
        kw = line_kw(g2)
        kw["label"] = lab
        h.append(Line2D([], [], **kw))
    fig.legend(handles=h, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.09), fontsize=8.6, handlelength=2.6)
    fig.tight_layout(w_pad=2)
    save(fig, "F4_sim_gap_v1_v2")


# ---------------------------------------------------------------- F5 time response
def fig5():
    paths = [os.path.join(RES2, f"time_response_{m}.npz") for m in ("ctbr", "motor")]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("skip F5 (no time_response_*.npz)")
        return
    d = {}
    for p in paths:
        d.update(dict(np.load(p)))
    t = d["t"]
    keys_all = list(d)
    groups = [g for g in ["pid_ctbr", "lqi_motor", "ppo_ctbr", "td3t_ctbr", "student_ctbr"]
              if any(k.startswith(g + "_s") for k in keys_all)]
    # panel (a): one episode copy (same start, wind direction and airframe for all) that no controller crashes in
    alive = np.all([np.isfinite(d[k]).all(0) for g in groups for k in d if k.startswith(g + "_s")], 0)
    ix = int(np.argmax(alive))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.35, 1]})
    for g in groups:
        keys = sorted(k for k in keys_all if k.startswith(g + "_s"))
        one = 100 * d[keys[0]][:, ix]
        axes[0].plot(t, one, lw=1.5, **{**line_kw(g), "marker": None})
        E = 100 * np.concatenate([d[k] for k in keys], 1)
        med = np.nanmedian(E, 1)
        q1, q3 = np.nanpercentile(E, 25, 1), np.nanpercentile(E, 75, 1)
        axes[1].plot(t, med, lw=1.7, **{**line_kw(g), "marker": None})
        axes[1].fill_between(t, q1, q3, color=STYLE[g][1], alpha=0.13, lw=0)
    for k, ax in enumerate(axes):
        ax.axvspan(6.0, 6.5, color="#f1e3da", lw=0, zorder=0)
        ax.axhline(5, color=INK3, lw=0.8, ls=(0, (2, 2)))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position error (cm)")
        panel(ax, k, x=-0.13)
    axes[0].set_xlim(0, 12)
    axes[0].text(5.9, axes[0].get_ylim()[1] * 0.97, "gust +40 % W", fontsize=7.8, color=INK2, va="top", ha="right")
    axes[0].text(11.9, 5.4, "5 cm", fontsize=7.5, color=INK3, ha="right", va="bottom")
    axes[1].set_xlim(5.5, 9.0)
    fig.legend(handles=[Line2D([], [], **{**line_kw(g), "marker": None}) for g in groups], loc="lower center",
               ncol=len(groups), bbox_to_anchor=(0.5, -0.09), fontsize=8.6, handlelength=2.8)
    fig.tight_layout(w_pad=2)
    save(fig, "F5_time_response")


# ---------------------------------------------------------------- F6 wind estimate
def fig6():
    d = np.load(os.path.join(RES2, "wind_estimate.npz"))
    meta = json.load(open(os.path.join(RES2, "wind_estimate.json")))
    t = d["t"]
    cases = meta["cases"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    for ci, w in enumerate(cases):
        ax = axes[ci]
        tru, est = 100 * d[f"s0_w{w}_true"], 100 * d[f"s0_w{w}_est"]
        for j, (axn, ls) in enumerate((("x", "-"), ("y", "--"))):
            ax.plot(t, tru[:, j], color=INK2, lw=1.6, ls=ls, label=f"true $F_{axn}$")
            ax.plot(t, est[:, j], color=BLUE, lw=1.3, ls=ls, label=f"estimated $\\hat F_{axn}$")
        ax.axvspan(6.0, 6.5, color="#f1e3da", lw=0, zorder=0)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Wind force (% of weight)")
        ax.text(0.02, 0.97, f"base wind {100 * w:.0f} % W" + (" (outside training range)" if w > 0.35 else ""),
                transform=ax.transAxes, fontsize=8, color=INK2, va="top")
        panel(ax, ci, x=-0.18)
    ax = axes[2]
    for w, ls in zip(cases, ("-", "--")):
        keys = sorted(k for k in d.files if k.endswith(f"_w{w}_rmse_t"))
        R = np.stack([100 * d[k] for k in keys])
        ax.plot(t, R.mean(0), color=BLUE, ls=ls, lw=1.6, label=f"base {100 * w:.0f} % W")
        ax.fill_between(t, R.mean(0) - R.std(0), R.mean(0) + R.std(0), color=BLUE, alpha=0.13, lw=0)
    ax.axvspan(6.0, 6.5, color="#f1e3da", lw=0, zorder=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Estimate RMSE (% of weight)")
    ax.legend(fontsize=8)
    panel(ax, 2, x=-0.18)
    h = [Line2D([], [], color=INK2, lw=1.6, label="true $F_x$"), Line2D([], [], color=BLUE, lw=1.3, label="estimated $\\hat F_x$"),
         Line2D([], [], color=INK2, lw=1.6, ls="--", label="true $F_y$"), Line2D([], [], color=BLUE, lw=1.3, ls="--", label="estimated $\\hat F_y$")]
    fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.36, -0.08), fontsize=8.6, handlelength=2.6)
    fig.tight_layout(w_pad=2)
    save(fig, "F6_rma_wind_estimate")


# ---------------------------------------------------------------- F7 training curves
def fig7():
    groups = [g for g in ["ppo_motor", "ppo_ctbr", "td3t_ctbr", "teacher_ctbr"]
              if glob.glob(os.path.join(RUNS2, f"{g}_s*", "train_log.csv"))]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    for k, (key, yl, sc) in enumerate((("mean_return", "Episode return", 1), ("mean_pos_err", "Training position error (cm)", 100))):
        ax = axes[k]
        for g in groups:
            x, Y = curves(g, key)
            mu, sd = sc * np.nanmean(Y, 0), sc * np.nanstd(Y, 0)
            ax.plot(x / 1e6, mu, lw=1.6, **{**line_kw(g), "marker": None})
            ax.fill_between(x / 1e6, mu - sd, mu + sd, color=STYLE[g][1], alpha=0.13, lw=0)
        ax.set_xlabel("Environment steps (M)")
        ax.set_ylabel(yl)
        if key == "mean_pos_err":
            ax.set_yscale("log")
            plain_log(ax)
        panel(ax, k, x=-0.18)
    ax = axes[2]
    x, Y = curves("student_ctbr", "wind_rmse_frac_w")
    if x is not None:
        mu, sd = 100 * np.nanmean(Y, 0), 100 * np.nanstd(Y, 0)
        ax.plot(x / 1e6, mu, lw=1.6, **{**line_kw("student_ctbr"), "marker": None})
        ax.fill_between(x / 1e6, mu - sd, mu + sd, color=BLUE, alpha=0.13, lw=0)
        ax.set_yscale("log")
        plain_log(ax)
    ax.set_xlabel("Student steps (M)")
    ax.set_ylabel("Wind-estimate RMSE (% of weight)")
    panel(ax, 2, x=-0.18)
    fig.legend(handles=[Line2D([], [], **{**line_kw(g), "marker": None}) for g in groups + ["student_ctbr"]],
               loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1), fontsize=8.6, handlelength=2.8)
    fig.tight_layout(w_pad=2)
    save(fig, "F7_training_curves")


# ---------------------------------------------------------------- F8 TD3 tuning
def load_sweep():
    p = os.path.join(TUNE, "eval_v2_ctbr.json")
    if not os.path.exists(p):
        return []
    ev = json.load(open(p))["models"]
    meta = json.load(open(os.path.join(TUNE, "sweep_configs.json")))
    rows = []
    for tag, info in meta["configs"].items():
        name = f"td3sw_{tag}_s0"
        if name not in ev and "diverged" not in info:
            continue
        m = ev.get(name, {"main": None, "sweep": {}})
        diag = os.path.join(TUNE, f"diag_{tag}.json")
        bias = json.load(open(diag))["per_wind"] if os.path.exists(diag) else None
        rows.append({"tag": tag, "label": info["label"], "hp": info["hp"], "round": info["round"],
                     "chosen": tag == meta.get("chosen"), "diverged": info.get("diverged"), "main": m["main"], "sweep": m["sweep"],
                     "bias_cm": {w: v["bias_norm_cm"] for w, v in bias.items()} if bias else None,
                     "log": os.path.join(SWEEP_RUNS, name, "train_log.csv")})
    return rows


def fig8(rows):
    if not rows:
        print("skip F8 (no sweep results)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), gridspec_kw={"width_ratios": [1.25, 1, 0.8]})
    ax = axes[0]
    for r in rows:
        if not os.path.exists(r["log"]):
            continue
        x, c = read_log(r["log"])
        col = ORANGE if r["chosen"] else (INK2 if r["tag"] == "v1hp" else INK3)
        lw = 2.2 if r["chosen"] else 1.1
        yv = 100 * smooth(c["mean_pos_err"], 81)
        ax.plot(x / 1e6, yv, color=col, lw=lw, ls="--" if r["tag"] == "v1hp" else "-", zorder=3 if r["chosen"] else 2)
        if r["chosen"] or r["tag"] == "v1hp":
            ax.text(x[-1] / 1e6 + 0.4, yv[-1], "selected" if r["chosen"] else "default TD3", fontsize=7.6, color=col,
                    va="center", fontweight="bold" if r["chosen"] else "normal")
    ax.text(25.4, 13, "1-step,\nlower noise", fontsize=7.4, color=INK3, va="center")
    ax.text(25.4, 4.75, "other n = 3, 5", fontsize=7.4, color=INK3, va="center")
    ax.text(25.4, 38, "diverged\n(n = 10;\nn = 5 + lr 1e-4)", fontsize=7.4, color=INK3, va="center")
    ax.set_xlim(0, 33)
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_yscale("log")
    plain_log(ax)
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel("Training position error (cm)")
    panel(ax, 0, x=-0.17)
    labs = [("default (1-step, σ 0.2)" if r["tag"] == "v1hp" else r["tag"]) for r in rows]
    y = np.arange(len(rows))[::-1]
    for k, (key, sc, xl) in enumerate((("pos_err_m", 100, "Steady error, eval (cm)"), ("success_rate", 100, "In-range success (%)"))):
        ax = axes[k + 1]
        for yi, r in zip(y, rows):
            col = ORANGE if r["chosen"] else (INK2 if r["tag"] == "v1hp" else INK3)
            if r["main"] is None:
                ax.text(0.5 if key == "pos_err_m" else 2, yi, "diverged", va="center", fontsize=7.6, color=INK3, style="italic")
                continue
            ax.barh(yi, sc * r["main"][key], height=0.66, edgecolor=col, hatch="////" if r["tag"] == "v1hp" else "",
                    facecolor=col if r["tag"] != "v1hp" else tint(col, 0.25))
            ax.text(sc * r["main"][key], yi, f" {sc * r['main'][key]:.1f}", va="center", fontsize=7.6, color=INK2)
        ax.set_yticks(y, labs if k == 0 else [])
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(xl)
        panel(ax, k + 1, x=-0.32 if k == 0 else -0.05)
    axes[2].set_xlim(0, 112)
    fig.tight_layout(w_pad=0.6)
    save(fig, "F8_td3_tuning")


# ---------------------------------------------------------------- tables
def fmt(ms, key, sc=100, nd=2):
    mu, sd, _ = stat(ms, key)
    return f"{sc * mu:.{nd}f} ± {sc * sd:.{nd}f}"


def tables(v1g, v2g, rows):
    L = ["# Final tables (paper)", "",
         "All numbers: mean ± std over 3 training seeds (RL) or 3 independent tunings on seeds 11-13 (classical).",
         "Evaluation: 512 episodes, evaluation seed 1000, 12 s, gust at 6 s for 0.5 s; success = no crash and mean "
         "error over the last 2 s < 10 cm; recovery = time until the error stays < 5 cm for 0.5 s "
         "(mean over the episodes that recover; their share in the 'recovered' column).", "",
         "## Table 1 - v2 (realistic simulator), wind inside the training range (0-35 % of weight), gust +40 %", "",
         "| Controller | Steady error (cm) | Gust peak (cm) | Recovery (s) | Recovered | Success (%) | Crash (%) |",
         "|---|---|---|---|---|---|---|"]
    for g in [g for g in V2_ORDER if g in v2g]:
        ms = v2g[g]
        L.append(f"| {STYLE[g][0]} | {fmt(ms, 'pos_err_m')} | {fmt(ms, 'gust_peak_err_m', nd=1)} | "
                 f"{fmt(ms, 'recovery_s', 1)} | {fmt(ms, 'recovered_rate', 100, 1)} % | {fmt(ms, 'success_rate', 100, 1)} | "
                 f"{fmt(ms, 'crash_rate', 100, 1)} |")
    sweep = ["0.4", "0.5", "0.6", "0.7", "0.8"]
    L += ["", "## Table 2 - v2 success rate (%) with the steady wind fixed beyond the training range (+40 % gust)", "",
          "| Controller | " + " | ".join(f"{100 * float(s):.0f} % W" for s in sweep) + " |", "|---|" + "---|" * len(sweep)]
    for g in [g for g in V2_ORDER if g in v2g]:
        L.append(f"| {STYLE[g][0]} | " + " | ".join(
            f"{100 * stat(v2g[g], 'success_rate', s)[0]:.1f} ± {100 * stat(v2g[g], 'success_rate', s)[1]:.1f}" for s in sweep) + " |")
    v1name = {"pid_motor": "PID", "lqr_motor": "LQR", "lqi_motor": "LQI", "ppo_motor": "PPO", "td3_ctbr": "TD3"}
    L += ["", "## Table 3 - v1 (idealised simulator, motor commands), gust +0.10 N (36 % of weight)", "",
          "| Controller | Steady error (cm) | Gust peak (cm) | Recovery (s) | Success (%) | Success at 0.12 N (%) | Success at 0.14 N (%) |",
          "|---|---|---|---|---|---|---|"]
    for g in ["pid_motor", "lqr_motor", "lqi_motor", "ppo_motor", "td3_ctbr"]:
        if g in v1g:
            ms = v1g[g]
            L.append(f"| {v1name[g]} | {fmt(ms, 'pos_err_m')} | {fmt(ms, 'gust_peak_err_m', nd=1)} | {fmt(ms, 'recovery_s', 1)} | "
                     f"{fmt(ms, 'success_rate', 100, 1)} | {100 * stat(ms, 'success_rate', '0.12')[0]:.1f} | "
                     f"{100 * stat(ms, 'success_rate', '0.14')[0]:.1f} |")
    L += ["", "(DQN, v1 only: see results/summary.csv.)"]
    if rows:
        L += ["", "## Table 4 - TD3 (CTBR) hyper-parameter study: 1 seed (0), 25 M-step runs, same evaluation protocol", "",
              "| Tag | Setting (changes vs default TD3) | Steps (M) | Steady error (cm) | Gust peak (cm) | Success (%) | "
              "Success 30 % W (%) | Static bias at 0 / 30 % W (cm) |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lg = os.path.join(SWEEP_RUNS, f"td3sw_{r['tag']}_s0", "train_log.csv")
            steps = float(open(lg).read().strip().split("\n")[-1].split(",")[0]) / 1e6
            b = r["bias_cm"]
            bs = f"{b['0.0']:.1f} / {b['0.3']:.1f}" if b else "-"
            s30 = r["sweep"].get("0.3", {}).get("success_rate")
            if r["main"] is None:
                L.append(f"| {r['tag']} | {r['label']} | {steps:.0f} | diverged: {r['diverged']} | | | | |")
                continue
            tag = "default" if r["tag"] == "v1hp" else r["tag"]
            L.append(f"| {'**' + tag + '**' if r['chosen'] else tag} | {r['label']} | {steps:.0f} | "
                     f"{100 * r['main']['pos_err_m']:.2f} | {100 * r['main']['gust_peak_err_m']:.1f} | "
                     f"{100 * r['main']['success_rate']:.1f} | {'-' if s30 is None else f'{100 * s30:.1f}'} | {bs} |")
    we = json.load(open(os.path.join(RES2, "wind_estimate.json")))["summary"]
    L += ["", "## Table 5 - RMA wind-estimate error (% of weight; mean over 3 students, 512 episodes each)", "",
          "| Base wind | RMSE steady (2-6 s) | RMSE during gust | Bias (steady) | Correlation |", "|---|---|---|---|---|"]
    for w in ("0.3", "0.6"):
        vs = [v for k, v in we.items() if k.endswith("@" + w)]
        f = lambda k: np.mean([v[k] for v in vs])  # noqa: E731
        L.append(f"| {100 * float(w):.0f} % W | {100 * f('rmse_steady'):.1f} | {100 * f('rmse_gust'):.1f} | "
                 f"{100 * f('magnitude_bias_steady'):+.1f} | {f('corr_xy'):.3f} |")
    open(os.path.join(RES2, "paper_tables.md"), "w").write("\n".join(L) + "\n")
    print("wrote results_v2/paper_tables.md")


def main():
    v2g, v1g = by_group(load_v2()), by_group(load_v1())
    rows = load_sweep()
    fig1()
    fig2(v2g)
    fig3(v2g)
    fig4(v1g, v2g)
    fig5()
    fig6()
    fig7()
    fig8(rows)
    tables(v1g, v2g, rows)


if __name__ == "__main__":
    main()
