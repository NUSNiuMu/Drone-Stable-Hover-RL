"""Render a scripted wind scenario for one trained policy (headless camera), frames -> MP4.

    hover_rl/run.sh hover_rl/record.py --model runs/ppo_s0/model.pt --algo ppo --label "PPO" --out media/ppo.mp4
    hover_rl/run.sh hover_rl/record.py --preset v2_ctbr_priv --algo pid --model pid/v2_ctbr_s11.json \
        --base_n 0.5 --gust_n 0.4 --label "PID" --out media/v2_pid.mp4      (v2: wind given as fraction of weight)

Scenario (same seed for every policy, so the wind is identical):
  0-3 s  no wind   |   3-12 s base wind (default 0.06 N)   |   7 s gust pulse (default +0.10 N, same direction) for 0.5 s
A red arrow shows the current wind force (length ~ magnitude), a green cube marks the target (0, 0, 1) m.
"""

import argparse
import math
import os
import subprocess
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--algo", choices=["ppo", "td3", "dqn", "pid", "student"], required=True)
parser.add_argument("--preset", default="v1")
parser.add_argument("--obs_dim", type=int, default=18, help="observation entries the RL policy was trained on")
parser.add_argument("--label", default=None)
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--size", type=int, default=640)
parser.add_argument("--arrow_k", type=float, default=1.0, help="arrow USD scale per metre of desired length")
parser.add_argument("--base_n", type=float, default=0.06)
parser.add_argument("--gust_n", type=float, default=0.10)
parser.add_argument("--wind_deg", type=float, default=215.0, help="base wind direction (deg)")
parser.add_argument("--gust_deg", type=float, default=215.0, help="gust direction (deg); same as wind -> they add up")
parser.add_argument("--max_s", type=float, default=None, help="stop early (for quick tests)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR  # noqa: E402
from isaaclab.utils.math import quat_from_euler_xyz  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hover_rl.control import load_controller  # noqa: E402
from hover_rl.env import HoverWindEnv  # noqa: E402
from hover_rl.presets import make_cfg  # noqa: E402

T_WIND, T_GUST, T_END = 3.0, 7.0, 12.0
GUST_S = 0.5


def font(sz):
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(f):
            return ImageFont.truetype(f, sz)
    return ImageFont.load_default()


def main():
    cfg = make_cfg(args.preset)
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.episode_length_s = T_END + 1.0
    cfg.randomize_episode_start = False
    cfg.random_gust_rate_hz = 0.0
    cfg.wind_min = cfg.wind_max = 0.0
    cfg.wind_variation = 0.0
    cfg.wind_dir_jitter = 0.0
    cfg.init_pos_noise = 0.0
    cfg.randomize_init_yaw = False
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (0.26, -0.36, 1.12)
    cfg.viewer.lookat = (0.0, 0.0, 0.99)
    cfg.viewer.resolution = (args.size, args.size)
    env = HoverWindEnv(cfg, render_mode="rgb_array")
    dev = env.device

    ctrl = load_controller(args.algo, args.model, env, args.obs_dim)
    unit = 1.0 if args.preset == "v1" else env.W0  # v2: wind arguments are fractions of the nominal weight

    markers = VisualizationMarkers(VisualizationMarkersCfg(
        prim_path="/Visuals/hover",
        markers={
            "target": sim_utils.CuboidCfg(size=(0.02, 0.02, 0.02),
                                          visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.2))),
            "wind": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd", scale=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.15, 0.1))),
        }))

    BASE_N, BASE_DIR = args.base_n * unit, math.radians(args.wind_deg)
    GUST_N, GUST_DIR = args.gust_n * unit, math.radians(args.gust_deg)
    obs, _ = env.reset(seed=args.seed)
    obs = obs["policy"]
    ctrl.reset()
    origin = env.scene.env_origins[0]
    target_w = origin + torch.tensor([0.0, 0.0, 1.0], device=dev)
    steps = int(round((args.max_s or T_END) / env.step_dt))
    W = H = args.size
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(int(round(1 / env.step_dt))), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "18", args.out], stdin=subprocess.PIPE)
    f_big, f_small = font(int(W * 0.05)), font(int(W * 0.036))
    errs = []
    for k in range(steps):
        t = k * env.step_dt
        # scripted wind (world frame)
        if abs(t - T_WIND) < 1e-6:
            env.wind_base_strength[0], env.wind_base_dir[0] = BASE_N, BASE_DIR
            env._set_wind(torch.tensor([0], device=dev), env.wind_base_strength[:1], env.wind_base_dir[:1])
        if abs(t - T_GUST) < 1e-6:
            env.gust_vec[0] = torch.tensor([GUST_N * math.cos(GUST_DIR), GUST_N * math.sin(GUST_DIR), 0.0], device=dev)
            env.gust_left[0] = int(round(GUST_S / env.step_dt)) + 1
        obs, _, term, _, _ = env.step(ctrl.act(obs, deterministic=True))
        obs = obs["policy"]
        pos_w = env._robot.data.root_pos_w[0]
        err = float(env.position_error()[0])
        errs.append(err)
        wind = env.total_wind()[0]
        mag = float(wind.norm())
        # markers: target cube + wind arrow placed upwind of the drone, pointing along the wind
        yaw = torch.atan2(wind[1], wind[0]).reshape(1)
        q = quat_from_euler_xyz(torch.zeros(1, device=dev), torch.zeros(1, device=dev), yaw)
        length = 0.04 + 0.8 * mag  # metres on screen
        arrow_pos = pos_w - (wind / max(mag, 1e-6)) * (0.07 + length / 2) if mag > 1e-4 else pos_w + 10.0
        markers.visualize(
            translations=torch.stack([target_w, arrow_pos]),
            orientations=torch.cat([torch.tensor([[1.0, 0, 0, 0]], device=dev), q]),
            scales=torch.tensor([[1.0, 1.0, 1.0], [args.arrow_k * length, args.arrow_k * 0.05, args.arrow_k * 0.05]], device=dev),
            marker_indices=torch.tensor([0, 1], device=dev))
        frame = env.render()
        if frame is None or frame.size == 0:
            continue
        img = Image.fromarray(np.ascontiguousarray(frame[:, :, :3]).astype(np.uint8))
        d = ImageDraw.Draw(img)
        pw = lambda f: f"{f:.2f} N" if unit == 1.0 else f"{f:.2f} N ({100 * f / unit:.0f}% wt)"  # noqa: E731
        phase = "no wind" if t < T_WIND else (f"GUST +{pw(GUST_N)}" if T_GUST <= t < T_GUST + GUST_S else f"wind {pw(BASE_N)}")
        d.rectangle([0, 0, W, int(H * 0.17)], fill=(0, 0, 0))
        d.text((int(W * 0.03), int(H * 0.02)), args.label or args.algo.upper(), font=f_big, fill=(255, 255, 255))
        d.text((int(W * 0.03), int(H * 0.095)), f"t = {t:4.1f} s   {phase}", font=f_small,
               fill=(255, 210, 120) if t >= T_WIND else (200, 200, 200))
        d.text((int(W * 0.62), int(H * 0.035)), f"error {100 * err:4.1f} cm", font=f_small, fill=(255, 255, 255))
        if bool(term[0]):
            d.text((int(W * 0.3), int(H * 0.5)), "CRASHED", font=f_big, fill=(255, 60, 60))
        ff.stdin.write(np.asarray(img).tobytes())
    ff.stdin.close()
    ff.wait()
    e = np.array(errs)
    tt = np.arange(len(e)) * env.step_dt
    print(f"RECORD {args.out}: mean err no-wind {e[(tt >= 1) & (tt < T_WIND)].mean():.4f} m, "
          f"wind {e[(tt >= T_WIND + 1) & (tt < T_GUST)].mean():.4f} m, gust peak {e[tt >= T_GUST].max():.4f} m", flush=True)
    # Isaac Sim sometimes hangs in shutdown after headless camera rendering; the video is complete at this point
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
    app.close()
