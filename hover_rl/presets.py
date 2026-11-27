"""Environment presets.

v1          : the environment of the 2026-11 re-run (RPM in [4500, 5500] ~ 0.81-1.21 x weight, ideal motors, no delay,
              no noise, wind 0.02-0.08 N).
v2_motor    : realistic Crazyflie-2.1-like actuation, per-motor thrust commands.
v2_ctbr     : same physics, collective thrust + body-rate commands with an onboard rate loop.
v2_*_priv   : + privileged observation (RMA teacher).

v2 physics (identical for every controller):
  * thrust-to-weight 1.9 (Crazyflie 2.1 is ~1.9-2.0 with stock props/battery);
  * first-order motor speed lag, tau ~ U[20, 50] ms (brushed coreless motors are reported in the tens of ms);
  * command latency of 1-2 policy steps (20-40 ms; radio + estimator latency);
  * domain randomisation: body mass/inertia x U[0.9, 1.1], thrust coefficient x U[0.9, 1.1];
  * sensor noise: position 5 mm, velocity 2 cm/s, attitude 0.01 rad, gyro 0.05 rad/s (1 sigma);
  * wind in % of the nominal weight W0 = 0.277 N: base U[0, 35] %, random training gusts U[10, 40] % for 0.2-0.5 s.
"""

from __future__ import annotations

from hover_rl.env import HoverWindEnvCfg

V2_WIND = (0.0, 0.35)  # fraction of nominal weight
V2_GUST = 0.40


def make_cfg(preset: str) -> HoverWindEnvCfg:
    cfg = HoverWindEnvCfg()
    if preset == "v1":
        return cfg
    base, _, extra = preset.partition("_priv")
    if base not in ("v2_motor", "v2_ctbr"):
        raise ValueError(preset)
    cfg.action_mode = base.split("_")[1]
    cfg.thrust_to_weight = 1.9
    cfg.motor_tau_range = (0.02, 0.05)
    cfg.delay_steps_range = (1, 2)
    cfg.mass_scale_range = (0.9, 1.1)
    cfg.kf_scale_range = (0.9, 1.1)
    cfg.noise_pos, cfg.noise_vel, cfg.noise_att, cfg.noise_gyro = 0.005, 0.02, 0.01, 0.05
    cfg.obs_prev_actions = 2
    cfg.wind_in_weight = True
    cfg.wind_min, cfg.wind_max = V2_WIND
    cfg.random_gust_N = (0.10, V2_GUST)
    cfg.gust_strength = V2_GUST
    cfg.action_rate_penalty = 0.01
    cfg.privileged_obs = preset.endswith("_priv")
    cfg.observation_space = 18 + 8 + (7 if cfg.privileged_obs else 0)
    return cfg
