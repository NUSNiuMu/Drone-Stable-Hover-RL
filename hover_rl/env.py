"""Unified wind-disturbed hover environment (Isaac Lab DirectRLEnv, Crazyflie 2.x).

One environment definition shared by PPO, TD3 and DQN so that the comparison is apples-to-apples.

Task      : hover at (0, 0, 1) m (relative to each env origin).
Obs (18)  : [pos(3), target(3), gravity_dir_body(3), lin_vel_body(3), ang_vel_body(3), pos_error_body(3)]
Action (4): per-motor command in [-1, 1] -> RPM in [4500, 5500]; thrust = k_f * omega^2 applied to each
            propeller body, plus the rotor reaction torque (k_m/k_f = 0.025 m) about body z so yaw is controllable.
            k_f is calibrated so that 5000 RPM on all four motors exactly balances gravity.
Init      : within +-0.2 m of the target, level, at rest, yaw ~ U(-pi, pi).
Control   : physics dt = 0.01 s, decimation 2 -> policy step 0.02 s (50 Hz).
Wind      : horizontal force in the WORLD frame (converted to the body frame before it is applied, because
            Isaac Lab external wrenches are expressed in the body frame). Base strength ~ U[wind_min, wind_max] N,
            random direction; every `wind_update_s` the strength is re-scaled by (1 + U(-0.2, 0.2)) and the
            direction jittered by U(-0.1, 0.1) rad (gusty wind).
Gusts     : additive pulses in a random horizontal direction. Training: Poisson rate 0.3 Hz, U[0.03, 0.10] N for
            U[0.2, 0.5] s. Evaluation: random gusts off, one scheduled pulse of `gust_strength` N for
            `gust_duration_s` at `gust_time_s` (used to measure recovery time).

v2 options (all default to the v1 behaviour above; presets live in hover_rl/presets.py):
  action_mode "motor" : per-motor thrust command a -> f = (a+1)/2 * f_max, f_max = TWR * W0 / 4 (TWR ~1.9 like a
                        Crazyflie 2.1), motor speed omega = sqrt(f / k_f).
  action_mode "ctbr"  : collective thrust + body rates; an onboard P rate loop (100 Hz, nominal inertia) + mixer
                        turns them into per-motor thrusts.
  motor lag           : first-order motor-speed dynamics, time constant sampled per episode.
  command delay       : the command applied is the one from `delay` policy steps ago (sampled per episode).
  domain randomisation: body mass/inertia (per env at start-up), thrust coefficient, motor tau, delay.
  sensor noise        : Gaussian noise on position, velocity, attitude and gyro (observation and PID inputs).
  privileged obs      : [wind force in body frame / W0, mass scale-1, k_f scale-1, tau (norm.), delay (norm.)] for
                        the RMA teacher.
"""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, quat_rotate, quat_rotate_inverse

from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip

MIN_RPM = 4500.0
MAX_RPM = 5500.0
HOVER_RPM = 5000.0


@configclass
class HoverWindEnvCfg(DirectRLEnvCfg):
    episode_length_s = 10.0
    decimation = 2
    action_space = 4
    observation_space = 18
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2.5, replicate_physics=True)
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    target = (0.0, 0.0, 1.0)
    init_pos_noise = 0.2  # m, uniform in each axis around the target

    # rotor reaction (yaw) torque: tau_z = -spin_dir * k_m/k_f * thrust (Crazyflie k_m/k_f ~ 0.025 m).
    # m1/m3 spin +z, m2/m4 spin -z (CRAZYFLIE_CFG joint velocities), so differential RPM controls yaw.
    yaw_torque_coeff = 0.025
    randomize_init_yaw = True

    # termination (relative to env origin), same bounds as the original drone_PPO.py
    min_height = 0.2
    max_height = 1.5
    xy_boundary = 1.0

    # wind (N). Crazyflie weight is ~0.27 N, see README for the feasibility argument.
    enable_wind = True
    wind_min = 0.02
    wind_max = 0.08
    wind_update_s = 1.0
    wind_variation = 0.2
    wind_dir_jitter = 0.1

    # random additive gust pulses during training: Poisson rate, strength and duration ranges
    random_gust_rate_hz = 0.3
    random_gust_N = (0.03, 0.10)
    random_gust_s = (0.2, 0.5)

    # evaluation gust pulse at a fixed time (disabled when gust_time_s < 0); random gusts are switched off in eval
    gust_time_s = -1.0
    gust_strength = 0.10
    gust_duration_s = 0.5

    # spread episode start times during training (disable for synchronous evaluation)
    randomize_episode_start = True

    # reward
    dist_scale = 0.3
    precision_sigma2 = 0.005
    lin_vel_penalty = 0.05
    ang_vel_penalty = 0.02

    # ---------------- v2 realism options (defaults reproduce the v1 environment exactly) ----------------
    action_mode = "rpm_v1"  # "rpm_v1" | "motor" | "ctbr"
    thrust_to_weight = 1.9  # total max thrust / nominal weight for "motor" and "ctbr"
    max_body_rate = (4.0, 4.0, 2.0)  # rad/s at |a| = 1 (ctbr)
    rate_gain = (20.0, 20.0, 10.0)  # onboard rate loop: tau = J * K * (w_des - w)   [1/s]
    motor_tau_range = (0.0, 0.0)  # s, first-order motor time constant (0 -> instantaneous)
    delay_steps_range = (0, 0)  # policy steps of command latency (inclusive range)
    mass_scale_range = (1.0, 1.0)  # body mass + inertia scale, per env at start-up
    kf_scale_range = (1.0, 1.0)  # thrust coefficient scale, per episode
    noise_pos = 0.0  # m
    noise_vel = 0.0  # m/s
    noise_att = 0.0  # rad (small random rotation)
    noise_gyro = 0.0  # rad/s
    obs_prev_actions = 0  # append the last k policy actions
    privileged_obs = False  # append 7 privileged dims (teacher)
    wind_in_weight = False  # wind_min/max, random_gust_N and gust_strength given as fractions of the nominal weight
    action_rate_penalty = 0.0


class HoverWindEnv(DirectRLEnv):
    cfg: HoverWindEnvCfg

    def __init__(self, cfg: HoverWindEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device

        self._body_id = self._robot.find_bodies("body")[0]
        self._prop_ids = self._robot.find_bodies("m.*_prop")[0]
        self._wrench_ids = list(self._body_id) + list(self._prop_ids)  # one call -> no buffer overwrite
        self._forces = torch.zeros(n, len(self._wrench_ids), 3, device=dev)
        self._torques = torch.zeros_like(self._forces)

        mass = self._robot.root_physx_view.get_masses()[0].sum().item()
        g = torch.tensor(self.sim.cfg.gravity).norm().item()
        self.robot_mass, self.robot_weight = mass, mass * g
        hover_omega = HOVER_RPM * 2.0 * math.pi / 60.0
        self.k_f = self.robot_weight / 4.0 / hover_omega**2

        self.target_local = torch.tensor(self.cfg.target, device=dev).repeat(n, 1)
        self.actions = torch.zeros(n, 4, device=dev)
        self.rpm = torch.full((n, 4), HOVER_RPM, device=dev)

        # wind state (world frame)
        self.wind_base_strength = torch.zeros(n, device=dev)
        self.wind_base_dir = torch.zeros(n, device=dev)
        self.wind_force_w = torch.zeros(n, 3, device=dev)
        self.wind_update_steps = max(1, int(round(self.cfg.wind_update_s / self.step_dt)))
        self.gust_vec = torch.zeros(n, 3, device=dev)
        self.gust_left = torch.zeros(n, dtype=torch.long, device=dev)

        # ---------------- v2 state ----------------
        c = self.cfg
        self.legacy = c.action_mode == "rpm_v1" and c.motor_tau_range[1] <= 0 and c.delay_steps_range[1] == 0 \
            and c.kf_scale_range == (1.0, 1.0)
        self.W0 = self.robot_weight  # nominal weight (before mass randomisation)
        self.wscale = self.W0 if c.wind_in_weight else 1.0
        self.f_max = c.thrust_to_weight * self.W0 / 4.0
        self.hover_omega = hover_omega
        # propeller positions in the body frame (default pose is level, identity orientation)
        rp = self._robot.data.body_pos_w[0, self._prop_ids] - self._robot.data.root_pos_w[0]
        self.prop_r = quat_rotate_inverse(self._robot.data.root_quat_w[0].repeat(4, 1), rp)
        self.spin = torch.tensor([1.0, -1.0, 1.0, -1.0], device=dev)
        A = torch.stack([torch.ones(4, device=dev), self.prop_r[:, 1], -self.prop_r[:, 0], -self.spin * c.yaw_torque_coeff])
        self.mix_inv = torch.linalg.inv(A)  # [T, tx, ty, tz] -> f_i
        # composite inertia about the CoM (nominal), diagonal
        inert = self._robot.root_physx_view.get_inertias()[0].reshape(-1, 3, 3).to(dev)
        masses = self._robot.root_physx_view.get_masses()[0].to(dev)
        rel = self._robot.data.body_pos_w[0] - self._robot.data.root_pos_w[0]
        J = inert.sum(0) + sum(masses[i] * (rel[i].dot(rel[i]) * torch.eye(3, device=dev) - torch.outer(rel[i], rel[i]))
                               for i in range(len(masses)))
        self.J = torch.diagonal(J).clone()
        self.rate_gain = torch.tensor(c.rate_gain, device=dev)
        self.max_rate = torch.tensor(c.max_body_rate, device=dev)
        # domain randomisation buffers
        self.mass_scale = torch.ones(n, device=dev)
        if c.mass_scale_range != (1.0, 1.0):
            ms = torch.empty(n).uniform_(*c.mass_scale_range)
            ids = torch.arange(n, device="cpu")
            m_all = self._robot.root_physx_view.get_masses()
            i_all = self._robot.root_physx_view.get_inertias()
            b = self._body_id[0]
            m_all[:, b] *= ms
            i_all[:, b] *= ms[:, None]
            self._robot.root_physx_view.set_masses(m_all, ids)
            self._robot.root_physx_view.set_inertias(i_all, ids)
            self.mass_scale = ms.to(dev)
        self.kf_scale = torch.ones(n, device=dev)
        self.motor_tau = torch.zeros(n, device=dev)
        self.delay = torch.zeros(n, dtype=torch.long, device=dev)
        self.omega = torch.full((n, 4), hover_omega, device=dev)
        self.act_buf = torch.zeros(n, max(c.delay_steps_range[1], c.obs_prev_actions, 1) + 1, 4, device=dev)
        self.cmd = torch.zeros(n, 4, device=dev)
        self.meas = {}
        self._gyro_gen = torch.Generator(device=dev).manual_seed(int(c.seed or 0) + 12345)
        self.obs_dim = 18 + 4 * c.obs_prev_actions + (7 if c.privileged_obs else 0)
        self.priv_dim = 7 if c.privileged_obs else 0

        self.terminal_obs = torch.zeros(n, self.obs_dim, device=dev)
        self.terminal_err = torch.zeros(n, device=dev)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
        self.scene.clone_environments(copy_from_source=False)
        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light.func("/World/Light", light)

    # ------------------------------------------------------------------ wind
    def _sample_wind(self, env_ids: torch.Tensor, strength: torch.Tensor | None = None):
        k = len(env_ids)
        if strength is None:
            strength = torch.empty(k, device=self.device).uniform_(self.cfg.wind_min, self.cfg.wind_max) * self.wscale
        self.wind_base_strength[env_ids] = strength
        self.wind_base_dir[env_ids] = torch.empty(k, device=self.device).uniform_(-math.pi, math.pi)
        self._set_wind(env_ids, self.wind_base_strength[env_ids], self.wind_base_dir[env_ids])

    def _set_wind(self, env_ids, strength, direction):
        if not self.cfg.enable_wind:
            strength = torch.zeros_like(strength)
        self.wind_force_w[env_ids, 0] = strength * torch.cos(direction)
        self.wind_force_w[env_ids, 1] = strength * torch.sin(direction)
        self.wind_force_w[env_ids, 2] = 0.0

    def _update_wind(self):
        step = self.episode_length_buf
        n, dev = self.num_envs, self.device
        # 1) gusty variation of the base wind every wind_update_steps (per env, own episode clock)
        upd = ((step % self.wind_update_steps) == 0) & (step > 0)
        ids = upd.nonzero(as_tuple=False).squeeze(-1)
        if len(ids) > 0:
            v = self.cfg.wind_variation
            s = self.wind_base_strength[ids] * (1.0 + torch.empty(len(ids), device=dev).uniform_(-v, v))
            j = self.cfg.wind_dir_jitter
            d = self.wind_base_dir[ids] + torch.empty(len(ids), device=dev).uniform_(-j, j)
            self._set_wind(ids, s, d)
        # 2) additive gust pulses
        self.gust_left = (self.gust_left - 1).clamp(min=0)
        start = torch.zeros(n, dtype=torch.bool, device=dev)
        strength = torch.zeros(n, device=dev)
        dur = torch.zeros(n, device=dev)
        if self.cfg.enable_wind and self.cfg.random_gust_rate_hz > 0:  # random gusts (training)
            p = self.cfg.random_gust_rate_hz * self.step_dt
            rnd = (torch.rand(n, device=dev) < p) & (self.gust_left == 0)
            start |= rnd
            strength = torch.where(rnd, torch.empty(n, device=dev).uniform_(*self.cfg.random_gust_N) * self.wscale, strength)
            dur = torch.where(rnd, torch.empty(n, device=dev).uniform_(*self.cfg.random_gust_s), dur)
        if self.cfg.gust_time_s >= 0:  # scheduled gust (evaluation)
            sched = step == int(round(self.cfg.gust_time_s / self.step_dt))
            start |= sched
            strength = torch.where(sched, torch.full_like(strength, self.cfg.gust_strength * self.wscale), strength)
            dur = torch.where(sched, torch.full_like(dur, self.cfg.gust_duration_s), dur)
        if start.any():
            d = torch.empty(n, device=dev).uniform_(-math.pi, math.pi)
            vec = torch.stack([strength * torch.cos(d), strength * torch.sin(d), torch.zeros_like(d)], -1)
            self.gust_vec = torch.where(start.unsqueeze(-1), vec, self.gust_vec)
            steps = torch.clamp(torch.round(dur / self.step_dt), min=1).long()
            self.gust_left = torch.where(start, steps, self.gust_left)

    def total_wind(self) -> torch.Tensor:
        return self.wind_force_w + self.gust_vec * (self.gust_left > 0).unsqueeze(-1).float()

    # ------------------------------------------------------------------ physics
    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        if self.legacy:  # v1 path, kept bit-for-bit
            self.rpm = MIN_RPM + (self.actions + 1.0) * 0.5 * (MAX_RPM - MIN_RPM)
            omega = self.rpm * 2.0 * math.pi / 60.0
            thrust = self.k_f * omega**2  # (n, 4)
            self._update_wind()
            quat = self._robot.data.root_quat_w
            self._forces.zero_()
            self._forces[:, 0, :] = quat_rotate_inverse(quat, self.total_wind())  # wind on the body, body frame
            self._forces[:, 1:, 2] = thrust  # thrust along each propeller's local z axis
            self._torques.zero_()
            self._torques[:, 0, 2] = -(self.spin * self.cfg.yaw_torque_coeff * thrust).sum(-1)  # reaction torque
            return
        self.act_buf = torch.roll(self.act_buf, 1, dims=1)
        self.act_buf[:, 0] = self.actions
        self.cmd = self.act_buf[torch.arange(self.num_envs, device=self.device), self.delay]
        self._update_wind()

    def mix(self, T: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Collective thrust (n,) + body torques (n,3) -> per-motor thrust (n,4), clipped to [0, f_max]."""
        f = torch.cat([T.unsqueeze(-1), tau], -1) @ self.mix_inv.T
        return f.clamp(0.0, self.f_max)

    def rate_loop(self, T, w_des):
        w = self._robot.data.root_ang_vel_b
        if self.cfg.noise_gyro > 0:  # own generator: the inner loop must not shift the global RNG stream
            w = w + self.cfg.noise_gyro * torch.randn(w.shape, device=self.device, generator=self._gyro_gen)
        return self.mix(T, self.J * self.rate_gain * (w_des - w))

    def _apply_action(self):
        if not self.legacy:
            c, cmd = self.cfg, self.cmd
            if c.action_mode == "rpm_v1":
                rpm = MIN_RPM + (cmd + 1.0) * 0.5 * (MAX_RPM - MIN_RPM)
                w_cmd = rpm * 2.0 * math.pi / 60.0
            else:
                if c.action_mode == "motor":
                    f = (cmd + 1.0) * 0.5 * self.f_max
                else:  # ctbr
                    f = self.rate_loop((cmd[:, 0] + 1.0) * 0.5 * 4.0 * self.f_max, cmd[:, 1:] * self.max_rate)
                w_cmd = torch.sqrt(f / self.k_f)  # the onboard controller assumes the nominal k_f
            alpha = 1.0 - torch.exp(-self.cfg.sim.dt / self.motor_tau.clamp(min=1e-6))
            self.omega = self.omega + alpha.unsqueeze(-1) * (w_cmd - self.omega)
            thrust = self.k_f * self.kf_scale.unsqueeze(-1) * self.omega**2
            self.rpm = self.omega * 60.0 / (2.0 * math.pi)
            self._forces.zero_()
            self._forces[:, 0, :] = quat_rotate_inverse(self._robot.data.root_quat_w, self.total_wind())
            self._forces[:, 1:, 2] = thrust
            self._torques.zero_()
            self._torques[:, 0, 2] = -(self.spin * self.cfg.yaw_torque_coeff * thrust).sum(-1)
        self._robot.set_external_force_and_torque(self._forces, self._torques, body_ids=self._wrench_ids)

    # ------------------------------------------------------------------ obs / reward / done
    def _compute_obs(self) -> torch.Tensor:
        """Body-frame observation (yaw-invariant, no Euler-angle wrap-around), built from (noisy) measurements.

        [pos (3, env frame), target (3), gravity direction in body frame (3), lin vel body (3), ang vel body (3),
         position error in body frame (3)] + last k actions (4k) + privileged (7, teacher only)
        """
        c = self.cfg
        pos = self._robot.data.root_pos_w - self.scene.env_origins
        quat = self._robot.data.root_quat_w
        vel = self._robot.data.root_lin_vel_w
        gyro = self._robot.data.root_ang_vel_b
        if c.noise_pos > 0 or c.noise_vel > 0 or c.noise_att > 0 or c.noise_gyro > 0:
            pos = pos + c.noise_pos * torch.randn_like(pos)
            vel = vel + c.noise_vel * torch.randn_like(vel)
            gyro = gyro + c.noise_gyro * torch.randn_like(gyro)
            rv = c.noise_att * torch.randn_like(pos)
            ang = rv.norm(dim=-1)
            quat = quat_mul(quat, quat_from_angle_axis(ang, rv / ang.clamp(min=1e-9).unsqueeze(-1)))
            grav = quat_rotate_inverse(quat, torch.tensor([0.0, 0.0, -1.0], device=self.device).expand_as(pos))
            vel_b = quat_rotate_inverse(quat, vel)
        else:
            grav = self._robot.data.projected_gravity_b
            vel_b = self._robot.data.root_lin_vel_b
        self.meas = {"pos": pos, "quat": quat, "vel_w": vel, "gyro": gyro}
        err_b = quat_rotate_inverse(quat, pos - self.target_local)
        parts = [pos, self.target_local, grav, vel_b, gyro, err_b]
        if c.obs_prev_actions > 0:
            parts.append(self.act_buf[:, : c.obs_prev_actions].reshape(self.num_envs, -1))
        if c.privileged_obs:
            parts.append(self.privileged())
        return torch.cat(parts, dim=-1)

    def privileged(self) -> torch.Tensor:
        wind_b = quat_rotate_inverse(self._robot.data.root_quat_w, self.total_wind()) / self.W0
        return torch.cat([wind_b, (self.mass_scale - 1.0).unsqueeze(-1), (self.kf_scale - 1.0).unsqueeze(-1),
                          ((self.motor_tau - 0.035) / 0.015).unsqueeze(-1), (self.delay.float() - 1.5).unsqueeze(-1)], -1)

    def _get_observations(self) -> dict:
        return {"policy": self._compute_obs()}

    def position_error(self) -> torch.Tensor:
        pos = self._robot.data.root_pos_w - self.scene.env_origins
        return torch.linalg.norm(pos - self.target_local, dim=-1)

    def _get_rewards(self) -> torch.Tensor:
        d = self.position_error()
        lin = torch.sum(self._robot.data.root_lin_vel_w**2, dim=-1)
        ang = torch.sum(self._robot.data.root_ang_vel_w**2, dim=-1)
        r = (1.0 - torch.tanh(d / self.cfg.dist_scale)) + torch.exp(-(d**2) / self.cfg.precision_sigma2)
        r = r - self.cfg.lin_vel_penalty * lin - self.cfg.ang_vel_penalty * ang
        if self.cfg.action_rate_penalty > 0:
            r = r - self.cfg.action_rate_penalty * torch.sum((self.act_buf[:, 0] - self.act_buf[:, 1]) ** 2, dim=-1)
        return r

    def _get_dones(self):
        pos = self._robot.data.root_pos_w - self.scene.env_origins
        died = (pos[:, 2] < self.cfg.min_height) | (pos[:, 2] > self.cfg.max_height)
        died |= (pos[:, 0].abs() > self.cfg.xy_boundary) | (pos[:, 1].abs() > self.cfg.xy_boundary)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # observation BEFORE the automatic reset (needed for correct bootstrapping on time-outs)
        self.terminal_obs = self._compute_obs()
        self.terminal_err = self.position_error()  # true error (no sensor noise), used by the evaluation
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if self.cfg.randomize_episode_start and len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        k = len(env_ids)
        root = self._robot.data.default_root_state[env_ids].clone()
        noise = torch.empty(k, 3, device=self.device).uniform_(-self.cfg.init_pos_noise, self.cfg.init_pos_noise)
        root[:, :3] = self.target_local[env_ids] + noise + self.scene.env_origins[env_ids]
        root[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # level attitude (w, x, y, z)
        if self.cfg.randomize_init_yaw:
            yaw = torch.empty(k, device=self.device).uniform_(-math.pi, math.pi)
            root[:, 3] = torch.cos(yaw / 2)
            root[:, 6] = torch.sin(yaw / 2)
        root[:, 7:] = 0.0
        self._robot.write_root_pose_to_sim(root[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids], self._robot.data.default_joint_vel[env_ids], None, env_ids
        )
        self.actions[env_ids] = 0.0
        self._sample_wind(env_ids)
        if hasattr(self, "gust_left"):
            self.gust_left[env_ids] = 0
            self.gust_vec[env_ids] = 0.0
        if hasattr(self, "act_buf"):
            c, dev = self.cfg, self.device
            self.act_buf[env_ids] = 0.0
            self.cmd[env_ids] = 0.0
            # degenerate ranges are filled without drawing random numbers (keeps the v1 RNG stream unchanged)
            def draw(lo, hi):
                return torch.empty(k, device=dev).uniform_(lo, hi) if hi > lo else torch.full((k,), float(lo), device=dev)

            self.kf_scale[env_ids] = draw(*c.kf_scale_range)
            self.motor_tau[env_ids] = draw(*c.motor_tau_range)
            lo, hi = c.delay_steps_range
            self.delay[env_ids] = torch.randint(lo, hi + 1, (k,), device=dev) if hi > lo else torch.full((k,), lo, device=dev)
            # motors start at the speed that hovers this airframe
            w0 = torch.sqrt(self.W0 * self.mass_scale[env_ids] / 4.0 / (self.k_f * self.kf_scale[env_ids]))
            self.omega[env_ids] = w0.unsqueeze(-1).expand(-1, 4)
