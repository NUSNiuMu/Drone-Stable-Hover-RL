"""Non-RL controllers and controller loading.

CascadePID : classical cascaded controller, the usual quadrotor structure (as in the Crazyflie firmware):
             position PID (world frame) -> desired thrust vector -> tilt-prioritised attitude P -> body-rate command
             -> body-rate P loop -> mixer -> per-motor command. In the "ctbr" action mode the rate loop + mixer run in
             the environment (the same onboard loop the CTBR RL policy uses); otherwise the controller runs them itself.
             It reads the same noisy measurements as the RL observation (env.meas) and only knows the NOMINAL mass.
LQR / LQI  : full-state linear-quadratic regulator around hover (small-angle model in the heading frame, nominal
             mass/inertia, no motor lag/delay in the model), discretised at the 50 Hz control rate (ZOH) and solved
             with the discrete algebraic Riccati equation. Output = collective-thrust deviation + body torques -> same
             mixer + motor saturation as the PID. LQI adds the integral of the position error as 3 extra states.
             Q/R are parameterised by Bryson's rule (maximum acceptable deviation per state/input) and tuned like the PID.
             Like the PID's tilt limit, the position error fed to the LQR is clipped per axis (err_clip, also tuned):
             without it the linear law saturates the motors after large deviations (initial offset, gusts) and crashes.
             LQI uses conditional integration (no integration on an axis while its error is clipped) as anti wind-up.
Student    : RMA phase-2 policy = frozen teacher actor + adaptation module that estimates the privileged vector
             (wind force in the body frame, mass/thrust/motor/delay parameters) from the last H observations.
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn as nn

from hover_rl.algos import ALGOS, PPO, mlp
from isaaclab.utils.math import quat_rotate, quat_rotate_inverse

GAIN_NAMES = ["kp_xy", "kd_xy", "ki_xy", "kp_z", "kd_z", "ki_z", "k_att", "k_rate"]
# log-uniform search ranges used by tune_pid.py
GAIN_RANGES = {"kp_xy": (0.5, 100.0), "kd_xy": (0.3, 30.0), "ki_xy": (0.01, 40.0), "kp_z": (0.5, 100.0),
               "kd_z": (0.3, 30.0), "ki_z": (0.01, 40.0), "k_att": (2.0, 80.0), "k_rate": (3.0, 80.0)}
G0 = 9.81
# LQR/LQI: Bryson's-rule maxima (state deviations, input magnitudes), log-uniform search ranges
LQR_NAMES = ["pos_xy", "pos_z", "vel_xy", "vel_z", "att", "rate", "rate_yaw", "thrust", "torque", "torque_yaw",
             "err_clip"]
LQR_RANGES = {"err_clip": (0.02, 2.0), "pos_xy": (0.003, 0.5), "pos_z": (0.003, 0.5), "vel_xy": (0.01, 3.0), "vel_z": (0.01, 3.0),
              "att": (0.01, 1.0), "rate": (0.05, 20.0), "rate_yaw": (0.05, 20.0), "thrust": (0.005, 0.5),
              "torque": (1e-5, 1e-2), "torque_yaw": (1e-6, 1e-2)}
# physically reasonable starting point for the search (5 cm, 0.3 m/s, 0.2 rad, 2 rad/s, 0.1 N, 1e-3 N m)
LQR_GUESS = {"pos_xy": 0.05, "pos_z": 0.05, "vel_xy": 0.3, "vel_z": 0.3, "att": 0.2, "rate": 2.0, "rate_yaw": 2.0,
             "thrust": 0.1, "torque": 1e-3, "torque_yaw": 1e-4, "int_xy": 0.02, "int_z": 0.02, "err_clip": 0.3}
LQI_NAMES = LQR_NAMES + ["int_xy", "int_z"]
LQI_RANGES = dict(LQR_RANGES, int_xy=(0.0005, 0.5), int_z=(0.0005, 0.5))


class CascadePID:
    name = "pid"

    def __init__(self, env, gains: torch.Tensor, max_tilt_deg: float | None = None, i_limit: float = 6.0):
        self.env = env
        self.g = gains.to(env.device).expand(env.num_envs, len(GAIN_NAMES)).clone()
        if max_tilt_deg is None:  # v1 motors only reach 1.21 x weight -> small tilt budget
            max_tilt_deg = 25.0 if env.cfg.action_mode == "rpm_v1" else 40.0
        self.max_tilt = math.radians(max_tilt_deg)
        self.i_limit = i_limit  # m/s^2, clamp on the integral term (anti wind-up)
        self.I = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, mask=None):
        if mask is None:
            self.I.zero_()
        else:
            self.I[mask] = 0.0

    @torch.no_grad()
    def act(self, obs=None, deterministic=True):
        e, g = self.env, self.g
        m = e.meas
        n, dev = e.num_envs, e.device
        p = m["pos"] - e.target_local
        v, q = m["vel_w"], m["quat"]
        kp = torch.stack([g[:, 0], g[:, 0], g[:, 3]], -1)
        kd = torch.stack([g[:, 1], g[:, 1], g[:, 4]], -1)
        ki = torch.stack([g[:, 2], g[:, 2], g[:, 5]], -1)
        self.I = self.I + p * e.step_dt
        self.I = torch.maximum(torch.minimum(self.I, self.i_limit / ki), -self.i_limit / ki)  # |ki*I| <= limit
        a = -kp * p - kd * v - ki * self.I
        lim = G0 * math.tan(self.max_tilt)
        nxy = a[:, :2].norm(dim=-1, keepdim=True)
        a = torch.cat([a[:, :2] * torch.clamp(lim / nxy.clamp(min=1e-6), max=1.0), a[:, 2:].clamp(-0.8 * G0, 0.8 * G0)], -1)
        F = e.robot_mass * (a + torch.tensor([0.0, 0.0, G0], device=dev))
        ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        b3 = quat_rotate(q, ez)
        T = (F * b3).sum(-1)
        b3d = F / F.norm(dim=-1, keepdim=True)
        e_att = quat_rotate_inverse(q, torch.cross(b3, b3d, dim=-1))
        w_des = torch.zeros(n, 3, device=dev)
        w_des[:, :2] = g[:, 6:7] * e_att[:, :2]
        w_des = torch.maximum(torch.minimum(w_des, e.max_rate), -e.max_rate)
        mode = e.cfg.action_mode
        if mode == "ctbr":
            return torch.cat([((T / (4.0 * e.f_max)) * 2.0 - 1.0).unsqueeze(-1), w_des / e.max_rate], -1).clamp(-1, 1)
        tau = e.J * g[:, 7:8] * (w_des - m["gyro"])
        return motor_command(e, T, tau)


def motor_command(e, T, tau):
    """Collective thrust + body torques -> per-motor thrust (mixer) -> saturated motor command in [-1, 1]."""
    f = torch.cat([T.unsqueeze(-1), tau], -1) @ e.mix_inv.T
    if e.cfg.action_mode == "rpm_v1":
        from hover_rl.env import MAX_RPM, MIN_RPM
        w_lo, w_hi = MIN_RPM * 2 * math.pi / 60, MAX_RPM * 2 * math.pi / 60
        f = f.clamp(e.k_f * w_lo**2, e.k_f * w_hi**2)
        rpm = torch.sqrt(f / e.k_f) * 60.0 / (2 * math.pi)
        return ((rpm - MIN_RPM) / (MAX_RPM - MIN_RPM) * 2.0 - 1.0).clamp(-1, 1)
    if e.cfg.action_mode != "motor":
        raise ValueError("motor_command needs action_mode rpm_v1 or motor")
    return (f.clamp(0, e.f_max) / e.f_max * 2.0 - 1.0).clamp(-1, 1)


def lqr_gains(params: torch.Tensor, integral: bool, mass: float, J, dt: float) -> torch.Tensor:
    """Bryson maxima (k, P) -> discrete LQR gains (k, 4, nx). State: [x, y, z, vx, vy, vz, roll, pitch, p, q, r]
    (+ [ix, iy, iz] for LQI) in the heading frame; input: [dT, tau_x, tau_y, tau_z]."""
    import numpy as np
    import scipy.linalg as sla

    Jx, Jy, Jz = (float(v) for v in J)
    nx = 14 if integral else 11
    A = np.zeros((nx, nx))
    B = np.zeros((nx, 4))
    A[0:3, 3:6] = np.eye(3)
    A[3, 7], A[4, 6] = G0, -G0  # small-angle: ax = g*pitch, ay = -g*roll
    A[6, 8], A[7, 9] = 1.0, 1.0
    B[5, 0] = 1.0 / mass
    B[8, 1], B[9, 2], B[10, 3] = 1.0 / Jx, 1.0 / Jy, 1.0 / Jz
    if integral:
        A[11:14, 0:3] = np.eye(3)
    M = np.zeros((nx + 4, nx + 4))
    M[:nx, :nx], M[:nx, nx:] = A * dt, B * dt
    E = sla.expm(M)
    Ad, Bd = E[:nx, :nx], E[:nx, nx:]
    out = []
    for prm in params.cpu().double().numpy():
        d = dict(zip(LQI_NAMES if integral else LQR_NAMES, prm))
        xm = [d["pos_xy"]] * 2 + [d["pos_z"]] + [d["vel_xy"]] * 2 + [d["vel_z"]] + [d["att"]] * 2 + [d["rate"]] * 2 \
            + [d["rate_yaw"]]
        if integral:
            xm += [d["int_xy"]] * 2 + [d["int_z"]]
        # solve in Bryson-normalised coordinates (s' = s / x_max, u' = u / u_max -> Q = R = I): well conditioned
        Sx = np.diag(np.array(xm))
        Su = np.diag(np.array([d["thrust"], d["torque"], d["torque"], d["torque_yaw"]]))
        An = np.linalg.solve(Sx, Ad @ Sx)
        Bn = np.linalg.solve(Sx, Bd @ Su)
        try:
            P = sla.solve_discrete_are(An, Bn, np.eye(nx), np.eye(4))
            Kn = np.linalg.solve(np.eye(4) + Bn.T @ P @ Bn, Bn.T @ P @ An)
            K = Su @ Kn @ np.linalg.inv(Sx)
            if not np.all(np.isfinite(K)):
                raise ValueError("non-finite gain")
        except (ValueError, np.linalg.LinAlgError):
            K = np.zeros((4, nx))  # infeasible candidate -> open loop (scores as a crash during tuning)
        out.append(K)
    return torch.tensor(np.stack(out), dtype=torch.float32)


class LQR:
    """Full-state LQR (integral=False) or LQI (integral=True); one gain matrix per env (used for tuning)."""

    def __init__(self, env, params: torch.Tensor, integral: bool = False, i_limit: float = 0.5):
        self.env, self.integral, self.i_limit = env, integral, i_limit
        self.name = "lqi" if integral else "lqr"
        names = LQI_NAMES if integral else LQR_NAMES
        params = params.reshape(-1, len(names))
        # identical parameter rows share one Riccati solve
        self.err_clip = params[:, names.index("err_clip")].to(env.device).expand(env.num_envs).unsqueeze(-1).clone() \
            if params.shape[0] == 1 else params[:, names.index("err_clip")].to(env.device).unsqueeze(-1)
        uniq, inv = torch.unique(params, dim=0, return_inverse=True)
        K = lqr_gains(uniq, integral, env.robot_mass, env.J.cpu(), env.step_dt).to(env.device)
        self.K = K[inv].expand(env.num_envs, *K.shape[1:]) if len(uniq) == 1 or params.shape[0] == 1 else K[inv]
        self.I = torch.zeros(env.num_envs, 3, device=env.device)

    def reset(self, mask=None):
        if mask is None:
            self.I.zero_()
        else:
            self.I[mask] = 0.0

    @torch.no_grad()
    def act(self, obs=None, deterministic=True):
        e = self.env
        m = e.meas
        q = m["quat"]
        w, x, y, z = q.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        cy, sy = torch.cos(yaw), torch.sin(yaw)

        def to_h(v):  # world -> heading frame (rotate by -yaw about z)
            return torch.stack([cy * v[:, 0] + sy * v[:, 1], -sy * v[:, 0] + cy * v[:, 1], v[:, 2]], -1)

        qh_inv = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), -torch.sin(yaw / 2)], -1)
        from isaaclab.utils.math import quat_mul
        qt = quat_mul(qh_inv, q)  # tilt part, ~[1, roll/2, pitch/2, 0]
        sgn = torch.sign(qt[:, :1]).clamp(min=0) * 2 - 1
        qt = qt * sgn
        roll, pitch = 2 * qt[:, 1], 2 * qt[:, 2]
        p_raw = m["pos"] - e.target_local
        p = torch.maximum(torch.minimum(p_raw, self.err_clip), -self.err_clip)
        parts = [to_h(p), to_h(m["vel_w"]), roll.unsqueeze(-1), pitch.unsqueeze(-1), m["gyro"]]
        if self.integral:
            # conditional integration (anti wind-up): an axis integrates only while its error is not clipped
            active = (p_raw.abs() < self.err_clip).float()
            self.I = (self.I + active * p * e.step_dt).clamp(-self.i_limit, self.i_limit)
            parts.append(to_h(self.I))
        s = torch.cat(parts, -1)
        u = -(self.K @ s.unsqueeze(-1)).squeeze(-1)
        T = e.robot_mass * G0 + u[:, 0]
        return motor_command(e, T, u[:, 1:])


def gains_from_json(path) -> torch.Tensor:
    d = json.load(open(path))
    names = {"pid": GAIN_NAMES, "lqr": LQR_NAMES, "lqi": LQI_NAMES}[d.get("ctrl", "pid")]
    return torch.tensor([d["gains"][k] for k in names], dtype=torch.float32)


def make_classical(kind, env, params):
    if kind == "pid":
        return CascadePID(env, params)
    return LQR(env, params, integral=(kind == "lqi"))


class Student(nn.Module):
    name = "student"

    def __init__(self, teacher: PPO, base_dim: int, priv_dim: int = 7, hist: int = 50):
        super().__init__()
        self.teacher, self.base_dim, self.priv_dim, self.hist = teacher, base_dim, priv_dim, hist
        dev = teacher.device
        self.adapter = mlp(hist * base_dim, priv_dim, (256, 128)).to(dev)
        self.buf = None
        self._refill = None

    def norm_base(self, ob):
        nm = self.teacher.norm
        return ((ob - nm.mean[: self.base_dim]) / torch.sqrt(nm.var[: self.base_dim] + 1e-8)).clamp(-10, 10)

    def norm_priv(self, e):
        nm = self.teacher.norm
        return (e - nm.mean[self.base_dim:]) / torch.sqrt(nm.var[self.base_dim:] + 1e-8)

    def reset(self, mask=None):
        if self.buf is None:
            return
        if mask is None:
            self._refill[:] = True
        else:
            self._refill |= mask

    def push(self, obs):
        nb = self.norm_base(obs[:, : self.base_dim])
        if self.buf is None or self.buf.shape[0] != nb.shape[0]:
            self.buf = nb.unsqueeze(1).repeat(1, self.hist, 1)
            self._refill = torch.zeros(nb.shape[0], dtype=torch.bool, device=nb.device)
        if self._refill.any():
            self.buf[self._refill] = nb[self._refill].unsqueeze(1)
            self._refill[:] = False
        self.buf = torch.cat([self.buf[:, 1:], nb.unsqueeze(1)], 1)
        return self.buf

    def estimate(self, hist):
        return self.adapter(hist.reshape(hist.shape[0], -1))

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        self.e_hat = self.estimate(self.push(obs))
        full = torch.cat([obs[:, : self.base_dim], self.e_hat], -1)
        return self.teacher.act(full, deterministic=True)

    def save(self, path, meta=None):
        torch.save({"teacher": self.teacher.state_dict(), "adapter": self.adapter.state_dict(), "base_dim": self.base_dim,
                    "priv_dim": self.priv_dim, "hist": self.hist, "meta": meta or {}}, path)

    @staticmethod
    def load(path, num_envs, device):
        sd = torch.load(path, map_location=device)
        t = PPO(sd["base_dim"] + sd["priv_dim"], 4, num_envs, device)
        t.load_state_dict(sd["teacher"])
        t.norm.frozen = True
        s = Student(t, sd["base_dim"], sd["priv_dim"], sd["hist"])
        s.adapter.load_state_dict(sd["adapter"])
        return s


class Sliced:
    """Deterministic wrapper for an RL agent that sees the first `dim` observation entries."""

    def __init__(self, agent, dim):
        self.agent, self.dim = agent, dim

    def reset(self, mask=None):
        pass

    @torch.no_grad()
    def act(self, obs, deterministic=True):
        return self.agent.act(obs[:, : self.dim], deterministic=True)


def load_controller(kind: str, path: str, env, obs_dim: int):
    """kind: ppo | td3 | dqn (obs_dim = what the agent was trained on) | pid | lqr | lqi (json) | student."""
    if kind in ("pid", "lqr", "lqi"):
        return make_classical(kind, env, gains_from_json(path))
    if kind == "student":
        return Student.load(path, env.num_envs, env.device)
    kw = {"capacity": 1} if kind in ("td3", "dqn") else {}  # no replay buffer needed for evaluation
    agent = ALGOS[kind](obs_dim, 4, env.num_envs, env.device, **kw)
    agent.load_state_dict(torch.load(path, map_location=env.device))
    agent.norm.frozen = True
    return Sliced(agent, obs_dim)
