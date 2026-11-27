"""PPO / TD3 / DQN for vectorised (batched-on-GPU) environments.

All three share: the same observation normaliser, the same 18-D observation and the same 4 motor commands.
DQN needs a discrete action set, so each motor gets 3 levels {-1, 0, +1} -> {4500, 5000, 5500} RPM, i.e.
3^4 = 81 joint actions.
"""

from __future__ import annotations

import copy
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------- utils
class RunningNorm(nn.Module):
    def __init__(self, dim: int, clip: float = 10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip
        self.frozen = False

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if self.frozen:
            return
        b_mean, b_var, b_n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, tot = b_mean - self.mean, self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta**2 * self.count * b_n / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-self.clip, self.clip)


def mlp(i, o, hidden=(256, 256), act=nn.ELU, out_act=None):
    layers, d = [], i
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, o))
    if out_act is not None:
        layers.append(out_act())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------- PPO
class PPO:
    name = "ppo"

    def __init__(self, obs_dim, act_dim, num_envs, device, *, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2,
                 horizon=32, epochs=5, minibatches=4, ent_coef=0.0, vf_coef=1.0, max_grad=1.0,
                 init_std=0.5, desired_kl=0.01):
        self.device, self.num_envs = device, num_envs
        self.norm = RunningNorm(obs_dim).to(device)
        self.actor = mlp(obs_dim, act_dim, (256, 128, 64)).to(device)
        self.critic = mlp(obs_dim, 1, (256, 128, 64)).to(device)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(torch.log(torch.tensor(init_std))), device=device))
        self.opt = torch.optim.Adam([*self.actor.parameters(), *self.critic.parameters(), self.log_std], lr=lr)
        self.lr, self.gamma, self.lam, self.clip = lr, gamma, lam, clip
        self.horizon, self.epochs, self.minibatches = horizon, epochs, minibatches
        self.ent_coef, self.vf_coef, self.max_grad, self.desired_kl = ent_coef, vf_coef, max_grad, desired_kl
        T, N = horizon, num_envs
        self.buf = {k: torch.zeros(T, N, *s, device=device) for k, s in
                    dict(obs=(obs_dim,), act=(act_dim,), logp=(), val=(), rew=(), done=(), mu=(act_dim,)).items()}
        self.t = 0

    def _dist(self, nobs):
        mu = self.actor(nobs)
        return torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu)), mu

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        nobs = self.norm(obs)
        dist, mu = self._dist(nobs)
        if deterministic:
            return mu.clamp(-1, 1)
        a = dist.sample()
        self._cache = (nobs, a, dist.log_prob(a).sum(-1), self.critic(nobs).squeeze(-1), mu)
        return a  # env clamps to [-1, 1]

    def observe(self, obs, rew, terminated, truncated, terminal_obs):
        with torch.no_grad():
            nobs, a, logp, val, mu = self._cache
            # bootstrap time-outs with the value of the pre-reset observation
            if truncated.any():
                v_term = self.critic(self.norm(terminal_obs)).squeeze(-1)
                rew = rew + self.gamma * v_term * truncated.float()
            t = self.t
            self.buf["obs"][t], self.buf["act"][t], self.buf["logp"][t] = nobs, a, logp
            self.buf["val"][t], self.buf["rew"][t], self.buf["mu"][t] = val, rew, mu
            self.buf["done"][t] = (terminated | truncated).float()
            self.norm.update(obs)
        self.t += 1
        if self.t == self.horizon:
            self.t = 0
            return self._update(obs)
        return None

    def _update(self, last_obs):
        b = self.buf
        with torch.no_grad():
            last_v = self.critic(self.norm(last_obs)).squeeze(-1)
            adv = torch.zeros_like(b["rew"])
            gae = torch.zeros(self.num_envs, device=self.device)
            for t in reversed(range(self.horizon)):
                nv = last_v if t == self.horizon - 1 else b["val"][t + 1]
                nd = 1.0 - b["done"][t]
                delta = b["rew"][t] + self.gamma * nv * nd - b["val"][t]
                gae = delta + self.gamma * self.lam * nd * gae
                adv[t] = gae
            ret = adv + b["val"]
        flat = lambda x: x.reshape(-1, *x.shape[2:])  # noqa: E731
        obs, act, logp_old, ret, adv, val_old = map(flat, (b["obs"], b["act"], b["logp"], ret, adv, b["val"]))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        n = obs.shape[0]
        mb = n // self.minibatches
        stats = []
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(self.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                dist, _ = self._dist(obs[idx])
                logp = dist.log_prob(act[idx]).sum(-1)
                ratio = (logp - logp_old[idx]).exp()
                pl = -torch.min(ratio * adv[idx], ratio.clamp(1 - self.clip, 1 + self.clip) * adv[idx]).mean()
                v = self.critic(obs[idx]).squeeze(-1)
                v_clip = val_old[idx] + (v - val_old[idx]).clamp(-self.clip, self.clip)
                vl = torch.max((v - ret[idx]) ** 2, (v_clip - ret[idx]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pl + self.vf_coef * vl - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters(), self.log_std],
                                         self.max_grad)
                self.opt.step()
                with torch.no_grad():  # adaptive learning rate on approx. KL (as in rsl_rl)
                    kl = (logp_old[idx] - logp).mean().abs().item()
                    if kl > 2 * self.desired_kl:
                        self.lr = max(1e-5, self.lr / 1.5)
                    elif kl < self.desired_kl / 2:
                        self.lr = min(1e-3, self.lr * 1.5)
                    for g in self.opt.param_groups:
                        g["lr"] = self.lr
                stats.append((pl.item(), vl.item()))
        return {"policy_loss": sum(s[0] for s in stats) / len(stats), "value_loss": sum(s[1] for s in stats) / len(stats),
                "lr": self.lr, "std": self.log_std.exp().mean().item()}

    def state_dict(self):
        return {"norm": self.norm.state_dict(), "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "log_std": self.log_std.detach()}

    def load_state_dict(self, sd):
        self.norm.load_state_dict(sd["norm"])
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.log_std.data.copy_(sd["log_std"])


# ---------------------------------------------------------------------- replay
class Replay:
    def __init__(self, obs_dim, act_dim, capacity, device, act_dtype=torch.float32):
        self.o = torch.zeros(capacity, obs_dim, device=device)
        self.a = torch.zeros(capacity, act_dim, device=device, dtype=act_dtype)
        self.r = torch.zeros(capacity, device=device)
        self.o2 = torch.zeros(capacity, obs_dim, device=device)
        self.d = torch.zeros(capacity, device=device)
        self.cap, self.ptr, self.size, self.device = capacity, 0, 0, device

    def add(self, o, a, r, o2, d):
        n = o.shape[0]
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.cap
        self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx] = o, a, r, o2, d
        self.ptr = (self.ptr + n) % self.cap
        self.size = min(self.size + n, self.cap)

    def sample(self, b):
        i = torch.randint(0, self.size, (b,), device=self.device)
        return self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i]


def _soft_update(target, src, tau):
    with torch.no_grad():
        for tp, p in zip(target.parameters(), src.parameters()):
            tp.mul_(1 - tau).add_(p, alpha=tau)


# ---------------------------------------------------------------------- TD3
class TD3:
    name = "td3"

    def __init__(self, obs_dim, act_dim, num_envs, device, *, lr=3e-4, gamma=0.99, tau=0.005, batch=4096,
                 capacity=2_000_000, grad_steps=8, policy_delay=2, expl_noise=0.2, target_noise=0.2,
                 noise_clip=0.5, warmup_steps=50, n_step=1, reward_scale=1.0, hidden=(256, 256)):
        # Defaults = the v1 hyper-parameters. n_step / reward_scale / hidden were added for the v2 re-tuning; with the
        # defaults the update is identical to the original implementation.
        self.device, self.num_envs = device, num_envs
        self.norm = RunningNorm(obs_dim).to(device)
        hidden = tuple(hidden)
        self.actor = mlp(obs_dim, act_dim, hidden, out_act=nn.Tanh).to(device)
        self.q1, self.q2 = mlp(obs_dim + act_dim, 1, hidden).to(device), mlp(obs_dim + act_dim, 1, hidden).to(device)
        self.actor_t, self.q1_t, self.q2_t = map(copy.deepcopy, (self.actor, self.q1, self.q2))
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam([*self.q1.parameters(), *self.q2.parameters()], lr=lr)
        self.rb = Replay(obs_dim, act_dim, capacity, device)
        self.n_step, self.reward_scale = n_step, reward_scale
        if n_step > 1:
            self.rb.g = torch.zeros(capacity, device=device)  # per-transition bootstrap discount gamma^k
            self.window = []
        self.gamma, self.tau, self.batch, self.grad_steps = gamma, tau, batch, grad_steps
        self.policy_delay, self.expl_noise, self.target_noise, self.noise_clip = policy_delay, expl_noise, target_noise, noise_clip
        self.warmup_steps, self.vec_steps, self.n_updates = warmup_steps, 0, 0
        self.act_dim = act_dim

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a = self.actor(self.norm(obs))
        if deterministic:
            return a
        if self.vec_steps < self.warmup_steps:
            a = torch.empty_like(a).uniform_(-1, 1)
        else:
            a = (a + self.expl_noise * torch.randn_like(a)).clamp(-1, 1)
        self._cache = (obs, a)
        return a

    @torch.no_grad()
    def _push_oldest(self):
        o0, a0 = self.window[0][0], self.window[0][1]
        n = o0.shape[0]
        R = torch.zeros(n, device=self.device)
        alive = torch.ones(n, dtype=torch.bool, device=self.device)
        nxt, done, g = self.window[0][5], self.window[0][3].float(), torch.full((n,), self.gamma, device=self.device)
        disc = 1.0
        for k, (_, _, r, term, ended, o2) in enumerate(self.window):
            R = R + alive.float() * disc * r
            nxt = torch.where(alive.unsqueeze(-1), o2, nxt)
            done = torch.where(alive, term.float(), done)
            g = torch.where(alive, torch.full_like(g, self.gamma ** (k + 1)), g)
            disc *= self.gamma
            alive = alive & ~ended
        ptr = self.rb.ptr
        self.rb.add(o0, a0, R, nxt, done)
        idx = (torch.arange(n, device=self.device) + ptr) % self.rb.cap
        self.rb.g[idx] = g
        self.window.pop(0)

    def observe(self, obs, rew, terminated, truncated, terminal_obs):
        o, a = self._cache
        # next observation for transitions that ended is the pre-reset observation
        ended = (terminated | truncated).unsqueeze(-1)
        o2 = torch.where(ended, terminal_obs, obs)
        if self.n_step > 1:
            self.window.append((o, a, rew * self.reward_scale, terminated, ended.squeeze(-1), o2))
            if len(self.window) == self.n_step:
                self._push_oldest()
        else:
            self.rb.add(o, a, rew * self.reward_scale if self.reward_scale != 1.0 else rew, o2, terminated.float())
        self.norm.update(o)
        self.vec_steps += 1
        if self.vec_steps < self.warmup_steps or self.rb.size == 0:
            return None
        return self._update()

    def _update(self):
        ql, al = 0.0, 0.0
        for _ in range(self.grad_steps):
            if self.n_step > 1:
                i = torch.randint(0, self.rb.size, (self.batch,), device=self.device)
                o, a, r, o2, d, g = self.rb.o[i], self.rb.a[i], self.rb.r[i], self.rb.o2[i], self.rb.d[i], self.rb.g[i]
            else:
                o, a, r, o2, d = self.rb.sample(self.batch)
                g = self.gamma
            no, no2 = self.norm(o), self.norm(o2)
            with torch.no_grad():
                noise = (self.target_noise * torch.randn_like(a)).clamp(-self.noise_clip, self.noise_clip)
                a2 = (self.actor_t(no2) + noise).clamp(-1, 1)
                x2 = torch.cat([no2, a2], -1)
                y = r + g * (1 - d) * torch.min(self.q1_t(x2), self.q2_t(x2)).squeeze(-1)
            x = torch.cat([no, a], -1)
            loss_q = F.mse_loss(self.q1(x).squeeze(-1), y) + F.mse_loss(self.q2(x).squeeze(-1), y)
            self.opt_q.zero_grad()
            loss_q.backward()
            self.opt_q.step()
            ql += loss_q.item()
            self.n_updates += 1
            if self.n_updates % self.policy_delay == 0:
                loss_a = -self.q1(torch.cat([no, self.actor(no)], -1)).mean()
                self.opt_a.zero_grad()
                loss_a.backward()
                self.opt_a.step()
                al += loss_a.item()
                for t, s in ((self.actor_t, self.actor), (self.q1_t, self.q1), (self.q2_t, self.q2)):
                    _soft_update(t, s, self.tau)
        return {"q_loss": ql / self.grad_steps, "actor_loss": al / max(1, self.grad_steps // self.policy_delay)}

    def state_dict(self):
        return {"norm": self.norm.state_dict(), "actor": self.actor.state_dict(), "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict()}

    def load_state_dict(self, sd):
        self.norm.load_state_dict(sd["norm"])
        self.actor.load_state_dict(sd["actor"])
        self.q1.load_state_dict(sd["q1"])
        self.q2.load_state_dict(sd["q2"])


# ---------------------------------------------------------------------- DQN
DQN_LEVELS = (-1.0, 0.0, 1.0)


class DQN:
    """Double DQN with a target network and n-step returns over 3^4 = 81 discrete motor-level combinations."""

    name = "dqn"

    def __init__(self, obs_dim, act_dim, num_envs, device, *, lr=1e-4, gamma=0.99, tau=0.001, batch=4096,
                 capacity=2_000_000, grad_steps=2, n_step=3, eps_start=1.0, eps_end=0.05, eps_decay_steps=6000,
                 warmup_steps=50):
        self.device, self.num_envs = device, num_envs
        self.table = torch.tensor(list(itertools.product(DQN_LEVELS, repeat=act_dim)), device=device)
        self.n_actions = self.table.shape[0]
        self.norm = RunningNorm(obs_dim).to(device)
        self.q = mlp(obs_dim, self.n_actions, (256, 256)).to(device)
        self.q_t = copy.deepcopy(self.q)
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.rb = Replay(obs_dim, 1, capacity, device, act_dtype=torch.long)
        self.rb.g = torch.zeros(capacity, device=device)  # per-transition bootstrap discount gamma^k
        self.gamma, self.tau, self.batch, self.grad_steps, self.n_step = gamma, tau, batch, grad_steps, n_step
        self.eps_start, self.eps_end, self.eps_decay_steps = eps_start, eps_end, eps_decay_steps
        self.warmup_steps, self.vec_steps = warmup_steps, 0
        self.window = []  # last n vector transitions (o, a, r, term, ended, o2)

    @property
    def eps(self):
        f = min(1.0, self.vec_steps / self.eps_decay_steps)
        return self.eps_start + f * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        idx = self.q(self.norm(obs)).argmax(-1)
        if not deterministic:
            rnd = torch.rand(idx.shape[0], device=self.device) < self.eps
            idx = torch.where(rnd, torch.randint_like(idx, self.n_actions), idx)
            self._cache = (obs, idx)
        return self.table[idx]

    @torch.no_grad()
    def _push_oldest(self):
        o0, a0 = self.window[0][0], self.window[0][1]
        n = o0.shape[0]
        R = torch.zeros(n, device=self.device)
        alive = torch.ones(n, dtype=torch.bool, device=self.device)
        nxt, done, g = self.window[0][5], self.window[0][3].float(), torch.full((n,), self.gamma, device=self.device)
        disc = 1.0
        for k, (_, _, r, term, ended, o2) in enumerate(self.window):
            R = R + alive.float() * disc * r
            nxt = torch.where(alive.unsqueeze(-1), o2, nxt)
            done = torch.where(alive, term.float(), done)
            g = torch.where(alive, torch.full_like(g, self.gamma ** (k + 1)), g)
            disc *= self.gamma
            alive = alive & ~ended
        ptr = self.rb.ptr
        self.rb.add(o0, a0.unsqueeze(-1), R, nxt, done)
        idx = (torch.arange(n, device=self.device) + ptr) % self.rb.cap
        self.rb.g[idx] = g
        self.window.pop(0)

    def observe(self, obs, rew, terminated, truncated, terminal_obs):
        o, idx = self._cache
        ended = terminated | truncated
        o2 = torch.where(ended.unsqueeze(-1), terminal_obs, obs)
        self.window.append((o, idx, rew, terminated, ended, o2))
        if len(self.window) == self.n_step:
            self._push_oldest()
        self.norm.update(o)
        self.vec_steps += 1
        if self.vec_steps < self.warmup_steps:
            return None
        loss_sum = 0.0
        for _ in range(self.grad_steps):
            i = torch.randint(0, self.rb.size, (self.batch,), device=self.device)
            o, a, r, o2, d, g = self.rb.o[i], self.rb.a[i], self.rb.r[i], self.rb.o2[i], self.rb.d[i], self.rb.g[i]
            no, no2 = self.norm(o), self.norm(o2)
            with torch.no_grad():
                a2 = self.q(no2).argmax(-1, keepdim=True)
                y = r + g * (1 - d) * self.q_t(no2).gather(-1, a2).squeeze(-1)
            q = self.q(no).gather(-1, a).squeeze(-1)
            loss = F.smooth_l1_loss(q, y)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
            self.opt.step()
            _soft_update(self.q_t, self.q, self.tau)
            loss_sum += loss.item()
        return {"q_loss": loss_sum / self.grad_steps, "eps": self.eps}

    def state_dict(self):
        return {"norm": self.norm.state_dict(), "q": self.q.state_dict()}

    def load_state_dict(self, sd):
        self.norm.load_state_dict(sd["norm"])
        self.q.load_state_dict(sd["q"])


ALGOS = {"ppo": PPO, "td3": TD3, "dqn": DQN}
