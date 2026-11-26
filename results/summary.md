Protocol: 512 episodes/model, eval seed 1000, 12 s episodes, base wind U[0.02, 0.08] N (drone weight 0.277 N), gust pulse +0.10 N for 0.5 s at 6 s. Recovery band 5 cm held 0.5 s; success = no crash and last-2 s error < 10 cm. Mean ± std across 3 training seeds.

| Algorithm | Position error (cm) | Gust peak error (cm) | Recovery time (s) | Success rate | Crash rate |
|---|---|---|---|---|---|
| PPO | 1.13 ± 0.24 | 7.1 ± 0.3 | 0.54 ± 0.04 | 99.7 ± 0.2% | 0.3% |
| TD3 | 1.84 ± 0.24 | 7.5 ± 0.2 | 0.62 ± 0.04 | 100.0 ± 0.0% | 0.0% |
| DQN | 0.80 ± 0.14 | 8.4 ± 0.1 | 0.63 ± 0.01 | 100.0 ± 0.0% | 0.0% |

Beyond the training range (base wind fixed, same gust pulse) — success rate / crash rate:

| Algorithm | 0.10 N (36 % weight) | 0.12 N (43 % weight) | 0.14 N (51 % weight) |
|---|---|---|---|
| PPO | 99.0% / 1.0% | 89.5% / 9.2% | 48.8% / 44.1% |
| TD3 | 96.9% / 2.2% | 79.8% / 14.1% | 42.8% / 37.6% |
| DQN | 99.6% / 0.4% | 80.5% / 13.5% | 10.4% / 56.2% |

- PPO relative to DQN: position error +41%, gust peak error -16%, recovery time -14% (negative = PPO better)
- PPO relative to TD3: position error -39%, gust peak error -5%, recovery time -13% (negative = PPO better)
