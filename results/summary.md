Protocol: 512 episodes/model, eval seed 1000, 12 s episodes, base wind U[0.02, 0.08] N (drone weight 0.277 N), gust pulse +0.10 N for 0.5 s at 6 s. Recovery band 5 cm held 0.5 s; success = no crash and last-2 s error < 10 cm. Mean ± std across 3 training seeds (classical controllers: 3 independently tuned parameter sets, tuning seeds 11-13).

| Algorithm | Position error (cm) | Gust peak error (cm) | Recovery time (s) | Success rate | Crash rate |
|---|---|---|---|---|---|
| PPO | 1.13 ± 0.24 | 7.1 ± 0.3 | 0.54 ± 0.04 | 99.7 ± 0.2% | 0.3% |
| TD3 | 1.84 ± 0.24 | 7.5 ± 0.2 | 0.62 ± 0.04 | 100.0 ± 0.0% | 0.0% |
| DQN | 0.80 ± 0.14 | 8.4 ± 0.1 | 0.63 ± 0.01 | 100.0 ± 0.0% | 0.0% |
| Cascaded PID | 0.81 ± 0.05 | 13.2 ± 0.3 | 1.10 ± 0.02 | 99.7 ± 0.4% | 0.3% |
| LQR | 4.65 ± 0.07 | 11.6 ± 0.1 | 1.22 ± 0.01 | 94.3 ± 2.7% | 5.5% |
| LQI (LQR + integral) | 1.06 ± 0.04 | 13.7 ± 0.8 | 1.35 ± 0.23 | 96.5 ± 2.3% | 2.9% |

Beyond the training range (base wind fixed, same gust pulse) — success rate / crash rate:

| Algorithm | 0.10 N (36 % weight) | 0.12 N (43 % weight) | 0.14 N (51 % weight) |
|---|---|---|---|
| PPO | 99.0% / 1.0% | 89.5% / 9.2% | 48.8% / 44.1% |
| TD3 | 96.9% / 2.2% | 79.8% / 14.1% | 42.8% / 37.6% |
| DQN | 99.6% / 0.4% | 80.5% / 13.5% | 10.4% / 56.2% |
| Cascaded PID | 89.8% / 10.2% | 37.2% / 52.4% | 0.1% / 99.6% |
| LQR | 42.3% / 33.6% | 1.8% / 57.6% | 0.0% / 89.3% |
| LQI (LQR + integral) | 35.3% / 34.2% | 12.2% / 76.0% | 6.1% / 89.7% |

- PPO relative to DQN: position error +41%, gust peak error -16%, recovery time -14% (negative = PPO better)
- PPO relative to TD3: position error -39%, gust peak error -5%, recovery time -13% (negative = PPO better)
- PPO relative to Cascaded PID: position error +40%, gust peak error -46%, recovery time -51% (negative = PPO better)
- PPO relative to LQR: position error -76%, gust peak error -39%, recovery time -56% (negative = PPO better)
- PPO relative to LQI (LQR + integral): position error +7%, gust peak error -48%, recovery time -60% (negative = PPO better)
