# Final tables (paper)

All numbers: mean ± std over 3 training seeds (RL) or 3 independent tunings on seeds 11-13 (classical).
Evaluation: 512 episodes, evaluation seed 1000, 12 s, gust at 6 s for 0.5 s; success = no crash and mean error over the last 2 s < 10 cm; recovery = time until the error stays < 5 cm for 0.5 s (mean over the episodes that recover; their share in the 'recovered' column).

## Table 1 - v2 (realistic simulator), wind inside the training range (0-35 % of weight), gust +40 %

| Controller | Steady error (cm) | Gust peak (cm) | Recovery (s) | Recovered | Success (%) | Crash (%) |
|---|---|---|---|---|---|---|
| PID (CTBR) | 1.66 ± 0.17 | 21.0 ± 0.4 | 1.58 ± 0.04 | 99.8 ± 0.3 % | 99.9 ± 0.2 | 0.1 ± 0.1 |
| PID (motor) | 1.73 ± 0.17 | 23.4 ± 0.6 | 1.94 ± 0.19 | 99.8 ± 0.3 % | 100.0 ± 0.0 | 0.0 ± 0.0 |
| LQR | 14.72 ± 1.61 | 33.5 ± 0.5 | 1.81 ± 0.30 | 20.4 ± 4.1 % | 35.3 ± 4.3 | 0.8 ± 1.0 |
| LQI | 2.27 ± 0.57 | 27.0 ± 0.3 | 1.76 ± 0.05 | 84.8 ± 5.3 % | 84.8 ± 5.3 | 14.4 ± 4.9 |
| PPO (motor) | 2.83 ± 0.54 | 11.6 ± 0.2 | 1.35 ± 0.06 | 98.6 ± 0.8 % | 99.5 ± 0.2 | 0.5 ± 0.2 |
| PPO (CTBR) | 2.15 ± 0.13 | 10.6 ± 0.3 | 1.09 ± 0.04 | 99.3 ± 0.8 % | 100.0 ± 0.0 | 0.0 ± 0.0 |
| TD3 | 4.39 ± 0.65 | 13.0 ± 0.5 | 1.95 ± 0.33 | 83.9 ± 10.8 % | 99.9 ± 0.2 | 0.0 ± 0.0 |
| RMA (student) | 0.90 ± 0.05 | 9.2 ± 0.2 | 0.84 ± 0.06 | 100.0 ± 0.0 % | 100.0 ± 0.0 | 0.0 ± 0.0 |
| RMA teacher (privileged) | 0.74 ± 0.06 | 7.7 ± 0.1 | 0.80 ± 0.03 | 100.0 ± 0.0 % | 100.0 ± 0.0 | 0.0 ± 0.0 |

## Table 2 - v2 success rate (%) with the steady wind fixed beyond the training range (+40 % gust)

| Controller | 40 % W | 50 % W | 60 % W | 70 % W | 80 % W |
|---|---|---|---|---|---|
| PID (CTBR) | 99.9 ± 0.2 | 99.9 ± 0.1 | 99.3 ± 0.4 | 82.0 ± 3.7 | 29.6 ± 3.1 |
| PID (motor) | 100.0 ± 0.0 | 99.7 ± 0.2 | 99.9 ± 0.1 | 99.0 ± 0.4 | 93.4 ± 4.3 |
| LQR | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| LQI | 59.8 ± 17.9 | 28.8 ± 20.7 | 10.2 ± 8.5 | 0.3 ± 0.3 | 0.0 ± 0.0 |
| PPO (motor) | 97.7 ± 1.2 | 88.5 ± 3.7 | 56.0 ± 4.2 | 9.6 ± 2.0 | 0.8 ± 0.3 |
| PPO (CTBR) | 99.8 ± 0.2 | 98.8 ± 0.3 | 83.9 ± 6.0 | 45.0 ± 15.5 | 12.0 ± 5.6 |
| TD3 | 98.4 ± 0.8 | 68.9 ± 7.9 | 14.9 ± 1.4 | 0.3 ± 0.2 | 0.0 ± 0.0 |
| RMA (student) | 100.0 ± 0.0 | 99.9 ± 0.1 | 99.7 ± 0.4 | 89.9 ± 5.6 | 54.3 ± 20.8 |
| RMA teacher (privileged) | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 93.9 ± 3.4 | 70.2 ± 12.5 |

## Table 3 - v1 (idealised simulator, motor commands), gust +0.10 N (36 % of weight)

| Controller | Steady error (cm) | Gust peak (cm) | Recovery (s) | Success (%) | Success at 0.12 N (%) | Success at 0.14 N (%) |
|---|---|---|---|---|---|---|
| PID | 0.81 ± 0.05 | 13.2 ± 0.3 | 1.10 ± 0.02 | 99.7 ± 0.4 | 37.2 | 0.1 |
| LQR | 4.65 ± 0.07 | 11.6 ± 0.1 | 1.22 ± 0.01 | 94.3 ± 2.7 | 1.8 | 0.0 |
| LQI | 1.06 ± 0.04 | 13.7 ± 0.8 | 1.35 ± 0.23 | 96.5 ± 2.3 | 12.2 | 6.1 |
| PPO | 1.13 ± 0.24 | 7.1 ± 0.3 | 0.54 ± 0.04 | 99.7 ± 0.2 | 89.5 | 48.8 |
| TD3 | 1.84 ± 0.24 | 7.5 ± 0.2 | 0.62 ± 0.04 | 100.0 ± 0.0 | 79.8 | 42.8 |

(DQN, v1 only: see results/summary.csv.)

## Table 4 - TD3 (CTBR) hyper-parameter study: 1 seed (0), 25 M-step runs, same evaluation protocol

| Tag | Setting (changes vs default TD3) | Steps (M) | Steady error (cm) | Gust peak (cm) | Success (%) | Success 30 % W (%) | Static bias at 0 / 30 % W (cm) |
|---|---|---|---|---|---|---|---|
| default | default TD3 (Fujimoto et al.): expl. noise 0.2, target noise 0.2 / clip 0.5, 1-step, lr 3e-4 | 25 | 9.55 | 21.7 | 56.2 | 6.6 | 1.5 / 12.6 |
| noise01 | expl. 0.1, target 0.1 / clip 0.25 | 25 | 12.54 | 26.6 | 35.9 | 0.0 | 4.3 / 16.7 |
| noise005 | expl. 0.05, target 0.05 / clip 0.1 | 25 | 13.18 | 25.9 | 2.1 | 0.0 | 10.2 / 14.3 |
| noise01_n3 | noise01 + 3-step returns | 25 | 4.46 | 14.2 | 100.0 | 99.6 | 2.3 / 5.1 |
| n3_noise02 | v1 noise + 3-step returns | 25 | 4.02 | 13.0 | 100.0 | 99.6 | 1.9 / 5.4 |
| noise01_n5 | noise01 + 5-step returns | 25 | 3.73 | 13.9 | 100.0 | 100.0 | 1.4 / 5.1 |
| **noise01_n3_lr1e4** | noise01_n3 + lr 1e-4 | 25 | 4.08 | 12.5 | 98.8 | 98.8 | 3.5 / 4.7 |
| noise01_n10 | noise01 + 10-step returns | 25 | diverged: critic diverged (Q loss > 1e4 from 3.7 M steps, > 1e13 by 11 M); every episode crashes | | | | |
| noise005_n5 | expl. 0.05, target 0.05 / clip 0.1 + 5-step | 25 | 4.13 | 15.0 | 100.0 | 100.0 | 1.0 / 6.1 |
| noise01_n5_lr1e4 | noise01_n5 + lr 1e-4 | 14 | diverged: returns collapsed at ~4 M steps, Q loss > 1e4 from 5.3 M; stopped at 13.8 M | | | | |

## Table 5 - RMA wind-estimate error (% of weight; mean over 3 students, 512 episodes each)

| Base wind | RMSE steady (2-6 s) | RMSE during gust | Bias (steady) | Correlation |
|---|---|---|---|---|
| 30 % W | 2.0 | 9.3 | -0.2 | 0.994 |
| 60 % W | 5.2 | 12.7 | -2.7 | 0.994 |
