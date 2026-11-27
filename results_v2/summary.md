v2 protocol: 512 episodes/model, eval seed 1000, 12 s episodes, base wind U[0, 35] % of the nominal weight (0.277 N), gust +40 % for 0.5 s at 6 s; motor lag 20-50 ms, 1-2 step (20-40 ms) delay, mass/thrust ±10 %, sensor noise; errors are TRUE position errors. Mean ± std over 3 seeds (RL: training seeds; PID: tuning seeds).

| Controller | Position error (cm) | Gust peak error (cm) | Recovery time (s) | Recovered | Success | Crash |
|---|---|---|---|---|---|---|
| Cascaded PID (CTBR, onboard rate loop) | 1.66 ± 0.17 | 21.0 ± 0.4 | 1.58 ± 0.04 | 99.8% | 99.9 ± 0.2% | 0.1% |
| Cascaded PID (motor commands) | 1.73 ± 0.17 | 23.4 ± 0.6 | 1.94 ± 0.19 | 99.8% | 100.0 ± 0.0% | 0.0% |
| LQR (motor commands) | 14.72 ± 1.61 | 33.5 ± 0.5 | 1.81 ± 0.30 | 20.4% | 35.3 ± 4.3% | 0.8% |
| LQI (motor commands) | 2.27 ± 0.57 | 27.0 ± 0.3 | 1.76 ± 0.05 | 84.8% | 84.8 ± 5.3% | 14.4% |
| PPO, motor commands | 2.83 ± 0.54 | 11.6 ± 0.2 | 1.35 ± 0.06 | 98.6% | 99.5 ± 0.2% | 0.5% |
| PPO, CTBR | 2.15 ± 0.13 | 10.6 ± 0.3 | 1.09 ± 0.04 | 99.3% | 100.0 ± 0.0% | 0.0% |
| TD3 (v1 hparams), CTBR | 9.90 ± 1.40 | 19.7 ± 0.6 | 1.91 ± 0.21 | 16.5% | 54.4 ± 14.2% | 0.0% |
| TD3 (tuned), CTBR | 4.39 ± 0.65 | 13.0 ± 0.5 | 1.95 ± 0.33 | 83.9% | 99.9 ± 0.2% | 0.0% |
| RMA student (CTBR, estimates wind) | 0.90 ± 0.05 | 9.2 ± 0.2 | 0.84 ± 0.06 | 100.0% | 100.0 ± 0.0% | 0.0% |
| RMA teacher (sees true wind; upper bound) | 0.74 ± 0.06 | 7.7 ± 0.1 | 0.80 ± 0.03 | 100.0% | 100.0 ± 0.0% | 0.0% |

Success rate / crash rate with the base wind fixed beyond the training range (same +40 % gust):

| Controller | 40 % wt | 50 % wt | 60 % wt | 70 % wt | 80 % wt |
|---|---|---|---|---|---|
| PID-CTBR | 99.9% / 0.1% | 99.9% / 0.1% | 99.3% / 0.6% | 82.0% / 15.6% | 29.6% / 64.3% |
| PID-motor | 100.0% / 0.0% | 99.7% / 0.0% | 99.9% / 0.1% | 99.0% / 0.8% | 93.4% / 2.4% |
| LQR | 0.0% / 1.6% | 0.0% / 10.2% | 0.0% / 30.8% | 0.0% / 78.3% | 0.0% / 98.7% |
| LQI | 59.8% / 32.8% | 28.8% / 67.6% | 10.2% / 88.8% | 0.3% / 99.5% | 0.0% / 100.0% |
| PPO-motor | 97.7% / 2.3% | 88.5% / 10.1% | 56.0% / 21.3% | 9.6% / 46.3% | 0.8% / 68.8% |
| PPO-CTBR | 99.8% / 0.2% | 98.8% / 0.6% | 83.9% / 1.2% | 45.0% / 4.5% | 12.0% / 13.5% |
| TD3 (v1 hparams) | 0.0% / 0.1% | 0.0% / 0.5% | 0.0% / 0.8% | 0.0% / 5.3% | 0.0% / 17.6% |
| TD3 (tuned) | 98.4% / 0.0% | 68.9% / 0.2% | 14.9% / 0.4% | 0.3% / 2.5% | 0.0% / 15.2% |
| RMA student | 100.0% / 0.0% | 99.9% / 0.0% | 99.7% / 0.1% | 89.9% / 1.0% | 54.3% / 5.7% |
| Teacher (privileged) | 100.0% / 0.0% | 100.0% / 0.0% | 100.0% / 0.0% | 93.9% / 0.0% | 70.2% / 1.3% |

- PPO-CTBR vs PPO-motor: position error -24%, gust peak -8%, recovery time -19% (negative = PPO-CTBR better); success beyond range 40%: 99.8 vs 97.7, 50%: 98.8 vs 88.5, 60%: 83.9 vs 56.0, 70%: 45.0 vs 9.6, 80%: 12.0 vs 0.8
- PPO-CTBR vs PID-CTBR: position error +29%, gust peak -49%, recovery time -31% (negative = PPO-CTBR better); success beyond range 40%: 99.8 vs 99.9, 50%: 98.8 vs 99.9, 60%: 83.9 vs 99.3, 70%: 45.0 vs 82.0, 80%: 12.0 vs 29.6
- RMA student vs PPO-CTBR: position error -58%, gust peak -13%, recovery time -23% (negative = RMA student better); success beyond range 40%: 100.0 vs 99.8, 50%: 99.9 vs 98.8, 60%: 99.7 vs 83.9, 70%: 89.9 vs 45.0, 80%: 54.3 vs 12.0
- RMA student vs PID-CTBR: position error -46%, gust peak -56%, recovery time -47% (negative = RMA student better); success beyond range 40%: 100.0 vs 99.9, 50%: 99.9 vs 99.9, 60%: 99.7 vs 99.3, 70%: 89.9 vs 82.0, 80%: 54.3 vs 29.6
- Teacher (privileged) vs RMA student: position error -17%, gust peak -16%, recovery time -5% (negative = Teacher (privileged) better); success beyond range 40%: 100.0 vs 100.0, 50%: 100.0 vs 99.9, 60%: 100.0 vs 99.7, 70%: 93.9 vs 89.9, 80%: 70.2 vs 54.3
- TD3 (v1 hparams) vs PPO-CTBR: position error +359%, gust peak +85%, recovery time +75% (negative = TD3 (v1 hparams) better); success beyond range 40%: 0.0 vs 99.8, 50%: 0.0 vs 98.8, 60%: 0.0 vs 83.9, 70%: 0.0 vs 45.0, 80%: 0.0 vs 12.0
- TD3 (tuned) vs PPO-CTBR: position error +104%, gust peak +22%, recovery time +78% (negative = TD3 (tuned) better); success beyond range 40%: 98.4 vs 99.8, 50%: 68.9 vs 98.8, 60%: 14.9 vs 83.9, 70%: 0.3 vs 45.0, 80%: 0.0 vs 12.0
- TD3 (tuned) vs TD3 (v1 hparams): position error -56%, gust peak -34%, recovery time +2% (negative = TD3 (tuned) better); success beyond range 40%: 98.4 vs 0.0, 50%: 68.9 vs 0.0, 60%: 14.9 vs 0.0, 70%: 0.3 vs 0.0, 80%: 0.0 vs 0.0
- LQR vs PID-motor: position error +749%, gust peak +44%, recovery time -7% (negative = LQR better); success beyond range 40%: 0.0 vs 100.0, 50%: 0.0 vs 99.7, 60%: 0.0 vs 99.9, 70%: 0.0 vs 99.0, 80%: 0.0 vs 93.4
- LQI vs PID-motor: position error +31%, gust peak +15%, recovery time -9% (negative = LQI better); success beyond range 40%: 59.8 vs 100.0, 50%: 28.8 vs 99.7, 60%: 10.2 vs 99.9, 70%: 0.3 vs 99.0, 80%: 0.0 vs 93.4
- LQI vs LQR: position error -85%, gust peak -20%, recovery time -3% (negative = LQI better); success beyond range 40%: 59.8 vs 0.0, 50%: 28.8 vs 0.0, 60%: 10.2 vs 0.0, 70%: 0.3 vs 0.0, 80%: 0.0 vs 0.0
- PID-CTBR vs PID-motor: position error -4%, gust peak -10%, recovery time -19% (negative = PID-CTBR better); success beyond range 40%: 99.9 vs 100.0, 50%: 99.9 vs 99.7, 60%: 99.3 vs 99.9, 70%: 82.0 vs 99.0, 80%: 29.6 vs 93.4
- PPO-motor vs PID-motor: position error +63%, gust peak -50%, recovery time -30% (negative = PPO-motor better); success beyond range 40%: 97.7 vs 100.0, 50%: 88.5 vs 99.7, 60%: 56.0 vs 99.9, 70%: 9.6 vs 99.0, 80%: 0.8 vs 93.4
- PPO-motor vs LQI: position error +25%, gust peak -57%, recovery time -23% (negative = PPO-motor better); success beyond range 40%: 97.7 vs 59.8, 50%: 88.5 vs 28.8, 60%: 56.0 vs 10.2, 70%: 9.6 vs 0.3, 80%: 0.8 vs 0.0

RMA wind estimate vs true wind (horizontal+vertical force, % of weight; mean over students):

| Base wind | RMSE all (t ≥ 1 s) | RMSE steady (2-6 s) | RMSE during gust | magnitude bias (steady) | corr (x,y) |
|---|---|---|---|---|---|
| 30 % wt | 3.1 | 2.0 | 9.3 | -0.2 | 0.994 |
| 60 % wt | 6.5 | 5.2 | 12.7 | -2.7 | 0.994 |
