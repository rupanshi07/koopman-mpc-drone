# Koopman-Operator-Based MPC for Payload-Carrying / Wind-Disturbed Drones

## Pipeline
1. Fly and log data (nominal, windy, payload-added conditions)
2. Train Koopman model (EDMD, degree-2 monomial lifting)
3. MPC controller (QP-based, linear in lifted space)
4. Send commands + environment selector (nominal/windy/payload)
5. Test under disturbance (payload mid-flight, wind gusts)
6. Compare to baseline (PID/LQR)

## Novelty
Most prior Koopman-MPC drone work (e.g. Oh et al. 2024, IEEE Access)
stops at step 4/5 without a classical-control baseline comparison.
This project adds:
- Payload disturbance testing (not present in base paper)
- A genuine PID/LQR baseline comparison (base paper only ablates its own selector)
- Full metric suite: RMSE, settling time, control effort, transient deviation

## Base paper
Oh, Y., Lee, M.H., Moon, J. "Koopman-Based Control System for Quadrotors
in Noisy Environments." IEEE Access, 2024.
