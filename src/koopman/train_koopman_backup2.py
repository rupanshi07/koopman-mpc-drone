import numpy as np
import sys
from itertools import combinations

CONDITION = sys.argv[1] if len(sys.argv) > 1 else "nominal"
DATA_PATH = f"data/{CONDITION}/flight_log.npz"
SAVE_DIR = f"data/{CONDITION}"

data = np.load(DATA_PATH)
states_full = data["states"]
actions = data["actions"]
states = states_full[:, :16]

def transform_state(x):
    # Replace roll/pitch/yaw (raw radians) with sin/cos pairs.
    # Raw Euler angles jump discontinuously (+pi -> -pi) when the drone
    # rotates past the wraparound point, which a linear model cannot
    # fit. sin/cos are continuous through that crossing.
    pos     = x[..., 0:3]
    quat    = x[..., 3:7]
    rpy     = x[..., 7:10]
    vel     = x[..., 10:13]
    ang_vel = x[..., 13:16]
    rpy_trig = np.concatenate([np.sin(rpy), np.cos(rpy)], axis=-1)
    return np.concatenate([pos, quat, rpy_trig, vel, ang_vel], axis=-1)

X_raw  = states[:-1]
Xp_raw = states[1:]
U = actions[:-1]

X  = transform_state(X_raw)
Xp = transform_state(Xp_raw)

print(f"Loaded {X.shape[0]} transitions for condition: {CONDITION}")
print(f"Transformed state dimension: {X.shape[1]} (raw was {X_raw.shape[1]})")

x_mean = X.mean(axis=0)
x_std  = X.std(axis=0) + 1e-8
u_mean = U.mean(axis=0)
u_std  = U.std(axis=0) + 1e-8

Xn  = (X  - x_mean) / x_std
Xpn = (Xp - x_mean) / x_std
Un  = (U  - u_mean) / u_std

# Cross-term pairs over: quaternion(3-6), sin_rpy(7-9), cos_rpy(10-12),
# velocity(13-15), angular velocity(16-18)
CROSS_IDX = list(range(3, 19))
CROSS_PAIRS = list(combinations(CROSS_IDX, 2))

def lift(x):
    quad = x**2
    cross = np.stack([x[:, i] * x[:, j] for i, j in CROSS_PAIRS], axis=1)
    return np.concatenate([x, quad, cross], axis=1)

Psi_X  = lift(Xn)
Psi_Xp = lift(Xpn)
print(f"Lifted dimension: {Psi_X.shape[1]}")

Z = np.hstack([Psi_X, Un])
reg = 1e-4
ZtZ = Z.T @ Z + reg * np.eye(Z.shape[1])
ZtY = Z.T @ Psi_Xp
AB_T = np.linalg.solve(ZtZ, ZtY)

L = Psi_X.shape[1]
A = AB_T[:L, :].T
B = AB_T[L:, :].T

Psi_pred = Psi_X @ A.T + Un @ B.T
pred_error = np.linalg.norm(Psi_pred - Psi_Xp, axis=1)
rmse = np.sqrt(np.mean(pred_error**2))
print(f"[{CONDITION}] One-step lifted prediction RMSE (normalized space): {rmse:.6f}")

np.savez(f"{SAVE_DIR}/koopman_model.npz",
         A=A, B=B,
         x_mean=x_mean, x_std=x_std,
         u_mean=u_mean, u_std=u_std,
         cross_pairs=np.array(CROSS_PAIRS))
print(f"Saved Koopman model (A: {A.shape}, B: {B.shape}) to {SAVE_DIR}/koopman_model.npz")
