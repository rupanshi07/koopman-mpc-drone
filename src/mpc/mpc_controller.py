import numpy as np
import cvxpy as cp
import sys
from itertools import combinations

CONDITION = sys.argv[1] if len(sys.argv) > 1 else "nominal"
model = np.load(f"data/{CONDITION}/koopman_model.npz")

A = model["A"]
B = model["B"]
x_mean, x_std = model["x_mean"], model["x_std"]
u_mean, u_std = model["u_mean"], model["u_std"]
cross_pairs = [tuple(p) for p in model["cross_pairs"]]

N_STATE = 16
N_LIFTED = A.shape[0]
N_ACTION = 4

def lift(x):
    quad = x**2
    cross = np.array([x[i] * x[j] for i, j in cross_pairs])
    return np.concatenate([x, quad, cross])

def normalize_state(x_raw):
    return (x_raw - x_mean) / x_std

def denormalize_action(u_norm):
    return u_norm * u_std + u_mean

HORIZON = 10
HOVER_RPM = 16000

target_state_raw = np.zeros(N_STATE)
target_state_raw[2] = 0.5
target_state_raw[6] = 1.0
target_lifted = lift(normalize_state(target_state_raw))

_last_u_norm = np.zeros(N_ACTION)  # persists across calls for smoothing

def solve_mpc(current_state_raw):
    global _last_u_norm
    x0_norm = normalize_state(current_state_raw)
    psi0 = lift(x0_norm)

    Psi = cp.Variable((HORIZON + 1, N_LIFTED))
    U = cp.Variable((HORIZON, N_ACTION))

    cost = 0
    constraints = [Psi[0] == psi0]

    Q_weight = 10.0
    R_weight = 1.0       # increased: penalize large control magnitude more
    DU_weight = 5.0      # NEW: penalize change from previous command (smoothing)
    U_BOUND = 30.0        # further loosened

    prev_u = _last_u_norm
    for t in range(HORIZON):
        constraints += [Psi[t+1] == A @ Psi[t] + B @ U[t]]
        cost += Q_weight * cp.sum_squares(Psi[t+1, :N_STATE] - target_lifted[:N_STATE])
        cost += R_weight * cp.sum_squares(U[t])
        cost += DU_weight * cp.sum_squares(U[t] - prev_u)
        constraints += [U[t] <= U_BOUND, U[t] >= -U_BOUND]
        prev_u = U[t]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP, verbose=False)

    if U.value is None:
        print("WARNING: MPC solve failed, holding hover RPM")
        return np.full(N_ACTION, HOVER_RPM)

    u0_norm = U.value[0]
    _last_u_norm = u0_norm
    u0_raw = denormalize_action(u0_norm)
    return np.clip(u0_raw, 0, 25000)

if __name__ == "__main__":
    test_state = np.zeros(N_STATE)
    test_state[2] = 0.1
    test_state[6] = 1.0
    action = solve_mpc(test_state)
    print(f"[{CONDITION}] MPC solved. Recommended RPM command: {action}")
