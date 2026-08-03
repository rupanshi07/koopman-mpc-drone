import numpy as np
import cvxpy as cp
from itertools import combinations

N_STATE = 16
N_ACTION = 4
HORIZON = 6
HOVER_RPM = 14436

_models = {}  # cache loaded models by condition name

def load_model(condition):
    if condition not in _models:
        m = np.load(f"data/{condition}/koopman_model.npz")
        _models[condition] = {
            "A": m["A"], "B": m["B"],
            "x_mean": m["x_mean"], "x_std": m["x_std"],
            "u_mean": m["u_mean"], "u_std": m["u_std"],
            "cross_pairs": [tuple(p) for p in m["cross_pairs"]]
        }
    return _models[condition]

def lift(x, cross_pairs):
    quad = x**2
    cross = np.array([x[i] * x[j] for i, j in cross_pairs])
    return np.concatenate([x, quad, cross])

target_state_raw = np.zeros(N_STATE)
target_state_raw[2] = 0.3
target_state_raw[6] = 1.0

_last_u_norm = np.zeros(N_ACTION)

def solve_mpc(current_state_raw, condition="nominal"):
    global _last_u_norm
    
    if np.any(np.isnan(current_state_raw)) or np.any(np.isinf(current_state_raw)) or np.any(np.abs(current_state_raw) > 1e4):
        model = load_model(condition)
        u0_raw = _last_u_norm * model["u_std"] + model["u_mean"]
        return np.clip(u0_raw, 0, 25000)

    model = load_model(condition)
    A, B = model["A"], model["B"]
    x_mean, x_std = model["x_mean"], model["x_std"]
    u_mean, u_std = model["u_mean"], model["u_std"]
    cross_pairs = model["cross_pairs"]

    x0_norm = (current_state_raw - x_mean) / x_std
    target_norm = (target_state_raw - x_mean) / x_std
    psi0 = lift(x0_norm, cross_pairs)
    target_lifted = lift(target_norm, cross_pairs)

    N_LIFTED = A.shape[0]
    Psi = cp.Variable((HORIZON + 1, N_LIFTED))
    U = cp.Variable((HORIZON, N_ACTION))

    cost = 0
    constraints = [Psi[0] == psi0]

    Q_weight = 20.0
    R_weight = 0.1
    DU_weight = 1.0
    U_BOUND = 4.0
    MAX_RATE = 1.0

    prev_u = _last_u_norm
    for t in range(HORIZON):
        constraints += [Psi[t+1] == A @ Psi[t] + B @ U[t]]
        cost += Q_weight * cp.sum_squares(Psi[t+1, :N_STATE] - target_lifted[:N_STATE])
        cost += R_weight * cp.sum_squares(U[t])
        cost += DU_weight * cp.sum_squares(U[t] - prev_u)
        constraints += [U[t] <= U_BOUND, U[t] >= -U_BOUND]
        constraints += [U[t] - prev_u <= MAX_RATE, U[t] - prev_u >= -MAX_RATE]
        prev_u = U[t]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    try:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    except cp.error.SolverError:
        pass # U.value will remain None, handled below

    if U.value is None or np.any(np.isnan(U.value)):
        # Safer fallback: hold previous command instead of jumping to hover
        u0_raw = _last_u_norm * u_std + u_mean
        return np.clip(u0_raw, 0, 25000)

    u0_norm = U.value[0]
    _last_u_norm = u0_norm
    u0_raw = u0_norm * u_std + u_mean
    return np.clip(u0_raw, 0, 25000)
