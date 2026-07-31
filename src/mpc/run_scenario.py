import numpy as np
import sys
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import pybullet as p
sys.path.insert(0, "src/mpc")
sys.path.insert(0, "src/env_selector")
from mpc_controller import solve_mpc
from env_selector import select_environment

# ---- SCENARIO CONFIG ----
# scenario 1: nominal env, nominal dynamics (selector on)
# scenario 2: windy env, nominal dynamics (selector OFF - ablation)
# scenario 3: windy env, windy dynamics (selector on)
# scenario 4: varying env nominal->windy (selector on)
SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "1"

CTRL_FREQ = 48
DURATION_SEC = 15
NUM_STEPS = DURATION_SEC * CTRL_FREQ
SELECTOR_WINDOW = 240  # 5 seconds, matching base paper

precision = np.load("data/environment_selector_precision.npz")
precision_dict = {c: precision[c] for c in ["nominal", "windy", "payload"]}

env = CtrlAviary(
    drone_model=DroneModel.CF2X, num_drones=1,
    physics=Physics.PYB, pyb_freq=240, ctrl_freq=CTRL_FREQ, gui=False
)
obs, info = env.reset()
DRONE_ID = env.DRONE_IDS[0]

WIND_FORCE = np.array([0.05, 0.0, 0.0])
state_buffer = []
log_states, log_actions, log_selected = [], [], []

active_condition = "nominal"

for step in range(NUM_STEPS):
    apply_wind = False
    if SCENARIO in ["2", "3"]:
        apply_wind = True
    elif SCENARIO == "4":
        apply_wind = step > NUM_STEPS // 2  # switches nominal->windy halfway

    if apply_wind:
        p.applyExternalForce(DRONE_ID, -1, forceObj=WIND_FORCE.tolist(),
                              posObj=[0,0,0], flags=p.LINK_FRAME,
                              physicsClientId=env.CLIENT)

    current_state = obs[0][:16]
    state_buffer.append(current_state.copy())
    if len(state_buffer) > SELECTOR_WINDOW:
        state_buffer.pop(0)

    use_selector = SCENARIO != "2"  # scenario 2 disables the selector (ablation)
    if use_selector and step % SELECTOR_WINDOW == 0 and len(state_buffer) == SELECTOR_WINDOW:
        window = np.array(state_buffer)
        active_condition, dists = select_environment(window, precision_dict)
    elif not use_selector:
        active_condition = "nominal"  # forced wrong model, per scenario 2 design

    action = solve_mpc(current_state, condition=active_condition)
    action = action.reshape(1, 4)

    obs, reward, terminated, truncated, info = env.step(action)

    log_states.append(obs[0].copy())
    log_actions.append(action[0].copy())
    log_selected.append(active_condition)

    if step % 48 == 0:
        z = obs[0][2]
        print(f"t={step/CTRL_FREQ:.1f}s | Z={z:.3f}m | active_model={active_condition}")

    if terminated or truncated:
        obs, info = env.reset()

env.close()

np.savez(f"results/scenario_{SCENARIO}.npz",
         states=np.array(log_states), actions=np.array(log_actions))
print(f"Saved results/scenario_{SCENARIO}.npz")
