import numpy as np
import os
import sys
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import pybullet as p

# ---- CONFIG ----
CONDITION = sys.argv[1] if len(sys.argv) > 1 else "nominal"
DURATION_SEC = 20
CTRL_FREQ = 48
NUM_STEPS = DURATION_SEC * CTRL_FREQ
SAVE_DIR = f"data/{CONDITION}"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---- INIT ENV ----
env = CtrlAviary(
    drone_model=DroneModel.CF2X,
    num_drones=1,
    physics=Physics.PYB,
    pyb_freq=240,
    ctrl_freq=CTRL_FREQ,
    gui=False
)

obs, info = env.reset()

# ---- CONDITION-SPECIFIC SETUP ----
EXTRA_PAYLOAD_MASS = 0.0
WIND_FORCE = np.array([0.0, 0.0, 0.0])

if CONDITION == "payload":
    EXTRA_PAYLOAD_MASS = 0.01  # kg, added mid-flight (~37% of drone mass)
elif CONDITION == "windy":
    WIND_FORCE = np.array([0.05, 0.0, 0.0])  # constant horizontal force in Newtons

DRONE_ID = env.DRONE_IDS[0]
PAYLOAD_APPLIED = False

log_states = []
log_actions = []

HOVER_RPM = 16000
RPM_RANGE = 3000

for step in range(NUM_STEPS):
    action = HOVER_RPM + np.random.uniform(-RPM_RANGE, RPM_RANGE, size=(1,4))
    obs, reward, terminated, truncated, info = env.step(action)

    # Apply wind as a continuous external force
    if CONDITION == "windy":
        p.applyExternalForce(
            DRONE_ID, -1,
            forceObj=WIND_FORCE.tolist(),
            posObj=[0, 0, 0],
            flags=p.LINK_FRAME,
            physicsClientId=env.CLIENT
        )

    # Apply payload mass change mid-flight (once, at the halfway point)
    if CONDITION == "payload" and step == NUM_STEPS // 2 and not PAYLOAD_APPLIED:
        current_mass = p.getDynamicsInfo(DRONE_ID, -1, physicsClientId=env.CLIENT)[0]
        p.changeDynamics(DRONE_ID, -1, mass=current_mass + EXTRA_PAYLOAD_MASS, physicsClientId=env.CLIENT)
        PAYLOAD_APPLIED = True

    log_states.append(obs[0].copy())
    log_actions.append(action[0].copy())

    if terminated or truncated:
        obs, info = env.reset()
        PAYLOAD_APPLIED = False

env.close()

log_states = np.array(log_states)
log_actions = np.array(log_actions)

np.savez(f"{SAVE_DIR}/flight_log.npz", states=log_states, actions=log_actions)
print(f"[{CONDITION}] Saved {log_states.shape[0]} steps to {SAVE_DIR}/flight_log.npz")
print(f"State shape: {log_states.shape}, Action shape: {log_actions.shape}")
