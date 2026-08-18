import numpy as np
import os
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import pybullet as p

CONDITION = "pivot"
DURATION_SEC = 60
CTRL_FREQ = 48
NUM_STEPS = DURATION_SEC * CTRL_FREQ
SAVE_DIR = f"data/{CONDITION}"
os.makedirs(SAVE_DIR, exist_ok=True)

HOVER_RPM = 14436
RPM_NOISE_STD = 600       # reduced from 2500 - excite near hover, not full tumbles
NOISE_HOLD_STEPS = 8

env = CtrlAviary(
    drone_model=DroneModel.CF2X, num_drones=1,
    physics=Physics.PYB, pyb_freq=240, ctrl_freq=CTRL_FREQ, gui=False
)
obs, info = env.reset()
DRONE_ID = env.DRONE_IDS[0]

def attach_pivot():
    pos, _ = p.getBasePositionAndOrientation(DRONE_ID, physicsClientId=env.CLIENT)
    cid = p.createConstraint(
        parentBodyUniqueId=DRONE_ID, parentLinkIndex=-1,
        childBodyUniqueId=-1, childLinkIndex=-1,
        jointType=p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0], childFramePosition=pos,
        physicsClientId=env.CLIENT
    )
    p.changeConstraint(cid, maxForce=1e6, physicsClientId=env.CLIENT)
    return cid

attach_pivot()

log_states, log_actions = [], []
reset_count = 0
rpm_offset = np.zeros(4)

for step in range(NUM_STEPS):
    if step % NOISE_HOLD_STEPS == 0:
        rpm_offset = np.random.uniform(-RPM_NOISE_STD, RPM_NOISE_STD, size=4)

    rpm = np.clip(HOVER_RPM + rpm_offset, 0, 25000)
    action = rpm.reshape(1, 4)

    obs, reward, terminated, truncated, info = env.step(action)

    state = obs[0].copy()
    # If the drone has drifted into a large-angle regime, reset it back
    # near hover attitude - keeps data collection in the near-hover
    # linearization region MPC actually needs, rather than full tumbles.
    roll, pitch = state[7], state[8]
    if abs(roll) > 0.6 or abs(pitch) > 0.6 or terminated or truncated:
        obs, info = env.reset()
        attach_pivot()
        reset_count += 1
        continue

    log_states.append(state)
    log_actions.append(action[0].copy())

env.close()
log_states, log_actions = np.array(log_states), np.array(log_actions)

# Enforce quaternion hemisphere continuity: q and -q are the same
# rotation, but PyBullet can flip sign between steps, which looks like
# a teleport to a linear model. Flip sign wherever consecutive
# quaternions point in opposite hemispheres.
quat = log_states[:, 3:7]
for i in range(1, len(quat)):
    if np.dot(quat[i], quat[i-1]) < 0:
        quat[i] *= -1
log_states[:, 3:7] = quat

np.savez(f"{SAVE_DIR}/flight_log.npz", states=log_states, actions=log_actions)

z, roll = log_states[:, 2], log_states[:, 7]
print(f"[{CONDITION}] Saved {log_states.shape[0]} steps, {reset_count} resets")
print(f"Z range: min={z.min():.3f}, max={z.max():.3f}, mean={z.mean():.3f}")
print(f"Roll range: min={roll.min():.3f}, max={roll.max():.3f}, std={roll.std():.3f}")

diffs = np.abs(np.diff(roll))
qdiffs = np.abs(np.diff(quat, axis=0)).max(axis=1)
print(f"Max roll jump: {diffs.max():.3f} | jumps>1.0: {(diffs>1.0).sum()}")
print(f"Max quat jump: {qdiffs.max():.3f} | jumps>1.0: {(qdiffs>1.0).sum()}")
