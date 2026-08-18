import numpy as np
import sys
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import pybullet as p
sys.path.insert(0, "src/mpc")
from mpc_controller import solve_mpc

CTRL_FREQ = 48
DURATION_SEC = 15
NUM_STEPS = DURATION_SEC * CTRL_FREQ

env = CtrlAviary(
    drone_model=DroneModel.CF2X, num_drones=1,
    physics=Physics.PYB, pyb_freq=240, ctrl_freq=CTRL_FREQ, gui=False
)
obs, info = env.reset()
DRONE_ID = env.DRONE_IDS[0]

pos, _ = p.getBasePositionAndOrientation(DRONE_ID, physicsClientId=env.CLIENT)
pivot_cid = p.createConstraint(
    parentBodyUniqueId=DRONE_ID, parentLinkIndex=-1,
    childBodyUniqueId=-1, childLinkIndex=-1,
    jointType=p.JOINT_POINT2POINT, jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0], childFramePosition=pos,
    physicsClientId=env.CLIENT
)
p.changeConstraint(pivot_cid, maxForce=1e6, physicsClientId=env.CLIENT)

log_states, log_actions = [], []

for step in range(NUM_STEPS):
    current_state = obs[0][:16]
    action = solve_mpc(current_state, condition="pivot")
    action = action.reshape(1, 4)

    obs, reward, terminated, truncated, info = env.step(action)

    log_states.append(obs[0].copy())
    log_actions.append(action[0].copy())

    if step % 48 == 0:
        roll, pitch = obs[0][7], obs[0][8]
        print(f"t={step/CTRL_FREQ:.1f}s | roll={roll:.3f} | pitch={pitch:.3f}", flush=True)

    if terminated or truncated:
        print(f"Ended early at t={step/CTRL_FREQ:.1f}s (terminated={terminated}, truncated={truncated})")
        break

env.close()
log_states = np.array(log_states)
np.savez("results/scenario_pivot.npz", states=log_states, actions=np.array(log_actions))

roll, pitch = log_states[:, 7], log_states[:, 8]
print(f"\nPivot-constrained run complete: {log_states.shape[0]} steps")
print(f"Roll: max={np.abs(roll).max():.3f} rad, std={roll.std():.3f}")
print(f"Pitch: max={np.abs(pitch).max():.3f} rad, std={pitch.std():.3f}")
print("Saved results/scenario_pivot.npz")
