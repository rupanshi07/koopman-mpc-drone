import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

data = np.load("results/scenario_pivot.npz")
states = data["states"]
roll, pitch, yaw = states[:, 7], states[:, 8], states[:, 9]

EXAGGERATE = 15  # visually amplify rotation so real (tiny) motion is visible
                  # actual roll/pitch stayed within +/-0.5 degrees - this
                  # multiplier is for display only, not a data change

CTRL_FREQ = 48
STEP = 4

fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection="3d")

ARM_LEN = 0.08
ax.set_xlim(-ARM_LEN*1.3, ARM_LEN*1.3)
ax.set_ylim(-ARM_LEN*1.3, ARM_LEN*1.3)
ax.set_zlim(-ARM_LEN*1.3, ARM_LEN*1.3)
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

arm_lines = [ax.plot([], [], [], "k-", linewidth=2)[0] for _ in range(2)]
center_point, = ax.plot([0], [0], [0], "ro", markersize=8)

def rotation_matrix(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    return Rz @ Ry @ Rx

def update(frame):
    i = frame * STEP
    if i >= len(roll):
        i = len(roll) - 1

    R = rotation_matrix(roll[i]*EXAGGERATE, pitch[i]*EXAGGERATE, yaw[i])
    arm1 = R @ np.array([ARM_LEN, 0, 0])
    arm2 = R @ np.array([0, ARM_LEN, 0])
    arm_lines[0].set_data([-arm1[0], arm1[0]], [-arm1[1], arm1[1]])
    arm_lines[0].set_3d_properties([-arm1[2], arm1[2]])
    arm_lines[1].set_data([-arm2[0], arm2[0]], [-arm2[1], arm2[1]])
    arm_lines[1].set_3d_properties([-arm2[2], arm2[2]])

    ax.set_title(f"Pivot-Constrained MPC (rotation shown at {EXAGGERATE}x for visibility)\n"
                 f"t={i/CTRL_FREQ:.2f}s | actual roll={np.degrees(roll[i]):.2f}deg pitch={np.degrees(pitch[i]):.2f}deg")
    return center_point, *arm_lines

n_frames = len(roll) // STEP
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=40, blit=False)
ani.save("results/plots_updated/flight_animation_pivot.gif", writer="pillow", fps=25)
print("Saved results/plots_updated/flight_animation_pivot.gif")
