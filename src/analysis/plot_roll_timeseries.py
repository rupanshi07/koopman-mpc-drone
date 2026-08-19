import numpy as np
import matplotlib.pyplot as plt
import os

CTRL_FREQ = 48
OUT_DIR = "results/analysis"
os.makedirs(OUT_DIR, exist_ok=True)

free = np.load("results/scenario_1.npz")
pivot = np.load("results/scenario_pivot.npz")

roll_free = free["states"][:, 7]
roll_pivot = pivot["states"][:, 7]

t_free = np.arange(len(roll_free)) / CTRL_FREQ
t_pivot = np.arange(len(roll_pivot)) / CTRL_FREQ

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t_free, np.degrees(roll_free), color="tab:red", label="Free flight (Scenario 1)")
ax.plot(t_pivot, np.degrees(roll_pivot), color="tab:blue", label="Pivot-constrained")
ax.axhline(180, color="gray", linestyle=":", alpha=0.5, label="±180deg (full flip)")
ax.axhline(-180, color="gray", linestyle=":", alpha=0.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Roll (degrees)")
ax.set_title("Roll Stability Over Time: Free Flight vs Pivot-Constrained")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()

out_path = os.path.join(OUT_DIR, "roll_timeseries_pivot_vs_free.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved {out_path}")
