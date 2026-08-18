"""
Pivot vs Free-Flight Stability Comparison
-------------------------------------------
Compares roll/pitch stability of the same Koopman-MPC controller under
free 6-DOF flight (scenarios 1-6) vs a fixed-pivot constraint (mimicking
the Quanser 3-DOF Hover rig used in Oh et al. 2024). Free flight and
pivot use the same near-hover training data pipeline; the only
difference is whether translation is mechanically constrained.

Usage:
    python src/analysis/compare_pivot_vs_free.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
OUT_DIR = os.path.join(RESULTS_DIR, "analysis")
os.makedirs(OUT_DIR, exist_ok=True)
CTRL_FREQ = 48

SCENARIO_LABELS = {
    1: "Nominal", 2: "Windy, selector OFF", 3: "Windy, selector ON",
    4: "Nominal->Windy transition", 5: "Payload mid-flight",
    6: "Payload + wind combined",
}

rows = []
for s in range(1, 7):
    d = np.load(f"{RESULTS_DIR}/scenario_{s}.npz")
    roll, pitch, z = d["states"][:, 7], d["states"][:, 8], d["states"][:, 2]
    rows.append({
        "run": f"Free flight S{s}", "scenario_label": SCENARIO_LABELS[s],
        "roll_max_rad": float(np.abs(roll).max()),
        "roll_std_rad": float(roll.std()),
        "pitch_max_rad": float(np.abs(pitch).max()),
        "z_min_m": float(z.min()),
        "crashed": bool(np.abs(roll).max() > 1.5),
    })

d = np.load(f"{RESULTS_DIR}/scenario_pivot.npz")
roll, pitch = d["states"][:, 7], d["states"][:, 8]
rows.append({
    "run": "Pivot-constrained", "scenario_label": "Fixed pivot (Oh et al. rig analog)",
    "roll_max_rad": float(np.abs(roll).max()),
    "roll_std_rad": float(roll.std()),
    "pitch_max_rad": float(np.abs(pitch).max()),
    "z_min_m": float("nan"),
    "crashed": bool(np.abs(roll).max() > 1.5),
})

df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "pivot_vs_free_flight.csv")
df.to_csv(csv_path, index=False)
print(f"Saved {csv_path}")
print(df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["tab:red" if c else "tab:blue" for c in df["crashed"]]
ax.bar(df["run"], df["roll_max_rad"], color=colors)
ax.axhline(np.pi, color="black", linestyle=":", alpha=0.5, label="π (full flip)")
ax.set_ylabel("Max |roll| (rad)")
ax.set_title("Koopman-MPC roll stability: free flight vs fixed pivot")
ax.legend()
plt.xticks(rotation=30, ha="right")
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "pivot_vs_free_flight.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved {out_path}")
