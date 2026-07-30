import numpy as np
import sys

CONDITIONS = ["nominal", "windy", "payload"]
precision_values = {}

# Use only RPY (indices 7-9) + angular velocity (indices 13-15) as the
# discriminating signature - matches the base paper's exact choice, and
# avoids double-counting orientation via the redundant quaternion.
SIG_IDX = list(range(7, 10)) + list(range(13, 16))

for cond in CONDITIONS:
    data = np.load(f"data/{cond}/flight_log.npz")
    states = data["states"][:, :16]
    signature = states[:, SIG_IDX].mean(axis=0)
    precision_values[cond] = signature
    print(f"{cond}: signature = {signature}")

np.savez("data/environment_selector_precision.npz",
         nominal=precision_values["nominal"],
         windy=precision_values["windy"],
         payload=precision_values["payload"])

print("Saved precision values to data/environment_selector_precision.npz")

def select_environment(window_states_16, precision_dict):
    sig = window_states_16[:, SIG_IDX].mean(axis=0)
    dists = {cond: np.linalg.norm(sig - precision_dict[cond]) for cond in precision_dict}
    return min(dists, key=dists.get), dists

if __name__ == "__main__":
    loaded = np.load("data/environment_selector_precision.npz")
    precision_dict = {c: loaded[c] for c in CONDITIONS}
    WINDOW = 240

    for cond in CONDITIONS:
        data = np.load(f"data/{cond}/flight_log.npz")
        states = data["states"][:, :16]
        start = int(states.shape[0] * 0.75)
        window = states[start:start+WINDOW]
        predicted, dists = select_environment(window, precision_dict)
        print(f"True: {cond:8s} | Predicted: {predicted:8s} | Distances: {dists}")
