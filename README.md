# Koopman‑MPC Drone – Disturbance‑Testing Work

## Overview
This repository contains the **Koopman‑MPC** controller for a quadrotor simulated with PyBullet and a systematic disturbance‑testing campaign. The work focuses on executing six predefined flight scenarios and saving the resulting state data.

## What Is Included
- **`src/mpc/run_scenario.py`** – Driver script that selects a scenario (1‑6), runs the MPC controller, prints diagnostic information, and stores the simulation trace in `results/scenario_<n>.npz`.
- **`src/mpc/mpc_controller.py`** – Implementation of the Koopman‑based MPC formulation. It uses the **CLARABEL** solver with robust error handling to avoid hangs.
- **`data/environment_selector_precision.npz`** – Calibration data required by the controller for the environment selector.
- **`results/`** – Generated output files (`scenario_1.npz` … `scenario_6.npz`). These files are created by the driver script and are intentionally excluded from version control via `.gitignore`.
- **Documentation artifacts** – `implementation_plan.md` and `walkthrough.md` provide a high‑level design description and a summary of execution steps.

## Architecture
```
project_root/
│
├─ data/                     # Calibration / static data
│   └─ environment_selector_precision.npz
│
├─ src/
│   └─ mpc/
│       ├─ run_scenario.py      # Scenario orchestration
│       └─ mpc_controller.py    # Koopman‑MPC implementation (CLARABEL solver)
│
└─ results/                  # Generated simulation traces (npz files)
```
- The **driver** (`run_scenario.py`) loads the appropriate disturbance configuration, creates the simulation environment, and repeatedly calls the **MPC controller** to compute control actions.
- The **controller** formulates a quadratic program based on Koopman‑derived dynamics and solves it with CLARABEL. A `try/except` block catches solver failures and safely aborts the run.
- Each simulation outputs a NumPy archive (`.npz`) containing state trajectories, control inputs, and timestamps.

## How to Verify Locally
```powershell
# 1. Activate the virtual environment
cd <project_root>
.\venv\Scripts\activate

# 2. Run all six scenarios (you will see console logs similar to the samples)
for %i in (1 2 3 4 5 6) do (
    echo -------------------------------------------------
    echo Running scenario %i
    .\venv\Scripts\python.exe src\mpc\run_scenario.py %i
)

# 3. Confirm the result files were created
dir results\scenario_*.npz
```
If the commands finish without errors and the `.npz` files appear, the disturbance‑testing component is fully functional.

---
*This README documents the completed disturbance‑testing portion of the project.*
