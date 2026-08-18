import re

with open("README.md", "r", encoding="utf-8") as f:
    content = f.read()

# --- Insert new Â§3.5 after Â§3.4 ---
marker_1 = """This isolates the failure to the controller/model itself "” specifically, to the
Euler-angle/quaternion-based orientation representation used in the Koopman lifting
function "” rather than to any disturbance-testing methodology in this project.

---

## 4. Discussion"""

new_section = """This isolates the failure to the controller/model itself "” specifically, to the
Euler-angle/quaternion-based orientation representation used in the Koopman lifting
function "” rather than to any disturbance-testing methodology in this project.

### 3.5 Pivot-Constrained Validation (Isolating the Coupling Mechanism)

To directly test *why* Oh et al.'s own validation (a Quanser 3 DOF Hover rig "”
a fixed-base, pivot-mounted platform that permits rotation about roll/pitch/yaw
but physically cannot translate) did not surface this instability, we replicated
their mechanical constraint in simulation. A `pybullet` point-to-point constraint
was attached at the drone's body origin, locking translational position while
leaving all three rotational degrees of freedom free "” mechanically equivalent
to the Quanser rig's pivot joint.

A new Koopman model was trained under this constraint, using the same near-hover
excitation philosophy as the free-flight data (small-amplitude random RPM
perturbation about hover, not aggressive full-range excitation), with the same
EDMD lifting structure and identical hyperparameters. The resulting controller
was then run for the same 15-second duration used in Â§3.1"“3.3.

| Run | Roll max (rad) | Roll Ïƒ | Pitch max (rad) | Outcome |
|---|---|---|---|---|
| Free flight, S1"“S6 (all) | 3.1416 (Ï€) | 1.16"“1.56 | 1.44 | Crashes, every scenario |
| Pivot-constrained | 0.009 | 0.004 | 0.001 | Stable, full 15s |

Every free-flight scenario "” independent of wind, payload, or their combination "”
saturates roll at exactly Ï€ (a full flip) with high variance. The pivot-constrained
run, using the same controller architecture and a comparably narrow training
distribution, remains stable to within a fraction of a degree for the entire run.

Because both conditions share the same lifting function, the same near-hover
training philosophy, and the same MPC formulation, and differ *only* in whether
translation is mechanically free, this isolates the coupling between translational
and rotational dynamics "” present in free 6-DOF flight, absent on a fixed
pivot "” as the mechanism that exposes the instability, rather than data quality
or excitation signal design alone. This also explains why Oh et al.'s own
validation platform did not detect the same failure mode we observe: a pivot-mounted
rig cannot express the position-attitude feedback loop that drives the crash.

*(Figure: `results/analysis/pivot_vs_free_flight.png`; data:
`results/analysis/pivot_vs_free_flight.csv`)*

---

## 4. Discussion"""

if marker_1 not in content:
    print("ERROR: marker_1 not found, no changes made. Check README formatting.")
else:
    content = content.replace(marker_1, new_section)
    print("Inserted Â§3.5.")

# --- Extend Â§4 Discussion with refined causal claim ---
marker_2 = """Migrating to that representation
is identified as the concrete next step for this project, though it requires
retraining the Koopman model on a re-derived state representation and is left as
future work to preserve fidelity to the base paper's stated methodology."""

new_discussion = """Migrating to that representation
is identified as the concrete next step for this project, though it requires
retraining the Koopman model on a re-derived state representation and is left as
future work to preserve fidelity to the base paper's stated methodology.

The pivot-constrained validation (Â§3.5) refines this diagnosis further: the same
lifting function and a comparably narrow training distribution produce a *stable*
controller once translation is mechanically decoupled from rotation. This indicates
the instability is not purely a static property of the Euler-angle/quaternion
representation in isolation, but is triggered specifically by the position-attitude
feedback loop inherent to free 6-DOF flight "” a small attitude error tilts the
thrust vector, which perturbs position, which the linear model (fit on near-hover
data) increasingly mispredicts as the state moves outside its training
distribution, compounding the original attitude error. A pivot-mounted rig such as
Oh et al.'s Quanser 3 DOF Hover structurally cannot enter this loop, which is
consistent with why their reported results do not exhibit this failure mode."""

if marker_2 not in content:
    print("ERROR: marker_2 not found, no changes made to Â§4.")
else:
    content = content.replace(marker_2, new_discussion)
    print("Extended Â§4 Discussion.")

# --- Update repository contents table ---
marker_3 = "| `results/plots_updated/` | Corrected, current Z-tracking comparison plots (all 3 controllers, all 6 scenarios) |"
new_row = marker_3 + "\n| `results/analysis/pivot_vs_free_flight.*` | Pivot-constrained vs free-flight stability comparison (Â§3.5) |"

if marker_3 not in content:
    print("ERROR: marker_3 not found, repo table not updated.")
else:
    content = content.replace(marker_3, new_row)
    print("Updated repository contents table.")

with open("README.md", "w", encoding="utf-8") as f:
    f.write(content)

print("Done. Review README.md before committing.")
