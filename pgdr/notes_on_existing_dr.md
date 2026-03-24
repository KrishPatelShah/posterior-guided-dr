# Notes on Existing Domain Randomization in SPI-Active

## Summary

SPI-Active (this repo) uses **IsaacGym** with **PhysX GPU** for the Unitree Go2.
PGDR targets the **Booster T1** on **MuJoCo MJX**. This document records the
SPI-Active DR scheme as a reference for what the literature baseline looks like,
and identifies the analogous MJX fields we must randomize.

---

## SPI-Active DR Parameters (IsaacGym / Go2)

Source: `spigym/config/domain_rand/domain_rand_base.yaml` and
`spigym/envs/legged_base_task/legged_robot_base.py`.

### Applied at environment creation (`_process_rigid_body_props`)

| Parameter | Range | IsaacGym field |
|---|---|---|
| Ground friction | U[0.5, 1.25] multiplier | `rigid_shape_props.friction` |
| Base COM offset | x: [-0.15, 0.15], y: [-0.05, 0.05], z: [-0.05, 0.10] | `rigid_body_props.com` |
| Link mass scale | U[0.8, 1.2] per link | `rigid_body_props.mass` |

### Applied at episode reset (`_episodic_domain_randomization`)

| Parameter | Range | Mechanism |
|---|---|---|
| PD Kp scale | U[0.75, 1.25] per joint | Multiplied in `_compute_torques` |
| PD Kd scale | U[0.75, 1.25] per joint | Multiplied in `_compute_torques` |
| RFI limit scale | U[0.5, 1.5] | Random friction injection amplitude |
| Control delay | U{0, 1, 2} steps | Action queue indexing |

### Applied at each step

| Parameter | Mechanism |
|---|---|
| RFI noise | Uniform noise added to torques, scaled by RFI limit |
| Motor saturation | `torque = A * tanh(action / A)` per joint group |

---

## MuJoCo Playground DR (Booster T1 — expected)

Based on MuJoCo Playground conventions, the T1 joystick environment likely
randomizes similar quantities via `mjx.Model` field replacement:

| Parameter | MJX field | Expected default range |
|---|---|---|
| Link masses | `model.body_mass` | U[0.8, 1.2] scale |
| Joint damping | `model.dof_damping` | U[0.5, 2.0] scale |
| Actuator gains | `model.actuator_gainprm[:, 0]` | U[0.8, 1.2] scale |
| Ground friction | `model.geom_friction[:, 0]` | U[0.5, 2.0] |
| Contact stiffness | `model.geom_solref[:, 0]` | varied |
| Contact damping | `model.geom_solref[:, 1]` | varied |

The Playground training loop vectorizes environments by `jax.vmap` over batched
`mjx.Model` pytrees. DR is applied by replacing model fields with sampled values
before each episode reset.

---

## PGDR Parameter Vector Mapping

Our ~65-dim parameter vector maps to these MJX fields:

| Group | Dim | MJX field | DR condition mapping |
|---|---|---|---|
| Joint friction (viscous) | 23 | `model.dof_damping[actuated_dof_ids]` | Additive offset from default |
| Link mass offsets | ~15 | `model.body_mass[movable_body_ids]` | Additive delta |
| Actuator gain scale | 23 | `model.actuator_gainprm[:, 0]` | Multiplicative factor |
| Ground contact | 4 | `geom_friction[foot_ids, 0]`, `geom_solref[foot_ids, :]` | Direct value |

**C1 (uniform DR)** uses the Playground defaults (wide uniform ranges).
**C2 (pure sys-id)** sets all params to p* with zero randomization.
**C3 (isotropic)** samples N(p*, β²I) with β chosen to match tr(αΣ).
**C4 (PGDR)** samples N(p*, αΣ) — the proposed method.

---

## Key Differences: SPI-Active vs PGDR

1. **SPI-Active** identifies a point estimate, then applies hand-tuned narrow DR.
   The covariance from CMA-ES is discarded.
2. **PGDR** retains the covariance Σ and uses it as the DR distribution.
   No hand-tuning of DR ranges is needed.
3. **SPI-Active** operates on IsaacGym (PhysX). PGDR operates on MJX (MuJoCo).
4. **SPI-Active** targets Go2 (quadruped, 12 DOF).
   PGDR targets T1 (humanoid, 23 DOF, ~65 param dims).
