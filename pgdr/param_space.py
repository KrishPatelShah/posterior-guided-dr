"""
Parameter vector definition for Booster T1 system identification.

Defines the ~65-dimensional parameter vector, normalization, and
inject/extract operations between flat vectors and mjx.Model instances.

The parameter groups (from the proposal):
    - Joint friction (viscous damping): 23 params
    - Link mass offsets: ~15 params (movable bodies, skip world/fixed)
    - Actuator gain scaling: 23 params
    - Ground contact properties: 4 params (friction, stiffness, damping, restitution)

All parameters are normalized to unit scale for CMA-ES.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


# ---------------------------------------------------------------------------
# Parameter definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamDef:
    """Single parameter entry in the identification space."""
    name: str
    group: str              # "friction", "mass", "actuator", "contact"
    mjx_field: str          # e.g. "dof_damping", "body_mass"
    index: int              # Row index into the MJX array
    default: float          # Nominal value from the XML model
    scale: float            # Normalization scale (maps ±1 → ±scale in physical units)
    mode: str = "additive"  # "additive" or "multiplicative"
    lower: float = -jnp.inf # Physical lower bound (e.g., mass > 0)
    upper: float = jnp.inf  # Physical upper bound
    col: int = 0            # Column index for 2D arrays (e.g., solref col 0 vs 1)


@dataclass
class ParamSpace:
    """
    Manages the d-dimensional parameter vector for T1 identification.

    Normalized space:  CMA-ES operates here.  Each dimension ≈ O(1).
    Physical space:    What gets injected into mjx.Model.

    Conversion:
        physical = default + normalized * scale   (additive mode)
        physical = default * (1 + normalized * scale)  (multiplicative mode)
    """
    params: list[ParamDef] = field(default_factory=list)

    # ---- Derived arrays (populated by finalize()) ---- #
    _defaults: Optional[jnp.ndarray] = field(default=None, repr=False)
    _scales: Optional[jnp.ndarray] = field(default=None, repr=False)
    _lowers: Optional[jnp.ndarray] = field(default=None, repr=False)
    _uppers: Optional[jnp.ndarray] = field(default=None, repr=False)
    _modes: Optional[list[str]] = field(default=None, repr=False)

    @property
    def d(self) -> int:
        return len(self.params)

    def finalize(self) -> ParamSpace:
        """Precompute vectorized arrays after all params are added."""
        self._defaults = jnp.array([p.default for p in self.params])
        self._scales = jnp.array([p.scale for p in self.params])
        self._lowers = jnp.array([p.lower for p in self.params])
        self._uppers = jnp.array([p.upper for p in self.params])
        self._modes = [p.mode for p in self.params]
        return self

    # ---- Coordinate transforms ---- #

    def to_physical(self, p_normalized: jnp.ndarray) -> jnp.ndarray:
        """Convert a normalized vector to physical parameter values."""
        physical = jnp.zeros_like(p_normalized)
        for i, param in enumerate(self.params):
            if param.mode == "additive":
                physical = physical.at[i].set(
                    param.default + p_normalized[i] * param.scale
                )
            else:  # multiplicative
                physical = physical.at[i].set(
                    param.default * (1.0 + p_normalized[i] * param.scale)
                )
        return jnp.clip(physical, self._lowers, self._uppers)

    def to_normalized(self, p_physical: jnp.ndarray) -> jnp.ndarray:
        """Convert a physical parameter vector to normalized space."""
        normalized = jnp.zeros_like(p_physical)
        for i, param in enumerate(self.params):
            if param.mode == "additive":
                normalized = normalized.at[i].set(
                    (p_physical[i] - param.default) / param.scale
                )
            else:
                normalized = normalized.at[i].set(
                    (p_physical[i] / param.default - 1.0) / param.scale
                )
        return normalized

    # ---- Vectorized transforms (JAX-friendly) ---- #

    def to_physical_vec(self, p_normalized: jnp.ndarray) -> jnp.ndarray:
        """Vectorized conversion — works inside jax.vmap / jax.jit."""
        additive_mask = jnp.array(
            [1.0 if m == "additive" else 0.0 for m in self._modes]
        )
        mult_mask = 1.0 - additive_mask

        p_add = self._defaults + p_normalized * self._scales
        p_mul = self._defaults * (1.0 + p_normalized * self._scales)
        physical = additive_mask * p_add + mult_mask * p_mul
        return jnp.clip(physical, self._lowers, self._uppers)

    def to_normalized_vec(self, p_physical: jnp.ndarray) -> jnp.ndarray:
        """Vectorized inverse."""
        additive_mask = jnp.array(
            [1.0 if m == "additive" else 0.0 for m in self._modes]
        )
        mult_mask = 1.0 - additive_mask

        n_add = (p_physical - self._defaults) / self._scales
        n_mul = (p_physical / self._defaults - 1.0) / self._scales
        return additive_mask * n_add + mult_mask * n_mul

    # ---- MJX model injection / extraction ---- #

    def inject(self, model: mjx.Model, p_normalized: jnp.ndarray) -> mjx.Model:
        """
        Given a normalized parameter vector, return a modified mjx.Model.

        This replaces specific fields in the model pytree with the
        physical values derived from the parameter vector.
        """
        p_phys = self.to_physical_vec(p_normalized)

        # Group updates by MJX field for efficiency
        updates = {}  # field_name -> list of (row_index, col_index, value)
        for i, param in enumerate(self.params):
            key = param.mjx_field
            if key not in updates:
                updates[key] = []
            updates[key].append((param.index, param.col, p_phys[i]))

        # Apply updates to model
        for field_name, index_vals in updates.items():
            arr = getattr(model, field_name)
            for idx, col, val in index_vals:
                if arr.ndim == 1:
                    arr = arr.at[idx].set(val)
                elif arr.ndim == 2:
                    arr = arr.at[idx, col].set(val)
                else:
                    arr = arr.at[idx].set(val)
            model = model.replace(**{field_name: arr})

        return model

    def extract(self, model: mjx.Model) -> jnp.ndarray:
        """Extract the current physical parameter values from an mjx.Model,
        return as a normalized vector."""
        physical = jnp.zeros(self.d)
        for i, param in enumerate(self.params):
            arr = getattr(model, param.mjx_field)
            if arr.ndim == 1:
                val = arr[param.index]
            elif arr.ndim == 2:
                val = arr[param.index, param.col]
            else:
                val = arr[param.index]
            physical = physical.at[i].set(val)
        return self.to_normalized_vec(physical)

    # ---- Persistence ---- #

    def save(self, path: str | Path) -> None:
        """Save parameter space definition to JSON."""
        data = []
        for p in self.params:
            data.append({
                "name": p.name,
                "group": p.group,
                "mjx_field": p.mjx_field,
                "index": p.index,
                "default": float(p.default),
                "scale": float(p.scale),
                "mode": p.mode,
                "lower": float(p.lower),
                "upper": float(p.upper),
                "col": p.col,
            })
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> ParamSpace:
        """Load parameter space from JSON."""
        data = json.loads(Path(path).read_text())
        ps = cls(params=[ParamDef(**d) for d in data])
        return ps.finalize()

    # ---- Subspace selection ---- #

    def select(self, indices: list[int]) -> ParamSpace:
        """Return a reduced ParamSpace keeping only the given indices."""
        ps = ParamSpace(params=[self.params[i] for i in indices])
        return ps.finalize()

    def select_by_group(self, groups: list[str]) -> ParamSpace:
        """Return a reduced ParamSpace keeping only the given groups."""
        ps = ParamSpace(
            params=[p for p in self.params if p.group in groups]
        )
        return ps.finalize()

    def group_indices(self, group: str) -> list[int]:
        """Return indices of parameters in a given group."""
        return [i for i, p in enumerate(self.params) if p.group == group]


# ---------------------------------------------------------------------------
# Factory: build the T1 parameter space from a MuJoCo model
# ---------------------------------------------------------------------------

def build_t1_param_space(mj_model: mujoco.MjModel) -> ParamSpace:
    """
    Construct the ~65-dim parameter space for the Booster T1 from its
    MuJoCo model specification.

    This inspects the model to determine body/joint/actuator counts and
    their default values, then defines the parameter vector accordingly.

    Args:
        mj_model: Loaded MuJoCo model (not MJX — we need name lookups).

    Returns:
        Finalized ParamSpace ready for identification.
    """
    params = []

    # --- 1. Joint friction (viscous damping) ---
    # One per actuated DoF.  The T1 has 23 actuated joints.
    num_actuators = mj_model.nu
    actuated_dof_ids = []
    for i in range(num_actuators):
        # Actuator i drives joint mj_model.actuator_trnid[i, 0]
        jnt_id = mj_model.actuator_trnid[i, 0]
        dof_adr = mj_model.jnt_dofadr[jnt_id]
        actuated_dof_ids.append(dof_adr)

        dof_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        default_damping = float(mj_model.dof_damping[dof_adr])
        # Scale: allow ±50% variation from default (or ±0.5 if default is ~0)
        scale = max(abs(default_damping) * 0.5, 0.5)

        params.append(ParamDef(
            name=f"friction_{dof_name}",
            group="friction",
            mjx_field="dof_damping",
            index=int(dof_adr),
            default=default_damping,
            scale=scale,
            mode="additive",
            lower=0.0,  # Damping must be non-negative
        ))

    # --- 2. Link mass offsets ---
    # Skip body 0 (world) and any fixed bodies.
    movable_body_ids = []
    for i in range(1, mj_model.nbody):  # skip world body
        body_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        mass = float(mj_model.body_mass[i])
        if mass < 1e-6:
            continue  # Skip massless bodies
        movable_body_ids.append(i)

        # Scale: ±20% of nominal mass
        scale = mass * 0.2

        params.append(ParamDef(
            name=f"mass_{body_name}",
            group="mass",
            mjx_field="body_mass",
            index=i,
            default=mass,
            scale=scale,
            mode="additive",
            lower=mass * 0.1,  # Don't allow near-zero masses
        ))

    # --- 3. Actuator gain scaling ---
    # Multiplicative factor on gainprm[:, 0] for each actuator.
    for i in range(num_actuators):
        act_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        default_gain = float(mj_model.actuator_gainprm[i, 0])
        if abs(default_gain) < 1e-8:
            default_gain = 1.0  # Fallback for zero-gain actuators

        params.append(ParamDef(
            name=f"gain_{act_name}",
            group="actuator",
            mjx_field="actuator_gainprm",
            index=i,
            default=default_gain,
            scale=0.3,  # ±30% multiplicative variation
            mode="multiplicative",
            lower=default_gain * 0.3,
            upper=default_gain * 3.0,
        ))

    # --- 4. Ground contact properties ---
    # Identify foot geom IDs by name convention (varies by model).
    # We parameterize: friction, solref stiffness, solref damping, solimp width.
    foot_geom_ids = _find_foot_geoms(mj_model)

    if foot_geom_ids:
        # Use the first foot geom as representative; all feet share the same params
        gid = foot_geom_ids[0]

        # 4a. Friction coefficient
        default_friction = float(mj_model.geom_friction[gid, 0])
        params.append(ParamDef(
            name="contact_friction",
            group="contact",
            mjx_field="geom_friction",
            index=gid,
            default=default_friction,
            scale=default_friction * 0.5,
            mode="additive",
            lower=0.05,
            upper=3.0,
        ))

        # 4b. Contact stiffness (solref[0] — time constant)
        default_stiffness = float(mj_model.geom_solref[gid, 0])
        params.append(ParamDef(
            name="contact_stiffness",
            group="contact",
            mjx_field="geom_solref",
            index=gid,
            default=default_stiffness,
            scale=abs(default_stiffness) * 0.3,
            mode="additive",
            lower=-1.0,  # Negative solref = stiffness mode
        ))

        # 4c. Contact damping (solref[1])
        default_damping = float(mj_model.geom_solref[gid, 1])
        params.append(ParamDef(
            name="contact_damping",
            group="contact",
            mjx_field="geom_solref",
            index=gid,
            default=default_damping,
            scale=abs(default_damping) * 0.3 + 0.1,
            mode="additive",
            lower=0.0,
            col=1,  # solref column 1 = damping ratio
        ))

        # 4d. Solimp width (controls contact softness)
        default_solimp = float(mj_model.geom_solimp[gid, 0])
        params.append(ParamDef(
            name="contact_solimp_width",
            group="contact",
            mjx_field="geom_solimp",
            index=gid,
            default=default_solimp,
            scale=0.3,
            mode="additive",
            lower=0.001,
            upper=0.999,
        ))

    ps = ParamSpace(params=params)
    return ps.finalize()


def _find_foot_geoms(mj_model: mujoco.MjModel) -> list[int]:
    """
    Heuristic to find foot geom IDs in the T1 model.
    Searches for geoms whose names contain 'foot', 'ankle', or 'sole'.
    Falls back to the last geoms in the kinematic tree.
    """
    foot_keywords = ["foot", "ankle", "sole", "toe"]
    foot_ids = []
    for i in range(mj_model.ngeom):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if name and any(kw in name.lower() for kw in foot_keywords):
            foot_ids.append(i)

    if not foot_ids:
        # Fallback: use the floor geom (id 0) or last geoms
        # In practice, the model XML should have named foot geoms
        pass

    return foot_ids


# ---------------------------------------------------------------------------
# Contact parameter injection helper
# ---------------------------------------------------------------------------

def inject_contact_params_to_all_feet(
    model: mjx.Model,
    foot_geom_ids: list[int],
    param_space: ParamSpace,
    p_normalized: jnp.ndarray,
) -> mjx.Model:
    """
    The main inject() only sets contact params on the representative foot geom.
    This helper propagates those values to ALL foot geoms for consistency.
    """
    contact_indices = param_space.group_indices("contact")
    if not contact_indices:
        return model

    p_phys = param_space.to_physical_vec(p_normalized)

    for param_idx in contact_indices:
        param = param_space.params[param_idx]
        val = p_phys[param_idx]
        arr = getattr(model, param.mjx_field)

        for gid in foot_geom_ids:
            if param.mjx_field == "geom_friction":
                arr = arr.at[gid, 0].set(val)
            elif param.mjx_field == "geom_solref":
                col = 0 if "stiffness" in param.name else 1
                arr = arr.at[gid, col].set(val)
            elif param.mjx_field == "geom_solimp":
                arr = arr.at[gid, 0].set(val)

        model = model.replace(**{param.mjx_field: arr})

    return model


# ---------------------------------------------------------------------------
# CLI: dump parameter space from a model file
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Inspect T1 parameter space")
    parser.add_argument("--model-xml", type=str, required=True,
                        help="Path to T1 MuJoCo XML model")
    parser.add_argument("--dump-defaults", action="store_true",
                        help="Print all parameters and their defaults")
    parser.add_argument("--save", type=str, default=None,
                        help="Save parameter space to JSON file")
    args = parser.parse_args()

    mj_model = mujoco.MjModel.from_xml_path(args.model_xml)
    ps = build_t1_param_space(mj_model)

    if args.dump_defaults:
        print(f"Total parameters: {ps.d}")
        print(f"  Friction:  {len(ps.group_indices('friction'))}")
        print(f"  Mass:      {len(ps.group_indices('mass'))}")
        print(f"  Actuator:  {len(ps.group_indices('actuator'))}")
        print(f"  Contact:   {len(ps.group_indices('contact'))}")
        print()
        for i, p in enumerate(ps.params):
            print(f"  [{i:3d}] {p.name:40s}  group={p.group:10s}  "
                  f"default={p.default:10.4f}  scale={p.scale:8.4f}  "
                  f"mode={p.mode}")

    if args.save:
        ps.save(args.save)
        print(f"\nSaved to {args.save}")
