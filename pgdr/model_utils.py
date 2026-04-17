from __future__ import annotations

from pathlib import Path
from typing import Optional

import mujoco


_PLAYGROUND_T1_ALIASES = {
    "playground:t1",
    "playground://t1",
    "t1",
    "booster_t1",
}


def resolve_model_xml(model_xml: str) -> str:
    """Resolve an on-disk model XML path or preserve a supported alias."""
    if model_xml in _PLAYGROUND_T1_ALIASES:
        return model_xml
    return str(Path(model_xml).expanduser().resolve())


def load_mj_model(model_xml: str, ccd_iterations: int = 500) -> mujoco.MjModel:
    """Load the MuJoCo model after resolving aliases and relative paths."""
    if model_xml in _PLAYGROUND_T1_ALIASES:
        from mujoco_playground._src.locomotion.t1 import base

        assets = base.get_assets()
        xml = assets["scene_mjx_feetonly_flat_terrain.xml"].decode()
        m = mujoco.MjModel.from_xml_string(xml, assets)
    else:
        m = mujoco.MjModel.from_xml_path(resolve_model_xml(model_xml))

    m.opt.ccd_iterations = ccd_iterations
    return m


def resolve_param_space_path(
    explicit_path: Optional[str],
    results_dir: Optional[str] = None,
) -> Optional[str]:
    """Prefer an explicit path, else auto-discover from a results directory."""
    if explicit_path:
        return str(Path(explicit_path).expanduser().resolve())

    if results_dir:
        candidate = Path(results_dir).expanduser().resolve() / "param_space.json"
        if candidate.exists():
            return str(candidate)

    return None
