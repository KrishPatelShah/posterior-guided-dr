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


def load_mj_model(model_xml: str, njmax: int = 500, nconmax: int = 200) -> mujoco.MjModel:
    """Load the MuJoCo model after resolving aliases and relative paths.

    njmax/nconmax are increased from their defaults to prevent 'nefc overflow'
    warnings during mujoco_warp parallel rollouts with large population sizes.
    """
    if model_xml in _PLAYGROUND_T1_ALIASES:
        from mujoco_playground._src.locomotion.t1 import base
        import re

        assets = base.get_assets()
        xml = assets["scene_mjx_feetonly_flat_terrain.xml"].decode()

        # Inject or replace <size> element to avoid nefc overflow in warp rollouts
        size_tag = f'<size njmax="{njmax}" nconmax="{nconmax}"/>'
        if "<size" in xml:
            xml = re.sub(r"<size[^/]*/?>", size_tag, xml)
        else:
            xml = re.sub(r"(<mujoco[^>]*>)", r"\1\n  " + size_tag, xml)

        return mujoco.MjModel.from_xml_string(xml, assets)

    return mujoco.MjModel.from_xml_path(resolve_model_xml(model_xml))


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
