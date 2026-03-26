"""
Interactive MuJoCo viewer for T1.

Opens a live 3D window showing the T1 robot.  Use the mouse to orbit/pan/zoom.

Usage:
    python pgdr/test/live_viewer.py [--mode {zero,sine,random}] [--speed SPEED]

Controls (MuJoCo viewer):
    Left-drag   : orbit
    Right-drag  : pan
    Scroll      : zoom
    Space       : pause/resume
    Backspace   : reset
    Ctrl+Q / Esc: quit
"""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Load T1 model via mujoco_playground assets
# ---------------------------------------------------------------------------

def load_t1():
    from mujoco_playground._src.locomotion.t1 import base
    from mujoco_playground._src import mjx_env

    assets = base.get_assets()
    menagerie_path = mjx_env.MENAGERIE_PATH / "booster_t1"
    mjx_env.update_assets(assets, menagerie_path, "*.xml")
    mjx_env.update_assets(assets, menagerie_path / "assets")

    xml = assets["scene_mjx_feetonly_flat_terrain.xml"].decode()
    return mujoco.MjModel.from_xml_string(xml, assets)


# ---------------------------------------------------------------------------
# Action generators
# ---------------------------------------------------------------------------

def sine_action(t: float, nu: int) -> np.ndarray:
    action = np.zeros(nu)
    action[0::3] = 0.15 * np.sin(2 * np.pi * 0.5 * t)
    action[1::3] = 0.08 * np.sin(2 * np.pi * 0.5 * t + 0.5)
    return action


def random_action(t: float, nu: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-0.3, 0.3, size=nu).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero", "sine", "random"], default="zero",
                        help="Action mode (default: zero)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Simulation speed multiplier (default: 1.0)")
    args = parser.parse_args()

    print("Loading T1 model...")
    mj_model = load_t1()
    mj_data = mujoco.MjData(mj_model)

    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    else:
        mujoco.mj_resetData(mj_model, mj_data)

    print(f"  nq={mj_model.nq}, nu={mj_model.nu}, nbody={mj_model.nbody}")
    print(f"  Action mode: {args.mode}")

    print("  Opening viewer... (Space=run sim, Ctrl+Q=quit)\n")
    mujoco.viewer.launch(mj_model, mj_data)


if __name__ == "__main__":
    main()
