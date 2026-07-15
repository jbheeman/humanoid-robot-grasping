"""Register the G1 bunny dataset in an external UniFoLM-VLA checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def edit_once(path: Path, needle: str, replacement: str) -> bool:
    text = path.read_text()
    if replacement in text:
        return False
    if needle not in text:
        raise RuntimeError(f"registration anchor not found in {path}: {needle!r}")
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(needle, replacement, 1))
    return True


def register(root: Path) -> list[Path]:
    base = root / "src/unifolm_vla/rlds_dataloader"
    configs = base / "datasets/rlds/oxe/configs.py"
    transforms = base / "datasets/rlds/oxe/transforms.py"
    mixtures = base / "datasets/rlds/oxe/mixtures.py"
    datasets = base / "datasets/datasets.py"
    constants = base / "constants.py"
    changed: list[Path] = []

    config_entry = '''    "g1_bunny_stop": {
        "image_obs_keys": {"primary": "image_left_top", "secondary": None, "left_wrist": "image_left_wrist", "right_wrist": "image_right_wrist"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.EE_R6_G1,
        "action_encoding": ActionEncoding.EE_R6_G1,
    },
'''
    if edit_once(configs, "OXE_DATASET_CONFIGS = {\n", "OXE_DATASET_CONFIGS = {\n" + config_entry):
        changed.append(configs)
    transform_entry = '    "g1_bunny_stop": unitree_g1_ee_6d_dataset_transform,\n'
    if edit_once(
        transforms,
        "OXE_STANDARDIZATION_TRANSFORMS = {\n",
        "OXE_STANDARDIZATION_TRANSFORMS = {\n" + transform_entry,
    ):
        changed.append(transforms)
    mixture_anchor = '    "g1_stack_block":[\n'
    mixture_entry = '    "g1_bunny_stop": [("g1_bunny_stop", 1.0)],\n'
    if edit_once(mixtures, mixture_anchor, mixture_entry + mixture_anchor):
        changed.append(mixtures)
    camera_anchor = 'elif "g1_stack_block" in self.data_mix:'
    camera_replacement = 'elif "g1_stack_block" in self.data_mix or "g1_bunny_stop" in self.data_mix:'
    if edit_once(datasets, camera_anchor, camera_replacement):
        changed.append(datasets)
    constant_anchor = 'elif "stack_block" in cmd_args:'
    constant_replacement = 'elif "stack_block" in cmd_args or "bunny_stop" in cmd_args:'
    if edit_once(constants, constant_anchor, constant_replacement):
        changed.append(constants)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("unifolm_checkout", type=Path)
    args = parser.parse_args()
    changed = register(args.unifolm_checkout.resolve())
    if changed:
        print("Registered g1_bunny_stop in:")
        for path in changed:
            print(path)
    else:
        print("g1_bunny_stop was already registered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

