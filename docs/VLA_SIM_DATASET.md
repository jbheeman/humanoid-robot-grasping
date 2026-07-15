# Unitree G1 moving-plush VLA dataset

This branch builds an Isaac Sim dataset for fine-tuning
[Unitree UniFoLM-VLA](https://github.com/unitreerobotics/unifolm-vla) to intercept,
grasp, and stop a moving bunny plush.

The format intentionally mirrors Unitree's
[`G1_Dex1_Stack_Block`](https://huggingface.co/datasets/unitreerobotics/G1_Dex1_Stack_Block)
dataset and the official UniFoLM conversion code:

- 30 Hz episodes with four 640x480 RGB views.
- 19-D joint state/action: left arm (7), right arm (7), right gripper
  (1), left gripper (1), waist RPY (3).
- 17-D end-effector state/action: left XYZ/RPY (6), right XYZ/RPY (6),
  right gripper (1), left gripper (1), waist RPY (3).
- Extra plush pose, plush velocity, contact force, and outcome signals for
  evaluation. UniFoLM ignores these extra fields during training.
- RLDS conversion expands the 17-D end-effector representation to the 23-D
  representation used by UniFoLM: XYZ + 6-D rotation for each arm, both
  grippers, and waist RPY.

## Workflow

1. Generate the measured bunny proxy with Isaac Sim's Python. The first asset
   represents the plush as a non-compressible rigid frame with a soft-looking
   visual surface: 15 cm front-to-back, 7 cm side-to-side, 14 cm tall, 150 g.

   ```bash
   G1_BUNNY_PROJECT_ROOT=/path/to/humanoid-robot-grasping \
     /path/to/isaac-python scripts/create_bunny_proxy_usd.py \
     --output /path/to/assets/g1_bunny_proxy.usda --headless
   ```

   It is intended for slow hand-push trajectories of 0.03--0.15 m/s on a
   tabletop; do not label it as a deformable simulation.

2. Run `scripts/probe_remote.sh` on the Isaac workstation and save the output.
3. Integrate `g1_bunny_vla.EpisodeWriter` into the Isaac control loop. A complete
   sample contract is in `examples/record_from_isaac.py`.
4. Record curriculum stages in order: static grasp, slow linear motion, then
   randomized moving trajectories.
5. Validate every episode before conversion:

   ```bash
   python -m g1_bunny_vla.validate_dataset /path/to/hdf5
   ```

6. Build RLDS directly with the project builder. This avoids the optional
   Apache Beam dependency pulled in by the `tfds` command-line wrapper:

   ```bash
   python scripts/build_rlds.py \
     --input '/data/g1_bunny_stop/episode_*.hdf5' \
     --output /data/rlds
   ```

7. Register `g1_bunny_stop` in a UniFoLM checkout:

   ```bash
   python scripts/register_unifolm_dataset.py /path/to/unifolm-vla
   ```

The registration script is idempotent and creates `.bak` files before changing
the external checkout.

## Dataset acceptance gates

An episode is rejected when required arrays or cameras are missing, dimensions
do not match the Unitree contract, values are non-finite, timestamps are not
strictly increasing, or the effective frame rate differs from 30 Hz by more
than the configured tolerance. Training and held-out episode IDs must be split
by trajectory seed, not by frame, to avoid near-duplicate leakage.

Synthetic smoke episodes are useful only to verify plumbing and are marked
`smoke_test=true`; they must never be included in model training.
