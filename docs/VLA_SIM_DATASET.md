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

1. Run `scripts/probe_remote.sh` on the Isaac workstation and save the output.
2. Integrate `g1_bunny_vla.EpisodeWriter` into the Isaac control loop. A complete
   sample contract is in `examples/record_from_isaac.py`.
3. Record curriculum stages in order: static grasp, slow linear motion, then
   randomized moving trajectories.
4. Validate every episode before conversion:

   ```bash
   python -m g1_bunny_vla.validate_dataset /path/to/hdf5
   ```

5. Copy the official `prepare_data/hdf5_to_rlds/rlds_dataset` builder, replace
   its builder module with `rlds/g1_bunny_stop.py`, and build:

   ```bash
   G1_BUNNY_HDF5_GLOB='/data/g1_bunny_stop/*.hdf5' \
     tfds build --data_dir /data/rlds/g1_bunny_stop
   ```

6. Register `g1_bunny_stop` in a UniFoLM checkout:

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

