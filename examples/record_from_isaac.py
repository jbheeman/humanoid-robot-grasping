"""Integration sketch for an Isaac controller.

Run this with Isaac Sim's Python after replacing `MyIsaacTask` with the scene's
real articulation, camera, contact-sensor, and expert-controller bindings.
"""

from g1_bunny_vla.isaac_adapter import record_episode


# The adapter deliberately does not import a version-specific Isaac API. This
# keeps the dataset contract stable across Isaac Sim releases. Your adapter must
# implement reset(stage, seed), step(), and succeeded(); see IsaacEpisodeSource.
from my_isaac_task import MyIsaacTask  # type: ignore[import-not-found]


task = MyIsaacTask(capture_hz=30, resolution=(640, 480))
record_episode(
    task,
    "/data/g1_bunny_stop/episode_000000.hdf5",
    stage="static_grasp",
    seed=0,
)

