"""Small interface between an Isaac task/controller and the dataset writer."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .contract import DatasetContract, FrameSample
from .episode_writer import EpisodeWriter


class IsaacEpisodeSource(Protocol):
    """Implemented by the actual Isaac scene/controller.

    `step` must advance physics until the next 30 Hz capture boundary and return
    the synchronized observation plus the absolute expert action applied at that
    boundary. Returning `None` terminates the episode.
    """

    def reset(self, *, stage: str, seed: int) -> None: ...

    def step(self) -> tuple[FrameSample, str] | None: ...

    def succeeded(self) -> bool: ...


def record_episode(
    source: IsaacEpisodeSource,
    output_path: str | Path,
    *,
    stage: str,
    seed: int,
    instruction: str = "Grasp the moving bunny and bring it to a gentle stop.",
    max_seconds: float = 20.0,
) -> bool:
    contract = DatasetContract()
    max_frames = round(max_seconds * contract.fps)
    source.reset(stage=stage, seed=seed)
    writer = EpisodeWriter(
        output_path,
        instruction,
        metadata={"curriculum_stage": stage, "trajectory_seed": seed, "smoke_test": False},
    )
    try:
        for _ in range(max_frames):
            result = source.step()
            if result is None:
                break
            sample, reasoning = result
            writer.append(sample, reasoning)
        success = source.succeeded()
        writer.close(success=success)
        return success
    except BaseException:
        writer.close(success=False)
        raise

