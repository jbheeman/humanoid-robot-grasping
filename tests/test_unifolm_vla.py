import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from object_tracking.unifolm_vla import (
    compose_pose23,
    parse_action_chunk,
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
)
from object_tracking.unifolm_vla_cli import (
    GuardedVLAExecutor,
    LiveObservationSource,
    Observation,
    UnifoLMRuntime,
    VLAError,
    _format_xyz,
    _is_explicit_right_arm_instruction,
    _load_checkpoint_state,
    _profile_gripper_means,
)


def test_rotation_6d_round_trip() -> None:
    angle = 0.4
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    encoded = rotation_matrix_to_6d(rotation)
    np.testing.assert_allclose(rotation_6d_to_matrix(encoded), rotation, atol=1e-8)


def test_pose23_uses_unitree_gripper_and_waist_order() -> None:
    left = np.eye(4)
    right = np.eye(4)
    left[:3, 3] = (0.2, 0.3, 0.4)
    right[:3, 3] = (0.5, -0.2, 0.1)
    pose = compose_pose23(
        left,
        right,
        right_gripper=4.5,
        left_gripper=3.5,
        waist_yaw_roll_pitch=(0.1, 0.2, 0.3),
    )
    assert len(pose) == 23
    assert pose[0:3] == (0.2, 0.3, 0.4)
    assert pose[9:12] == (0.5, -0.2, 0.1)
    assert pose[18:23] == (4.5, 3.5, 0.1, 0.2, 0.3)


def test_action_chunk_rejects_degenerate_rotation() -> None:
    invalid = [0.0] * 23
    with pytest.raises(ValueError, match="degenerate"):
        parse_action_chunk([invalid])


def test_no_hand_grippers_use_profile_means(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    checkpoint = model_root / "checkpoints/pytorch_model.pt"
    checkpoint.parent.mkdir(parents=True)
    means = [0.0] * 23
    means[18:20] = [3.95, 4.03]
    (model_root / "dataset_statistics.json").write_text(
        json.dumps({"g1_stack_block": {"proprio": {"mean": means}}}),
        encoding="utf-8",
    )

    assert _profile_gripper_means(checkpoint, "g1_stack_block") == (3.95, 4.03)


def test_runtime_background_load_is_single_and_joined_by_foreground(tmp_path: Path) -> None:
    runtime = UnifoLMRuntime(tmp_path / "checkpoint", tmp_path / "vlm", "profile")
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def load_model() -> None:
        calls.append(True)
        entered.set()
        assert release.wait(1.0)
        runtime.model = object()

    runtime._load_model = load_model
    runtime.start_loading()
    assert entered.wait(1.0)
    assert runtime.load_status().startswith("loading in background")

    release.set()
    runtime.load()

    assert runtime.load_status() == "ready"
    assert len(calls) == 1


def test_quantized_checkpoint_accepts_only_bitsandbytes_metadata() -> None:
    class Model:
        def load_state_dict(self, state: object, *, strict: bool) -> object:
            assert state == {"weight": object_state}
            assert strict is False
            return SimpleNamespace(
                missing_keys=[],
                unexpected_keys=[
                    "vlm.layer.weight.absmax",
                    "vlm.layer.weight.quant_state.bitsandbytes__nf4",
                ],
            )

    object_state = object()
    _load_checkpoint_state(Model(), {"weight": object_state}, quantized=True)


def test_quantized_checkpoint_rejects_real_schema_mismatch() -> None:
    class Model:
        def load_state_dict(self, state: object, *, strict: bool) -> object:
            del state, strict
            return SimpleNamespace(
                missing_keys=["action_model.weight"],
                unexpected_keys=["wrong.weight"],
            )

    with pytest.raises(VLAError, match="missing keys: action_model.weight"):
        _load_checkpoint_state(Model(), {}, quantized=True)


@pytest.mark.parametrize(
    "instruction",
    (
        "raise the right hand slightly",
        "move right arm forward",
        "touch the plush with the right palm",
    ),
)
def test_explicit_right_arm_instruction_is_accepted(instruction: str) -> None:
    assert _is_explicit_right_arm_instruction(instruction)


@pytest.mark.parametrize(
    "instruction",
    ("hi", "grab the plush", "move left hand", "right hand", "be careful"),
)
def test_ambiguous_execution_instruction_is_rejected(instruction: str) -> None:
    assert not _is_explicit_right_arm_instruction(instruction)


def test_xyz_display_has_explicit_sign_and_metric_units() -> None:
    assert _format_xyz((0.02, -0.004, 0.007)) == "[+0.020, -0.004, +0.007] m"


def test_guarded_executor_clears_then_streams_bounded_vla_waypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    support = SimpleNamespace(certified_edges=("u_min", "u_max", "v_min", "v_max"))
    start = (0.0,) * 7
    cleared = (0.01,) * 7
    moved = (0.02,) * 7

    class Solver:
        def plan_guided_clearance(self, q: object, *, support_plane: object) -> object:
            assert tuple(q) == start and support_plane is support
            return SimpleNamespace(ok=True, q_path=(start, cleared), reason="")

        def solve_local_translation(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            return SimpleNamespace(ok=True, q_rad=moved, reason="")

        def validate_joint_path(self, path: object, *, support_plane: object) -> None:
            assert tuple(path) == (cleared, moved) and support_plane is support
            return None

    class Transport:
        def __init__(self) -> None:
            self.targets: list[tuple[float, ...]] = []
            self.stops: list[str] = []
            self.heartbeats = 0
            self.state = "DISARMED"
            self.sessions: list[str] = []

        def enable_arm(self, session: str, calibration: str) -> dict[str, object]:
            assert session.startswith("vla-") and calibration == "calibration"
            self.sessions.append(session)
            self.state = "ARMED"
            return {"ok": True}

        def arm_state(self) -> dict[str, object]:
            return {"state": self.state}

        def publish_target(self, *args: object, **kwargs: object) -> None:
            del kwargs
            self.targets.append(tuple(args[3]))

        def heartbeat_arm(self, session: str) -> None:
            assert session.startswith("vla-")
            self.heartbeats += 1

        def stop_arm(self, reason: str) -> None:
            self.stops.append(reason)
            self.state = "DISARMED"

    source = LiveObservationSource.__new__(LiveObservationSource)
    source.last_state = {
        "calibration_id": "calibration",
        "measured_arm_q": [0.0] * 14,
        "visualization": {},
    }

    def snapshot() -> Observation:
        source.last_state["measured_arm_q"] = [0.0] * 7 + list(cleared)
        return Observation(Image.new("RGB", (8, 8)), (0.0,) * 23, 0.0, 0.0)

    source.snapshot = snapshot
    action = np.asarray(
        [compose_pose23(np.eye(4), np.eye(4), right_gripper=4.0, left_gripper=4.0, waist_yaw_roll_pitch=(0, 0, 0))]
    )
    def predict(observation: object, instruction: str) -> tuple[np.ndarray, float]:
        del observation, instruction
        threading.Event().wait(0.12)
        return action, 0.1

    runtime = SimpleNamespace(predict=predict)
    executor = GuardedVLAExecutor.__new__(GuardedVLAExecutor)
    executor.solver = Solver()
    executor.calibration_id = "calibration"
    executor.calibrated_support = support
    executor.period_s = 0.1
    executor.max_waypoints = 1
    executor.direct_vla_waypoint = False
    executor.transport = Transport()
    monkeypatch.setattr("object_tracking.unifolm_vla_cli.time.sleep", lambda _: None)

    assert executor.execute(source, runtime, "move the hand") == (1, 0.1)
    assert executor.transport.targets == [cleared, moved]
    assert executor.transport.heartbeats >= 2
    assert len(set(executor.transport.sessions)) == 2
    assert executor.transport.stops == [
        "vla_clearance_complete",
        "vla_smoke_test_complete",
    ]


def test_direct_vla_waypoint_skips_table_clearance_but_keeps_bounded_ik(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = (0.0,) * 7
    moved = (0.01,) * 7

    class Solver:
        def solve_local_translation(self, target: object, q: object, **kwargs: object) -> object:
            del target
            assert tuple(q) == start
            assert kwargs["support_plane"] is None
            assert kwargs["maximum_joint_step_rad"] == 0.025
            assert kwargs["validate_path"] is False
            return SimpleNamespace(ok=True, q_rad=moved, reason=None)

        def validate_joint_path(self, path: object, *, support_plane: object) -> None:
            del path, support_plane
            raise AssertionError("direct mode must not run path validation")

        def plan_guided_clearance(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("direct mode must not plan table/hip clearance")

    class Transport:
        def __init__(self) -> None:
            self.targets = []
            self.state = "DISARMED"

        def enable_arm(self, session: str, calibration: str) -> dict[str, object]:
            del session, calibration
            self.state = "ARMED"
            return {"ok": True}

        def heartbeat_arm(self, session: str) -> dict[str, object]:
            del session
            return {"ok": True}

        def arm_state(self) -> dict[str, object]:
            return {"state": self.state}

        def publish_target(self, *args: object, **kwargs: object) -> None:
            del kwargs
            self.targets.append(tuple(args[3]))

        def stop_arm(self, reason: str) -> None:
            assert reason == "direct_vla_smoke_test_complete"
            self.state = "DISARMED"

    source = LiveObservationSource.__new__(LiveObservationSource)
    source.last_state = {
        "calibration_id": "calibration",
        "measured_arm_q": [0.0] * 14,
        "visualization": {},
    }
    action = np.asarray(
        [compose_pose23(np.eye(4), np.eye(4), right_gripper=4.0, left_gripper=4.0, waist_yaw_roll_pitch=(0, 0, 0))]
    )
    executor = GuardedVLAExecutor.__new__(GuardedVLAExecutor)
    executor.solver = Solver()
    executor.calibration_id = "calibration"
    executor.calibrated_support = SimpleNamespace(
        certified_edges=("u_min", "u_max", "v_min", "v_max")
    )
    executor.period_s = 0.1
    executor.max_waypoints = 1
    executor.direct_vla_waypoint = True
    executor.transport = Transport()
    monkeypatch.setattr("object_tracking.unifolm_vla_cli.time.sleep", lambda _: None)

    assert executor.execute(
        source,
        SimpleNamespace(),
        "raise the right hand",
        proposed_action=action,
        proposal_inference_s=0.2,
    ) == (1, 0.2)
    assert executor.transport.targets == [moved]
