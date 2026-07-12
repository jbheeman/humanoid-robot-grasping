from __future__ import annotations

from dataclasses import replace

import pytest

from object_tracking.arm_tracking.arm_bridge import (
    ArmBridgeConfig,
    ArmBridgeController,
    ArmBridgeError,
    ArmControlMode,
    ArmState,
    RobotState,
)
from object_tracking.arm_tracking.arm_commissioning import (
    OPERATOR_ACK,
    CommissioningConfig,
    CommissioningController,
    load_home_profile,
)
from object_tracking.arm_tracking.joints import RIGHT_ARM_JOINT_NAMES, joint_contract_id


class Clock:
    def __init__(self) -> None:
        self.value = 10.0

    def advance(self, seconds: float) -> None:
        self.value += seconds


class Hardware:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.state = RobotState(
            arm_q=(0.0,) * 14,
            arm_dq=(0.0,) * 14,
            received_at=clock.value,
            standing=True,
            standing_since=0.0,
            compatible_motion_mode=True,
            controller_available=True,
            motion_mode_name="normal",
            motion_mode_verified=True,
            controller_ownership_verified=True,
            motor_status_verified=True,
            motor_state_healthy=True,
        )
        self.commands = []

    def start(self) -> None:
        pass

    def latest_state(self) -> RobotState:
        return self.state

    def publish(self, command: object) -> None:
        self.commands.append(command)

    def close(self) -> None:
        pass

    def update(self, *, right_q: tuple[float, ...] | None = None) -> None:
        arm_q = self.state.arm_q if right_q is None else (*self.state.arm_q[:7], *right_q)
        self.state = replace(self.state, arm_q=arm_q, received_at=self.clock.value)


def setup(tmp_path, *, movable_joint_names=RIGHT_ARM_JOINT_NAMES):
    clock = Clock()
    hardware = Hardware(clock)
    bridge = ArmBridgeController(
        hardware,
        ArmBridgeConfig(
            control_mode=ArmControlMode.COMMISSIONING,
            allow_movement=True,
            joint_contract_id=joint_contract_id(),
            max_velocity_rad_s=0.10,
            max_acceleration_rad_s2=0.50,
            max_following_error_rad=0.05,
        ),
        monotonic=lambda: clock.value,
        wall_time=lambda: 1_800_000_000.0,
    )
    commissioning = CommissioningController(
        bridge,
        CommissioningConfig(
            movable_joint_names=tuple(movable_joint_names),
            research_root=tmp_path / "runs",
            profile_path=tmp_path / "right-arm-home.json",
        ),
        monotonic=lambda: clock.value,
        robot_identity={"model": "g1-29dof", "robot_id": "test"},
    )
    return clock, hardware, bridge, commissioning


def create_and_enable(clock, hardware, bridge, commissioning) -> str:
    report = commissioning.create_session(
        operator_ack=OPERATOR_ACK,
        operator="tester",
        client_id="pytest",
    )
    session_id = report["session_id"]
    commissioning.enable(session_id)
    clock.advance(bridge.config.weight_ramp_s)
    hardware.update()
    bridge.tick()
    commissioning.heartbeat(session_id)
    assert commissioning.report()["phase"] == "READY"
    return session_id


def settle(clock, hardware, bridge, commissioning, session_id, target):
    for _ in range(7):
        clock.advance(0.1)
        hardware.update(right_q=tuple(target))
        commissioning.heartbeat(session_id)
        bridge.tick()
        commissioning.report()


def test_commissioning_requires_verified_mode_and_ownership(tmp_path) -> None:
    clock, hardware, _bridge, commissioning = setup(tmp_path)
    report = commissioning.create_session(
        operator_ack=OPERATOR_ACK,
        operator="tester",
        client_id="pytest",
    )
    hardware.state = replace(hardware.state, motion_mode_verified=False)

    with pytest.raises(ArmBridgeError) as rejected:
        commissioning.enable(report["session_id"])

    assert rejected.value.code == "motion_mode_unverified"


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"controller_ownership_verified": False}, "controller_ownership_unverified"),
        (
            {"motor_status_verified": True, "motor_state_healthy": False},
            "motor_state_fault",
        ),
    ],
)
def test_commissioning_rejects_unverified_live_gates(tmp_path, changes, code) -> None:
    clock, hardware, _bridge, commissioning = setup(tmp_path)
    report = commissioning.create_session(
        operator_ack=OPERATOR_ACK,
        operator="tester",
        client_id="pytest",
    )
    hardware.state = replace(hardware.state, **changes)

    with pytest.raises(ArmBridgeError) as rejected:
        commissioning.enable(report["session_id"])

    assert rejected.value.code == code


def test_one_joint_jog_settles_and_requires_confirmation(tmp_path) -> None:
    clock, hardware, bridge, commissioning = setup(tmp_path)
    session_id = create_and_enable(clock, hardware, bridge, commissioning)

    report = commissioning.jog(
        session_id,
        sequence=1,
        joint_name=RIGHT_ARM_JOINT_NAMES[0],
        direction=1,
    )
    target = report["bridge"]["commanded_arm_q"][7:]
    assert target[0] == pytest.approx(0.0)
    assert bridge.desired_right == pytest.approx((0.01, 0, 0, 0, 0, 0, 0))
    settle(clock, hardware, bridge, commissioning, session_id, bridge.desired_right)
    assert commissioning.report()["phase"] == "AWAITING_CONFIRM"

    with pytest.raises(ArmBridgeError) as busy:
        commissioning.jog(
            session_id,
            sequence=2,
            joint_name=RIGHT_ARM_JOINT_NAMES[1],
            direction=1,
        )
    assert busy.value.code == "motion_busy"

    report = commissioning.confirm_motion(
        session_id,
        sequence=2,
        outcome="accepted",
    )
    assert report["phase"] == "READY"
    assert report["measured_q"] == pytest.approx([0.01, 0, 0, 0, 0, 0, 0])


def test_commissioning_rejects_joints_not_enabled_by_configuration(tmp_path) -> None:
    clock, _hardware, _bridge, commissioning = setup(
        tmp_path, movable_joint_names=RIGHT_ARM_JOINT_NAMES[:3]
    )
    session_id = create_and_enable(clock, _hardware, _bridge, commissioning)

    with pytest.raises(ArmBridgeError) as rejected:
        commissioning.jog(
            session_id,
            sequence=1,
            joint_name="right_elbow_joint",
            direction=1,
        )

    assert rejected.value.code == "joint_not_enabled"

def test_sign_check_requires_matching_encoder_delta(tmp_path) -> None:
    clock, hardware, bridge, commissioning = setup(tmp_path)
    session_id = create_and_enable(clock, hardware, bridge, commissioning)
    commissioning.jog(
        session_id,
        sequence=1,
        joint_name=RIGHT_ARM_JOINT_NAMES[0],
        direction=1,
        kind="sign_check",
    )
    wrong_target = (-0.01, 0, 0, 0, 0, 0, 0)
    settle(clock, hardware, bridge, commissioning, session_id, wrong_target)
    # Settling is relative to the commanded target, so place the arm at the target
    # before confirming but alter the recorded baseline to emulate a reversed encoder.
    session = commissioning.session
    assert session is not None and session.pending is not None
    session.pending.baseline_q = (0.02, 0, 0, 0, 0, 0, 0)
    hardware.update(right_q=tuple(bridge.desired_right))
    settle(clock, hardware, bridge, commissioning, session_id, bridge.desired_right)

    with pytest.raises(ArmBridgeError) as rejected:
        commissioning.confirm_motion(session_id, sequence=2, outcome="expected")

    assert rejected.value.code == "sign_measurement_mismatch"


def test_stage_and_session_envelopes_are_fail_closed(tmp_path) -> None:
    clock, hardware, bridge, commissioning = setup(tmp_path)
    session_id = create_and_enable(clock, hardware, bridge, commissioning)

    for sequence in range(1, 6):
        commissioning.jog(
            session_id,
            sequence=sequence * 2 - 1,
            joint_name=RIGHT_ARM_JOINT_NAMES[0],
            direction=1,
        )
        settle(clock, hardware, bridge, commissioning, session_id, bridge.desired_right)
        commissioning.confirm_motion(
            session_id,
            sequence=sequence * 2,
            outcome="accepted",
        )

    with pytest.raises(ArmBridgeError) as rejected:
        commissioning.jog(
            session_id,
            sequence=11,
            joint_name=RIGHT_ARM_JOINT_NAMES[0],
            direction=1,
        )
    assert rejected.value.code == "stage_limit"
    commissioning.checkpoint(session_id, label="stage-1")
    assert commissioning.report()["stage_baseline_q"][0] == pytest.approx(0.05)


def test_left_arm_drift_faults_commissioning(tmp_path) -> None:
    clock, hardware, bridge, commissioning = setup(tmp_path)
    session_id = create_and_enable(clock, hardware, bridge, commissioning)
    commissioning.heartbeat(session_id)
    hardware.state = replace(
        hardware.state,
        arm_q=(0.02,) + hardware.state.arm_q[1:],
        received_at=clock.value,
    )

    bridge.tick()

    assert bridge.state is ArmState.FAULT
    assert bridge.fault_reason == "left_arm_drift"


def test_sign_check_evidence_and_profile_promotion(tmp_path) -> None:
    clock, hardware, bridge, commissioning = setup(
        tmp_path, movable_joint_names=RIGHT_ARM_JOINT_NAMES
    )
    session_id = create_and_enable(clock, hardware, bridge, commissioning)
    sequence = 0
    for name in RIGHT_ARM_JOINT_NAMES:
        for direction in (-1, 1):
            sequence += 1
            commissioning.jog(
                session_id,
                sequence=sequence,
                joint_name=name,
                direction=direction,
                kind="sign_check",
            )
            settle(clock, hardware, bridge, commissioning, session_id, bridge.desired_right)
            sequence += 1
            commissioning.confirm_motion(
                session_id,
                sequence=sequence,
                outcome="expected",
            )
        commissioning.checkpoint(session_id, label=f"verified-{name}")

    report = commissioning.capture_candidate(session_id, label="safe_chest")
    assert report["candidate"]["label"] == "safe_chest"
    for _ in range(2):
        for _ in range(2):
            sequence += 1
            commissioning.jog(
                session_id,
                sequence=sequence,
                joint_name=RIGHT_ARM_JOINT_NAMES[0],
                direction=1,
            )
            settle(clock, hardware, bridge, commissioning, session_id, bridge.desired_right)
            sequence += 1
            commissioning.confirm_motion(
                session_id,
                sequence=sequence,
                outcome="accepted",
            )
        assert commissioning.report()["candidate"]["departed"] is True
        for _ in range(2):
            sequence += 1
            commissioning.replay_step(session_id, sequence=sequence)
            settle(clock, hardware, bridge, commissioning, session_id, bridge.desired_right)
            sequence += 1
            commissioning.confirm_motion(
                session_id,
                sequence=sequence,
                outcome="accepted",
            )
        commissioning.validate_replay(session_id)
    commissioning.stop(session_id)
    clock.advance(bridge.config.weight_ramp_s)
    hardware.update()
    bridge.tick()
    assert bridge.state is ArmState.DISARMED

    profile = commissioning.promote(session_id)
    assert profile["label"] == "safe_chest"
    assert commissioning.config.profile_path.is_file()
    assert (
        load_home_profile(
            commissioning.config.profile_path,
            expected_robot_id="test",
        )["profile_hash"]
        == profile["profile_hash"]
    )
    with pytest.raises(ArmBridgeError) as wrong_robot:
        load_home_profile(
            commissioning.config.profile_path,
            expected_robot_id="another-robot",
        )
    assert wrong_robot.value.code == "invalid_home_profile"

    tampered = commissioning.config.profile_path.read_text(encoding="utf-8").replace(
        '"label": "safe_chest"', '"label": "unsafe"'
    )
    commissioning.config.profile_path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ArmBridgeError) as rejected:
        load_home_profile(commissioning.config.profile_path)
    assert rejected.value.code == "invalid_home_profile"
