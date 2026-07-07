"""Steady two-arm grasping for the Unitree G1 (arms and hands only).

This module drives *only* the upper-body arm joints and the Dex3-1 hands. It
never commands the legs or the waist, so the loco/balance controller keeps the
robot standing while the arms reach out, close the hands, and hold an object
firmly.

How it works
------------
The G1 exposes a low-level "arm SDK" DDS channel (``rt/arm_sdk``) that accepts a
``unitree_hg`` ``LowCmd_``. Each entry in ``motor_cmd`` is a position/torque
target for one joint. A dedicated blending joint (``kNotUsedJoint`` = index 29)
carries a *weight* in ``[0, 1]`` in its ``q`` field:

* ``weight = 0`` -> arm_sdk contributes nothing, the built-in controller owns
  the arms (safe idle / release state).
* ``weight = 1`` -> arm_sdk fully owns every joint we give a non-zero ``kp``.

We only assign gains to the 14 arm joints (indices 15-28), so ramping the weight
to 1 hands us the arms *and nothing else*. Joint targets read back from
``rt/lowstate``; we interpolate smoothly from the measured pose so there is no
jerk when we take control.

The Dex3-1 fingers are commanded on the separate ``rt/dex3/left/cmd`` and
``rt/dex3/right/cmd`` channels.

Safety / tuning notes
---------------------
* Stand the robot up first: ``uv run loco <ip> stand_up`` then
  ``uv run loco <ip> balance_stand``.
* The joint targets in :data:`DEFAULT_POSES` and the hand open/close vectors are
  reasonable starting points but **must be tuned on your hardware** -- sign
  conventions and reachable ranges vary per unit. Use ``--dry-run`` (logs the
  planned motion without publishing) and ``--check`` (prints the live joint
  angles) before commanding real motion, and keep a hand on the e-stop.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from object_tracking.unitree_g1 import UnitreeG1Error, resolve_network_interface

# --- Unitree SDK2 (unitree_hg) imports, guarded like unitree_g1.py -----------
try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
except Exception as exc:  # pragma: no cover - surfaced at runtime when missing
    ChannelFactoryInitialize = None
    ChannelPublisher = None
    ChannelSubscriber = None
    unitree_hg_msg_dds__LowCmd_ = None
    LowCmd_ = None
    LowState_ = None
    CRC = None
    _ARM_SDK_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover
    _ARM_SDK_IMPORT_ERROR = None

# Dex3-1 hand messages live in the same package but are optional (some units
# ship a simple gripper or no hand at all), so import them separately.
try:
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
except Exception as exc:  # pragma: no cover
    unitree_hg_msg_dds__HandCmd_ = None
    HandCmd_ = None
    _HAND_SDK_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover
    _HAND_SDK_IMPORT_ERROR = None


# --- Joint layout ------------------------------------------------------------
class G1JointIndex:
    """Motor indices for the G1 (29 DOF) upper body.

    Only the arm joints (15-28) are ever commanded by this module. ``WaistYaw``
    etc. are listed for reference but intentionally left under loco control.
    ``kNotUsedJoint`` (29) carries the arm_sdk blend weight in its ``q`` field.
    """

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20
    LeftWristYaw = 21
    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27
    RightWristYaw = 28
    # Waist (NOT commanded here)
    WaistYaw = 12
    WaistRoll = 13
    WaistPitch = 14
    # arm_sdk blend-weight carrier
    kNotUsedJoint = 29


LEFT_ARM_JOINTS: tuple[int, ...] = (
    G1JointIndex.LeftShoulderPitch,
    G1JointIndex.LeftShoulderRoll,
    G1JointIndex.LeftShoulderYaw,
    G1JointIndex.LeftElbow,
    G1JointIndex.LeftWristRoll,
    G1JointIndex.LeftWristPitch,
    G1JointIndex.LeftWristYaw,
)
RIGHT_ARM_JOINTS: tuple[int, ...] = (
    G1JointIndex.RightShoulderPitch,
    G1JointIndex.RightShoulderRoll,
    G1JointIndex.RightShoulderYaw,
    G1JointIndex.RightElbow,
    G1JointIndex.RightWristRoll,
    G1JointIndex.RightWristPitch,
    G1JointIndex.RightWristYaw,
)
ARM_JOINTS: tuple[int, ...] = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

# Order of the 7-element per-arm vectors used to describe a pose:
#   [ShoulderPitch, ShoulderRoll, ShoulderYaw, Elbow, WristRoll, WristPitch, WristYaw]


def joints_for_side(side: str) -> tuple[int, ...]:
    if side == "left":
        return LEFT_ARM_JOINTS
    if side == "right":
        return RIGHT_ARM_JOINTS
    if side == "both":
        return ARM_JOINTS
    raise ValueError(f"side must be 'left', 'right', or 'both', got {side!r}")


def make_pose(left7: list[float], right7: list[float]) -> dict[int, float]:
    """Build a ``{joint_index: radians}`` pose from per-arm 7-vectors."""
    if len(left7) != 7 or len(right7) != 7:
        raise ValueError("left7 and right7 must each have 7 elements")
    pose: dict[int, float] = {}
    for value, joint in zip(left7, LEFT_ARM_JOINTS):
        pose[joint] = float(value)
    for value, joint in zip(right7, RIGHT_ARM_JOINTS):
        pose[joint] = float(value)
    return pose


@dataclass(frozen=True)
class GraspPoses:
    """Named arm poses for the grasp sequence (radians).

    TUNE THESE ON HARDWARE. Values are approximate and mirrored left/right.
    ``home`` is arms-at-sides; ``ready`` lifts the arms forward with the elbows
    bent and hands apart; ``reach`` extends forward and brings the hands toward
    the centreline to close around an object in front of the chest; ``lift``
    raises the grasped object slightly by pitching the shoulders up.
    """

    home: dict[int, float] = field(
        default_factory=lambda: make_pose([0.0] * 7, [0.0] * 7)
    )
    ready: dict[int, float] = field(
        default_factory=lambda: make_pose(
            #   sPitch sRoll sYaw  elbow wRoll wPitch wYaw
            [-0.30, 0.25, 0.00, 0.90, 0.00, 0.00, 0.00],
            [-0.30, -0.25, 0.00, 0.90, 0.00, 0.00, 0.00],
        )
    )
    reach: dict[int, float] = field(
        default_factory=lambda: make_pose(
            [-0.50, 0.15, 0.10, 0.60, 0.00, 0.20, 0.00],
            [-0.50, -0.15, -0.10, 0.60, 0.00, 0.20, 0.00],
        )
    )
    lift: dict[int, float] = field(
        default_factory=lambda: make_pose(
            [-0.70, 0.18, 0.10, 0.75, 0.00, 0.20, 0.00],
            [-0.70, -0.18, -0.10, 0.75, 0.00, 0.20, 0.00],
        )
    )


DEFAULT_POSES = GraspPoses()


# --- Dex3-1 hand -------------------------------------------------------------
DEX3_NUM_MOTORS = 7

# Open = fingers extended; close = curled to grip. Per-hand 7-vectors, TUNE ON
# HARDWARE (joint order and sign depend on the Dex3-1 firmware/mount).
DEX3_LEFT_OPEN: tuple[float, ...] = (0.0,) * DEX3_NUM_MOTORS
DEX3_RIGHT_OPEN: tuple[float, ...] = (0.0,) * DEX3_NUM_MOTORS
DEX3_LEFT_CLOSE: tuple[float, ...] = (0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8)
DEX3_RIGHT_CLOSE: tuple[float, ...] = (-0.8, -0.8, -0.8, -0.8, -0.8, -0.8, -0.8)


def _dex3_mode(motor_id: int, status: int = 0x01, timeout: int = 0) -> int:
    """Pack a Dex3-1 motor control mode byte (id | status | timeout)."""
    mode = 0
    mode |= motor_id & 0x0F
    mode |= (status & 0x07) << 4
    mode |= (timeout & 0x01) << 7
    return mode


def _smoothstep(alpha: float) -> float:
    """Ease-in/ease-out on [0, 1] so motions accelerate and decelerate gently."""
    alpha = min(1.0, max(0.0, alpha))
    return alpha * alpha * (3.0 - 2.0 * alpha)


# ChannelFactoryInitialize must run exactly once per process.
_CHANNEL_INITIALIZED = False


def _init_channel_factory(network_interface: str) -> None:
    global _CHANNEL_INITIALIZED
    if _CHANNEL_INITIALIZED:
        return
    if ChannelFactoryInitialize is None:
        raise UnitreeG1Error(
            "Could not import unitree-sdk2 Python package. Install loco "
            "dependencies with: `uv sync --extra loco`.\n"
            f"Original error: {_ARM_SDK_IMPORT_ERROR}"
        )
    ChannelFactoryInitialize(0, network_interface)
    _CHANNEL_INITIALIZED = True


def _require_arm_sdk() -> None:
    if _ARM_SDK_IMPORT_ERROR is not None or LowCmd_ is None:
        raise UnitreeG1Error(
            "Could not import unitree-sdk2 arm SDK types. Install loco "
            "dependencies with: `uv sync --extra loco`.\n"
            f"Original error: {_ARM_SDK_IMPORT_ERROR}"
        )


def _require_hand_sdk() -> None:
    if _HAND_SDK_IMPORT_ERROR is not None or HandCmd_ is None:
        raise UnitreeG1Error(
            "Could not import unitree-sdk2 Dex3-1 hand types. Install loco "
            "dependencies with `uv sync --extra loco`, or run with `--hand none` "
            "to command the arms only.\n"
            f"Original error: {_HAND_SDK_IMPORT_ERROR}"
        )


class G1ArmController:
    """Low-level arm control over the ``rt/arm_sdk`` DDS channel.

    Commands only the joints in ``joints`` (default: both arms). The weight
    blend joint is ramped 0->1 on :meth:`engage` and 1->0 on :meth:`release`,
    so the legs and waist stay under the balance controller the whole time.
    """

    def __init__(
        self,
        network_interface: str,
        joints: tuple[int, ...] = ARM_JOINTS,
        control_dt: float = 0.02,
        kp: float = 60.0,
        kd: float = 1.5,
        weight_ramp_s: float = 1.5,
        dry_run: bool = False,
    ) -> None:
        self.joints = tuple(joints)
        self.control_dt = control_dt
        self.kp = kp
        self.kd = kd
        self.weight_ramp_s = weight_ramp_s
        self.dry_run = dry_run

        self._state: object | None = None
        self._state_lock = threading.Lock()

        if not dry_run:
            _require_arm_sdk()
            self._crc = CRC()
            self._cmd = unitree_hg_msg_dds__LowCmd_()
            _init_channel_factory(network_interface)
            self._pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
            self._pub.Init()
            self._sub = ChannelSubscriber("rt/lowstate", LowState_)
            self._sub.Init(self._on_state, 10)
        else:
            self._pub = None
            self._sub = None

    # -- state --
    def _on_state(self, msg: object) -> None:
        with self._state_lock:
            self._state = msg

    def wait_for_state(self, timeout_s: float = 5.0) -> None:
        """Block until the first ``rt/lowstate`` message arrives."""
        if self.dry_run:
            return
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._state_lock:
                if self._state is not None:
                    return
            time.sleep(0.02)
        raise UnitreeG1Error(
            "No rt/lowstate received. Is the robot powered, on the same network, "
            "and is the interface correct? Try `uv run loco <ip> --diagnose`."
        )

    def current_arm_q(self) -> dict[int, float]:
        """Latest measured joint angles for the commanded joints."""
        if self.dry_run:
            return {joint: 0.0 for joint in self.joints}
        with self._state_lock:
            state = self._state
        if state is None:
            raise UnitreeG1Error("No lowstate yet; call wait_for_state() first.")
        return {joint: float(state.motor_state[joint].q) for joint in self.joints}

    # -- publishing --
    def _publish(self, q_by_joint: dict[int, float], weight: float) -> None:
        weight = min(1.0, max(0.0, weight))
        if self.dry_run:
            return
        for joint in self.joints:
            motor = self._cmd.motor_cmd[joint]
            motor.mode = 1  # enable
            motor.q = float(q_by_joint[joint])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = self.kp
            motor.kd = self.kd
        # Blend weight for arm_sdk lives in the spare joint's position field.
        self._cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = weight
        self._cmd.crc = self._crc.Crc(self._cmd)
        self._pub.Write(self._cmd)

    def _sleep_cycle(self) -> None:
        time.sleep(self.control_dt)

    def engage(self) -> dict[int, float]:
        """Ramp the arm_sdk weight 0->1 while holding the current pose."""
        start = self.current_arm_q()
        steps = max(1, int(self.weight_ramp_s / self.control_dt))
        for step in range(1, steps + 1):
            self._publish(start, weight=step / steps)
            self._sleep_cycle()
        return start

    def move_to(
        self,
        target: dict[int, float],
        duration_s: float,
        on_cycle=None,
    ) -> None:
        """Smoothly interpolate from the current pose to ``target``."""
        start = self.current_arm_q()
        steps = max(1, int(duration_s / self.control_dt))
        for step in range(1, steps + 1):
            alpha = _smoothstep(step / steps)
            frame = {
                joint: start[joint]
                + (target.get(joint, start[joint]) - start[joint]) * alpha
                for joint in self.joints
            }
            self._publish(frame, weight=1.0)
            if on_cycle is not None:
                on_cycle()
            self._sleep_cycle()

    def hold(
        self,
        target: dict[int, float],
        seconds: float | None,
        on_cycle=None,
        should_continue=None,
    ) -> None:
        """Keep commanding ``target`` so the grip stays firm and steady.

        ``seconds=None`` holds indefinitely until ``should_continue()`` returns
        False or the caller interrupts (Ctrl+C).
        """
        frame = {joint: target.get(joint, 0.0) for joint in self.joints}
        if seconds is None:
            while should_continue is None or should_continue():
                self._publish(frame, weight=1.0)
                if on_cycle is not None:
                    on_cycle()
                self._sleep_cycle()
            return
        steps = max(1, int(seconds / self.control_dt))
        for _ in range(steps):
            if should_continue is not None and not should_continue():
                break
            self._publish(frame, weight=1.0)
            if on_cycle is not None:
                on_cycle()
            self._sleep_cycle()

    def release(self, home: dict[int, float] | None, duration_s: float = 2.0) -> None:
        """Return toward ``home`` (if given) then ramp the weight 1->0."""
        if self.dry_run:
            return
        with self._state_lock:
            have_state = self._state is not None
        if not have_state:
            return
        target = home if home is not None else self.current_arm_q()
        if home is not None:
            self.move_to(home, duration_s)
        steps = max(1, int(self.weight_ramp_s / self.control_dt))
        for step in range(1, steps + 1):
            self._publish(target, weight=1.0 - step / steps)
            self._sleep_cycle()
        self._publish(target, weight=0.0)


class G1HandController:
    """Open/close control for the Dex3-1 hands (``rt/dex3/{left,right}/cmd``)."""

    def __init__(
        self,
        network_interface: str,
        side: str = "both",
        kp: float = 1.5,
        kd: float = 0.1,
        dry_run: bool = False,
    ) -> None:
        self.side = side
        self.kp = kp
        self.kd = kd
        self.dry_run = dry_run
        self._left_pub = None
        self._right_pub = None

        if not dry_run:
            _require_hand_sdk()
            _init_channel_factory(network_interface)
            if side in ("both", "left"):
                self._left_pub = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
                self._left_pub.Init()
            if side in ("both", "right"):
                self._right_pub = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
                self._right_pub.Init()

    def _send(self, publisher, q_vec: tuple[float, ...]) -> None:
        if self.dry_run or publisher is None:
            return
        cmd = unitree_hg_msg_dds__HandCmd_()
        for i in range(DEX3_NUM_MOTORS):
            motor = cmd.motor_cmd[i]
            motor.mode = _dex3_mode(i)
            motor.q = float(q_vec[i])
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = self.kp
            motor.kd = self.kd
        publisher.Write(cmd)

    def open(self) -> None:
        self._send(self._left_pub, DEX3_LEFT_OPEN)
        self._send(self._right_pub, DEX3_RIGHT_OPEN)

    def close(self) -> None:
        self._send(self._left_pub, DEX3_LEFT_CLOSE)
        self._send(self._right_pub, DEX3_RIGHT_CLOSE)


@dataclass
class GraspConfig:
    robot_ip: str
    network_interface: str | None = None
    side: str = "both"  # both | left | right
    use_hands: bool = True
    poses: GraspPoses = field(default_factory=GraspPoses)
    move_time_s: float = 3.0
    settle_s: float = 0.6
    grip_close_s: float = 1.0
    hold_seconds: float | None = 10.0  # None => hold until interrupted
    do_lift: bool = False
    return_home: bool = True
    kp: float = 60.0
    kd: float = 1.5
    dry_run: bool = False


def run_grasp(
    config: GraspConfig,
    on_event=None,
    should_continue=None,
) -> None:
    """Execute a full steady grasp: engage -> ready -> reach -> close -> hold.

    ``on_event(name, detail)`` is called at each phase for logging. The arms and
    hands are always released gracefully on exit (including Ctrl+C / errors).
    """

    def emit(name: str, **detail: object) -> None:
        if on_event is not None:
            on_event(name, detail)

    if config.dry_run:
        network_interface = config.network_interface or "dry-run"
    else:
        network_interface = config.network_interface or resolve_network_interface(
            config.robot_ip
        )
    joints = joints_for_side(config.side)

    arm = G1ArmController(
        network_interface,
        joints=joints,
        kp=config.kp,
        kd=config.kd,
        dry_run=config.dry_run,
    )
    hands: G1HandController | None = None
    if config.use_hands:
        hands = G1HandController(
            network_interface, side=config.side, dry_run=config.dry_run
        )

    poses = config.poses
    # Re-assert the closed grip every control cycle during reach/hold so the
    # fingers keep squeezing steadily.
    keep_closed = (lambda: hands.close()) if hands is not None else None

    emit("start", robot_ip=config.robot_ip, interface=network_interface, side=config.side)
    try:
        arm.wait_for_state()
        emit("engage")
        if hands is not None:
            hands.open()
        arm.engage()

        emit("move_ready", pose="ready")
        arm.move_to(poses.ready, config.move_time_s, on_cycle=None)
        if hands is not None:
            hands.open()
        arm.hold(poses.ready, config.settle_s)

        emit("move_reach", pose="reach")
        arm.move_to(poses.reach, config.move_time_s)
        arm.hold(poses.reach, config.settle_s)

        emit("close_hands")
        if hands is not None:
            hands.close()
        # Keep pressing the reach pose while the fingers close.
        arm.hold(poses.reach, config.grip_close_s, on_cycle=keep_closed)

        hold_pose = poses.reach
        if config.do_lift:
            emit("lift")
            arm.move_to(poses.lift, config.move_time_s, on_cycle=keep_closed)
            hold_pose = poses.lift

        emit("hold", seconds=config.hold_seconds)
        arm.hold(
            hold_pose,
            config.hold_seconds,
            on_cycle=keep_closed,
            should_continue=should_continue,
        )
        emit("hold_done")
    except KeyboardInterrupt:
        emit("interrupted")
    finally:
        emit("release")
        if hands is not None:
            # Open the hands to let go before the arms relax.
            hands.open()
        home = poses.home if config.return_home else None
        arm.release(home)
        emit("done")
