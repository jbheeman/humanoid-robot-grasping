#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import signal
import sys
import threading
import time

G1_NUM_MOTOR = 29

LeftShoulderPitch = 15
LeftShoulderRoll = 16
LeftShoulderYaw = 17
LeftElbow = 18
LeftWristRoll = 19

RightShoulderPitch = 22
RightShoulderRoll = 23
RightShoulderYaw = 24
RightElbow = 25
RightWristRoll = 26

ARM_JOINTS = (
    LeftShoulderPitch,
    LeftShoulderRoll,
    LeftShoulderYaw,
    LeftElbow,
    LeftWristRoll,
    RightShoulderPitch,
    RightShoulderRoll,
    RightShoulderYaw,
    RightElbow,
    RightWristRoll,
)

CONTROL_DT = 0.002
RETURN_SECONDS = 2.0


class LowStateBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()
        self.lowstate = None

    def callback(self, msg: object) -> None:
        with self._lock:
            self.lowstate = msg
            self._event.set()

    def wait(self, timeout_s: float) -> object:
        if not self._event.wait(timeout_s):
            raise TimeoutError("Timed out waiting for rt/lowstate.")
        with self._lock:
            if self.lowstate is None:
                raise TimeoutError("LowState event fired but no lowstate was stored.")
            return self.lowstate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Robot-local low-level G1 arm test: move shoulder pitch only while holding body posture."
    )
    parser.add_argument("--interface", default="wlan0")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--ramp-seconds", type=float, default=3.0)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--delta", type=float, default=0.25)
    parser.add_argument("--sign", type=int, default=1, choices=(-1, 1))
    parser.add_argument(
        "--direction",
        choices=("positive", "negative"),
        default=None,
        help="Alternative to --sign. positive maps to +1; negative maps to -1.",
    )
    parser.add_argument("--left-sign", type=int, choices=(-1, 1), default=None)
    parser.add_argument("--right-sign", type=int, choices=(-1, 1), default=None)
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--kp-arm", type=float, default=25.0)
    parser.add_argument("--kd-arm", type=float, default=1.0)
    parser.add_argument("--kp-body", type=float, default=40.0)
    parser.add_argument("--kd-body", type=float, default=1.0)
    parser.add_argument("--smoke", action="store_true", help="Print the planned shoulder pitch offsets and exit before DDS.")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive_values = {
        "ramp_seconds": args.ramp_seconds,
        "delta": args.delta,
    }
    nonnegative_values = {
        "hold_seconds": args.hold_seconds,
        "kp_arm": args.kp_arm,
        "kd_arm": args.kd_arm,
        "kp_body": args.kp_body,
        "kd_body": args.kd_body,
    }
    for name, value in positive_values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and > 0")
    for name, value in nonnegative_values.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")


def status_tuple(value: object) -> str:
    return repr(value)


def import_unitree_sdk() -> dict[str, object]:
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC

    return {
        "MotionSwitcherClient": MotionSwitcherClient,
        "ChannelFactoryInitialize": ChannelFactoryInitialize,
        "ChannelPublisher": ChannelPublisher,
        "ChannelSubscriber": ChannelSubscriber,
        "unitree_hg_msg_dds__LowCmd_": unitree_hg_msg_dds__LowCmd_,
        "unitree_hg_msg_dds__LowState_": unitree_hg_msg_dds__LowState_,
        "LowCmd_": LowCmd_,
        "LowState_": LowState_,
        "CRC": CRC,
    }


def release_motion_mode(MotionSwitcherClient: object, timeout_s: float = 10.0) -> None:
    print("constructing MotionSwitcherClient")
    client = MotionSwitcherClient()
    client.SetTimeout(timeout_s)
    init_ret = client.Init()
    print(f"MotionSwitcherClient.Init(): {status_tuple(init_ret)}")

    check_ret = client.CheckMode()
    print(f"MotionSwitcherClient.CheckMode(): {status_tuple(check_ret)}")

    mode_name = ""
    if isinstance(check_ret, tuple) and len(check_ret) >= 2 and isinstance(check_ret[1], dict):
        mode_name = str(check_ret[1].get("name") or "")

    if mode_name:
        release_ret = client.ReleaseMode()
        print(f"MotionSwitcherClient.ReleaseMode(): {status_tuple(release_ret)}")
    else:
        print("MotionSwitcherClient.ReleaseMode(): skipped because mode name is empty")


def current_q_from_lowstate(lowstate: object) -> list[float]:
    q: list[float] = []
    for index in range(G1_NUM_MOTOR):
        q.append(float(lowstate.motor_state[index].q))
    return q


def selected_shoulder_pitch_joints(side: str) -> list[int]:
    if side == "left":
        return [LeftShoulderPitch]
    if side == "right":
        return [RightShoulderPitch]
    return [LeftShoulderPitch, RightShoulderPitch]


def base_sign(args: argparse.Namespace) -> int:
    if args.direction == "positive":
        return 1
    if args.direction == "negative":
        return -1
    return int(args.sign)


def shoulder_pitch_signs(args: argparse.Namespace) -> dict[int, int]:
    sign = base_sign(args)
    return {
        LeftShoulderPitch: args.left_sign if args.left_sign is not None else sign,
        RightShoulderPitch: args.right_sign if args.right_sign is not None else sign,
    }


def lerp(start: list[float], target: list[float], alpha: float) -> list[float]:
    alpha = min(max(alpha, 0.0), 1.0)
    return [(1.0 - alpha) * a + alpha * b for a, b in zip(start, target, strict=True)]


def fill_lowcmd(
    low_cmd: object,
    q_target: list[float],
    mode_machine: int,
    kp_body: float,
    kd_body: float,
    kp_arm: float,
    kd_arm: float,
    crc: object,
) -> None:
    low_cmd.mode_pr = 0
    low_cmd.mode_machine = mode_machine

    for index in range(G1_NUM_MOTOR):
        motor = low_cmd.motor_cmd[index]
        motor.mode = 1
        motor.q = q_target[index]
        motor.dq = 0.0
        motor.tau = 0.0
        if index in ARM_JOINTS:
            motor.kp = kp_arm
            motor.kd = kd_arm
        else:
            motor.kp = kp_body
            motor.kd = kd_body

    low_cmd.crc = crc.Crc(low_cmd)


def publish_for_duration(
    publisher: ChannelPublisher,
    low_cmd: object,
    start_q: list[float],
    target_q: list[float],
    seconds: float,
    mode_machine: int,
    kp_body: float,
    kd_body: float,
    kp_arm: float,
    kd_arm: float,
    crc: CRC,
    stop_event: threading.Event,
) -> None:
    if seconds <= 0.0:
        return
    started_at = time.monotonic()
    while not stop_event.is_set():
        elapsed = time.monotonic() - started_at
        if elapsed >= seconds:
            break
        alpha = elapsed / seconds
        q = lerp(start_q, target_q, alpha)
        fill_lowcmd(low_cmd, q, mode_machine, kp_body, kd_body, kp_arm, kd_arm, crc)
        publisher.Write(low_cmd)
        time.sleep(CONTROL_DT)


def hold_target(
    publisher: ChannelPublisher,
    low_cmd: object,
    target_q: list[float],
    seconds: float,
    mode_machine: int,
    kp_body: float,
    kd_body: float,
    kp_arm: float,
    kd_arm: float,
    crc: CRC,
    stop_event: threading.Event,
) -> None:
    if seconds <= 0.0:
        return
    deadline = time.monotonic() + seconds
    while not stop_event.is_set() and time.monotonic() < deadline:
        fill_lowcmd(low_cmd, target_q, mode_machine, kp_body, kd_body, kp_arm, kd_arm, crc)
        publisher.Write(low_cmd)
        time.sleep(CONTROL_DT)


def main() -> int:
    args = build_parser().parse_args()
    validate_args(args)
    signs = shoulder_pitch_signs(args)

    selected_joints = selected_shoulder_pitch_joints(args.side)
    if args.smoke:
        print("smoke: no DDS initialization and no movement")
        print(f"interface={args.interface} domain_id={args.domain_id}")
        print(f"side={args.side} delta={args.delta}")
        for joint in selected_joints:
            label = "left" if joint == LeftShoulderPitch else "right"
            offset = signs[joint] * args.delta
            print(f"{label} shoulder pitch index={joint} offset={offset:+.6f}")
        return 0

    print(
        "Robot must be physically supported or clear of people. "
        "Low-level mode can destabilize standing balance.",
        file=sys.stderr,
    )

    stop_event = threading.Event()

    def handle_signal(signum: int, frame: object) -> None:
        del signum, frame
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"initializing DDS domain={args.domain_id} interface={args.interface}")
    sdk = import_unitree_sdk()
    ChannelFactoryInitialize = sdk["ChannelFactoryInitialize"]
    ChannelPublisher = sdk["ChannelPublisher"]
    ChannelSubscriber = sdk["ChannelSubscriber"]
    MotionSwitcherClient = sdk["MotionSwitcherClient"]
    LowCmd_ = sdk["LowCmd_"]
    LowState_ = sdk["LowState_"]
    unitree_hg_msg_dds__LowCmd_ = sdk["unitree_hg_msg_dds__LowCmd_"]
    unitree_hg_msg_dds__LowState_ = sdk["unitree_hg_msg_dds__LowState_"]
    CRC = sdk["CRC"]

    ChannelFactoryInitialize(args.domain_id, args.interface)

    release_motion_mode(MotionSwitcherClient)

    state_buffer = LowStateBuffer()
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(state_buffer.callback, 10)

    print("waiting for rt/lowstate")
    lowstate = state_buffer.wait(10.0)
    print("received lowstate")

    mode_machine = int(getattr(lowstate, "mode_machine", 0))
    start_q = current_q_from_lowstate(lowstate)
    target_q = start_q.copy()
    for joint in selected_joints:
        target_q[joint] = start_q[joint] + signs[joint] * args.delta

    print(
        "starting q for shoulders: "
        f"left={start_q[LeftShoulderPitch]:.6f}, right={start_q[RightShoulderPitch]:.6f}"
    )
    print(
        "target q for shoulders: "
        f"left={target_q[LeftShoulderPitch]:.6f}, right={target_q[RightShoulderPitch]:.6f}"
    )

    publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    publisher.Init()
    low_cmd = unitree_hg_msg_dds__LowCmd_()
    _unused_low_state_default = unitree_hg_msg_dds__LowState_()
    del _unused_low_state_default
    crc = CRC()

    try:
        print("ramp start")
        publish_for_duration(
            publisher,
            low_cmd,
            start_q,
            target_q,
            args.ramp_seconds,
            mode_machine,
            args.kp_body,
            args.kd_body,
            args.kp_arm,
            args.kd_arm,
            crc,
            stop_event,
        )

        if not stop_event.is_set():
            print("hold start")
            hold_target(
                publisher,
                low_cmd,
                target_q,
                args.hold_seconds,
                mode_machine,
                args.kp_body,
                args.kd_body,
                args.kp_arm,
                args.kd_arm,
                crc,
                stop_event,
            )
    finally:
        print("return-to-neutral start")
        publish_for_duration(
            publisher,
            low_cmd,
            target_q,
            start_q,
            RETURN_SECONDS,
            mode_machine,
            args.kp_body,
            args.kd_body,
            args.kp_arm,
            args.kd_arm,
            crc,
            threading.Event(),
        )

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
