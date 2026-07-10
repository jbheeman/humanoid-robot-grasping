from __future__ import annotations

from dataclasses import dataclass

from object_tracking.arm_tracking.arm_bridge import ArmCommand
from object_tracking.arm_tracking.arm_unitree import (
    ARM_INDICES,
    ARM_WEIGHT_INDEX,
    UnitreeArmHardware,
)


@dataclass
class Motor:
    mode: int = -1
    q: float = -999.0
    dq: float = -999.0
    kp: float = -999.0
    kd: float = -999.0
    tau: float = -999.0


class LowCommand:
    def __init__(self) -> None:
        self.mode_pr = -1
        self.mode_machine = -1
        self.motor_cmd = [Motor() for _ in range(30)]
        self.crc = -1


class CRC:
    def Crc(self, command: LowCommand) -> int:
        return 12345


class Publisher:
    def __init__(self) -> None:
        self.writes: list[LowCommand] = []

    def Write(self, command: LowCommand) -> None:
        self.writes.append(command)


def test_sdk_adapter_writes_only_all_fourteen_arm_slots_and_weight() -> None:
    hardware = UnitreeArmHardware()
    low_command = LowCommand()
    publisher = Publisher()
    hardware._started = True
    hardware._low_cmd = low_command
    hardware._crc = CRC()
    hardware._publisher = publisher
    command = ArmCommand(
        q=tuple(float(index) / 10.0 for index in range(14)),
        dq=(0.0,) * 14,
        kp=(60.0,) * 14,
        kd=(1.5,) * 14,
        weight=0.75,
        mode_machine=5,
        published_at=10.0,
    )

    hardware.publish(command)

    assert publisher.writes == [low_command]
    assert low_command.mode_pr == 0
    assert low_command.mode_machine == 5
    assert low_command.crc == 12345
    for offset, joint_index in enumerate(ARM_INDICES):
        motor = low_command.motor_cmd[joint_index]
        assert motor.q == command.q[offset]
        assert motor.dq == 0.0
        assert motor.kp == 60.0
        assert motor.kd == 1.5
        assert motor.tau == 0.0
    assert low_command.motor_cmd[ARM_WEIGHT_INDEX].q == 0.75
    # Waist slots must not be commanded by this bridge.
    assert [low_command.motor_cmd[index].q for index in (12, 13, 14)] == [-999.0, -999.0, -999.0]
