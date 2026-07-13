#!/usr/bin/env bash
set -euo pipefail

# Run a stock, firmware-owned G1 arm action.  This is intentionally separate
# from the custom commissioning process: never run both command sources at
# once.  Defaults are conservative and target the robot's native DDS network.

ACTION="${1:-right_hand_up}"
HARDWARE_INTERFACE="${HARDWARE_INTERFACE:-eth0}"
HARDWARE_DOMAIN_ID="${HARDWARE_DOMAIN_ID:-0}"
HOLD_S="${G1_ACTION_HOLD_S:-3}"
UNITREE_SDK_PYTHONPATH="${UNITREE_SDK_PYTHONPATH:-${HOME}/unitree_sdk2_python}"

if [[ ! -f "${UNITREE_SDK_PYTHONPATH}/unitree_sdk2py/__init__.py" ]]; then
  echo "Native Unitree SDK2 was not found at ${UNITREE_SDK_PYTHONPATH}." >&2
  exit 1
fi
if [[ "${ACTION}" != "list" && "${ACTION}" != "release_arm" && "${ACTION}" != "right_hand_up" ]]; then
  echo "Usage: $0 [list|right_hand_up|release_arm]" >&2
  exit 2
fi

export PYTHONPATH="${UNITREE_SDK_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"

python3 - "${ACTION}" "${HARDWARE_INTERFACE}" "${HARDWARE_DOMAIN_ID}" "${HOLD_S}" <<'PY'
import sys
import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient

action_name, interface, domain_id, hold_s = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
ChannelFactoryInitialize(domain_id, interface)
client = G1ArmActionClient()
client.SetTimeout(10.0)
client.Init()
status, groups = client.GetActionList()
if status != 0:
    raise SystemExit(f"GetActionList failed: status={status}")
available = {item["name"]: item["id"] for group in groups for item in group if "id" in item}
if action_name == "list":
    for name, action_id in sorted(available.items(), key=lambda item: item[1]):
        print(f"{action_id:>3}  {name}")
    raise SystemExit(0)
if action_name not in available:
    raise SystemExit(f"Action {action_name!r} is unavailable; run with list to inspect this robot.")
print(f"Executing official Unitree action: {action_name} ({available[action_name]})")
result = client.ExecuteAction(available[action_name])
print(f"ExecuteAction result: {result}")
if action_name != "release_arm":
    time.sleep(hold_s)
    print("Releasing arms with the official release_arm action.")
    print(f"Release result: {client.ExecuteAction(available['release_arm'])}")
PY
