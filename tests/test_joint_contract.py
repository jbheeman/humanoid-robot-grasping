from object_tracking.arm_tracking.joints import (
    RIGHT_ARM_INDICES,
    RIGHT_ARM_JOINT_NAMES,
    audit_urdf,
    joint_contract,
)


def test_joint_contract_matches_unitree_29_dof_slots() -> None:
    contract = joint_contract()

    assert [item["position"] for item in contract] == list(range(7))
    assert [item["name"] for item in contract] == list(RIGHT_ARM_JOINT_NAMES)
    assert [item["sdk_index"] for item in contract] == list(RIGHT_ARM_INDICES)


def test_urdf_audit_detects_order_and_limit_contract(tmp_path) -> None:
    contract = joint_contract()
    joints = "\n".join(
        f'''<joint name="{item["name"]}" type="revolute">
          <parent link="p{item["position"]}"/><child link="c{item["position"]}"/>
          <limit lower="{item["lower_rad"]}" upper="{item["upper_rad"]}"/>
        </joint>'''
        for item in contract
    )
    path = tmp_path / "g1.urdf"
    path.write_text(f'<robot name="g1">{joints}</robot>', encoding="utf-8")

    assert audit_urdf(path)["ok"] is True

    reversed_joints = "\n".join(
        f'''<joint name="{item["name"]}" type="revolute">
          <parent link="rp{item["position"]}"/><child link="rc{item["position"]}"/>
          <limit lower="{item["lower_rad"]}" upper="{item["upper_rad"]}"/>
        </joint>'''
        for item in reversed(contract)
    )
    reversed_path = tmp_path / "reversed.urdf"
    reversed_path.write_text(f'<robot name="g1">{reversed_joints}</robot>', encoding="utf-8")
    assert audit_urdf(reversed_path)["ok"] is False
    assert audit_urdf(tmp_path / "missing.urdf")["ok"] is False
