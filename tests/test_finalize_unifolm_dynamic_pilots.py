from pathlib import Path


def test_finalizer_is_event_driven_and_never_authorizes_robot() -> None:
    script = (
        Path(__file__).parents[1]
        / "scripts/training/finalize_unifolm_dynamic_pilots.sh"
    ).read_text()

    assert "wait_for_completion.py" in script
    assert "sleep " not in script
    assert "physical_robot_authorized" in script
    assert "evaluate_unifolm_dynamic_pilots.py" in script
