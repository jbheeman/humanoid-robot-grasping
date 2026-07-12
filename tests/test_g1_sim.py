from object_tracking.g1_sim_cli import Candidate, REVISION, _candidate, _wrapper_xml


def test_candidate_sampling_is_deterministic_and_bounded() -> None:
    assert _candidate(5, 7) == _candidate(5, 7)
    c = _candidate(5, 7)
    assert 30 <= c.kp <= 80
    assert 0.5 <= c.kd <= 4
    assert 0.05 <= c.vmax <= 0.25
    assert 0.25 <= c.amax <= 2


def test_wrapper_welds_pelvis_and_never_adds_robot_io() -> None:
    xml = _wrapper_xml(__import__("pathlib").Path("/tmp/g1.xml"))
    assert 'body1="pelvis"' in xml
    assert "LowCmd" not in xml
    assert len(REVISION) == 40


def test_candidate_schema() -> None:
    assert Candidate(60, 1.5, 0.1, 0.5).kp == 60
