from object_tracking.g1_gpu_supervisor import next_candidate_count


def test_adaptive_batch_grows_when_vram_is_available() -> None:
    assert next_candidate_count(
        64, peak_mib=5000, target_mib=11264, aborted=False, minimum=8, maximum=1024
    ) == 115


def test_adaptive_batch_backs_off_after_pressure() -> None:
    assert next_candidate_count(
        100, peak_mib=11264, target_mib=11264, aborted=True, minimum=8, maximum=1024
    ) == 60


def test_adaptive_batch_respects_limits() -> None:
    assert next_candidate_count(
        1000, peak_mib=4000, target_mib=11264, aborted=False, minimum=8, maximum=1024
    ) == 1024
