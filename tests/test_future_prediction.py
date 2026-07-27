import numpy as np

from object_tracking.future_prediction import TrackObservation, evaluate_future_positions


def test_constant_velocity_future_prediction_is_accurate_after_warmup() -> None:
    observations = [
        TrackObservation(index * 0.1, 1, np.array([index * 10.0, 20.0]), 1.0, 0.9)
        for index in range(12)
    ]
    report = evaluate_future_positions(observations, horizons_s=(0.2,))
    metrics = report["horizons"]["200ms"]
    assert metrics["samples"] > 5
    assert metrics["pixel_median"] < 2.0
    assert metrics["three_d_median"] < 2.0


def test_low_confidence_observations_are_rejected() -> None:
    observations = [
        TrackObservation(0.0, 1, np.array([0.0, 0.0]), None, 0.1),
        TrackObservation(0.2, 1, np.array([2.0, 0.0]), None, 0.1),
    ]
    report = evaluate_future_positions(observations, horizons_s=(0.1,))
    assert report["horizons"]["100ms"]["samples"] == 0
