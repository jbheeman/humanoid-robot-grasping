import pytest

from scripts.training.apply_real_plush_review import apply_decisions


def test_review_promotes_good_and_rejects_false_contact() -> None:
    records = [
        {"episode_id": 85, "status": "needs_review", "reasons": ["visibility"]},
        {"episode_id": 134, "status": "needs_review", "reasons": ["no_contact"]},
    ]
    decisions = {
        85: {"status": "accepted", "reason": "visible moving plush reaches palm"},
        134: {
            "status": "automatic_reject",
            "reason": "plush absent at annotated contact",
        },
    }
    result = apply_decisions(records, decisions)
    assert result[0]["status"] == "accepted"
    assert result[0]["reasons"] == []
    assert result[1]["status"] == "automatic_reject"
    assert result[1]["human_review"]["reason"] == decisions[134]["reason"]


def test_review_requires_every_flagged_episode_to_be_resolved() -> None:
    with pytest.raises(ValueError, match="unresolved"):
        apply_decisions(
            [{"episode_id": 85, "status": "needs_review", "reasons": []}],
            {},
        )
