from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PLUSHIE_TERMS = ("plushie", "stuffed animal", "stuffed toy", "soft toy")


@dataclass
class Intent:
    target_class: str
    action: str
    constraints: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_class": self.target_class,
            "action": self.action,
            "constraints": self.constraints,
        }


def parse_instruction(text: str) -> dict[str, Any]:
    normalized = " ".join(text.lower().strip().split())

    target_class = "plushie" if any(term in normalized for term in PLUSHIE_TERMS) else "unknown"
    action = "observe"
    constraints: dict[str, Any] = {
        "max_speed": "slow",
        "avoid_contact_with_people": True,
        "stop_before_collision": True,
    }

    if "ignore everything except" in normalized and target_class == "plushie":
        action = "filter_targets"
        constraints["allowed_classes"] = ["plushie"]
    elif "stop" in normalized and target_class == "plushie":
        action = "block_path"

    return Intent(
        target_class=target_class,
        action=action,
        constraints=constraints,
    ).as_dict()
