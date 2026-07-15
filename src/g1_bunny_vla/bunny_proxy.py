"""Physical specification for the first Isaac Sim bunny-plush proxy.

The real object has a soft exterior around a rigid, non-compressible frame, so
the first simulation asset intentionally uses rigid compound collision geometry.
It is a contact-and-grasp proxy, not a deformable-cloth model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BunnyProxySpec:
    """Measured dimensions and conservative tabletop-contact parameters in SI units."""

    height_m: float = 0.14
    depth_m: float = 0.15
    width_m: float = 0.07
    mass_kg: float = 0.15
    static_friction: float = 0.65
    dynamic_friction: float = 0.50
    restitution: float = 0.02
    min_push_speed_mps: float = 0.03
    max_push_speed_mps: float = 0.15

    def validate(self) -> None:
        values = (self.height_m, self.depth_m, self.width_m, self.mass_kg)
        if any(value <= 0 for value in values):
            raise ValueError("bunny dimensions and mass must be positive")
        if self.min_push_speed_mps < 0 or self.max_push_speed_mps < self.min_push_speed_mps:
            raise ValueError("push-speed bounds must be non-negative and ordered")


DEFAULT_BUNNY_PROXY = BunnyProxySpec()
