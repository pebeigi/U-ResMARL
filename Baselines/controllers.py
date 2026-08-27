"""Controller interface shared by all benchmarked models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from utility_model import TrafficAgent

if TYPE_CHECKING:  # pragma: no cover
    from Baselines.scenario import Scenario


@runtime_checkable
class Controller(Protocol):
    """Maps the joint state to one (accel, steering) command per agent."""

    name: str

    def reset(self, scenario: "Scenario") -> None:
        ...

    def compute_controls(
        self,
        agents: list[TrafficAgent],
        scenario: "Scenario",
        step: int,
    ) -> list[tuple[float, float]]:
        ...


class BaseController:
    """Convenience base with a no-op reset."""

    name = "base"

    def reset(self, scenario: "Scenario") -> None:  # noqa: D401
        return None

    def compute_controls(
        self,
        agents: list[TrafficAgent],
        scenario: "Scenario",
        step: int,
    ) -> list[tuple[float, float]]:
        raise NotImplementedError
