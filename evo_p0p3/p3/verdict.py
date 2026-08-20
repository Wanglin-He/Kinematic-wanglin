"""What one predicate returns, and the rules every predicate obeys.

Three outcomes, not two. ``N/A`` exists because the alternative is worse than useless: a
claim the tool cannot evaluate, scored as a pass, is indistinguishable from a claim it
evaluated and found satisfied. The previous project scored an unreadable field as an
absent one and charged the reader's gap to the asset.

Every result carries the number it was decided by and the threshold it was compared
against. A verdict without them cannot be argued with, and a verdict nobody can argue with
is not evidence -- it is an assertion. This is also the practical form of the reporting
rule the specification states as "do not hide failures in one scalar".
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NA = "na"
    """The claim could not be evaluated. Never counted as either satisfied or violated,
    and excluded from the denominator rather than folded into it."""


@dataclass(frozen=True, slots=True)
class ClaimResult:
    """One predicate applied to one subject."""

    predicate: str
    """Stable id, e.g. ``KF1.parent``."""

    subject: str
    """What it was applied to: a joint id, a part id, or a contact pair."""

    verdict: Verdict
    reason: str
    """One sentence a human can act on, naming the observed value."""

    measured: Mapping[str, object] = field(default_factory=dict)
    """The numbers the verdict came from."""

    threshold: Mapping[str, object] = field(default_factory=dict)
    """What they were compared against, by tolerance key."""

    evidence: Mapping[str, object] = field(default_factory=dict)
    """Anything else that helps: body names, the failing configuration, geom indices."""

    @property
    def applicable(self) -> bool:
        return self.verdict is not Verdict.NA

    @property
    def satisfied(self) -> bool:
        return self.verdict is Verdict.PASS

    def __str__(self) -> str:
        return f"[{self.verdict.value:4s}] {self.predicate} on {self.subject}: {self.reason}"


def score(results: tuple[ClaimResult, ...]) -> float | None:
    """Satisfied over applicable. ``None`` when nothing was applicable.

    ``None`` rather than 1.0, and rather than 0.0. A metric with an empty denominator has
    not measured a perfect asset; it has not measured anything, and reporting either
    extreme invents a result. The same reasoning removes N/A dimensions from the profile
    instead of averaging them in.
    """
    applicable = [r for r in results if r.applicable]
    if not applicable:
        return None
    return sum(1 for r in applicable if r.satisfied) / len(applicable)
