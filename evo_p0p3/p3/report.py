"""The Kinematic Profile, and the evidence that makes it arguable.

The specification is explicit that a single scalar must not hide failures, and the profile
is a triple for that reason -- but a triple hides them too if that is all anyone prints.
So the report carries, for every claim: the verdict, the number it was decided by, the
threshold it was compared against, and enough evidence to reproduce it. A verdict nobody
can argue with is an assertion, not a finding.

Two things beyond the specification are carried, both because their absence would let a
reader mistake our patches for the asset's properties:

**Provenance.** The loader synthesises inertia where the source declares none and recovers
couplings from ``<mimic>`` that MuJoCo's importer discards. Neither influences a score, and
both are stated in every report so nobody has to take that on trust.

**Coverage.** The sweep's size, its per-layer breakdown, and which joint pairs the
adjacency gate pruned. A schedule that silently skipped half the configuration space would
otherwise read exactly like one that covered it.

Aggregation excludes N/A rather than folding it in. An unmeasured dimension is not a
perfect one, and a metric whose applicable set is empty reports ``None`` -- not 1.0, and
not 0.0, either of which would invent a result.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from evo_p0p3.p0.schema import Contract
from evo_p0p3.p3.gate import Admission
from evo_p0p3.p3.kf1 import DECLARATION_READS
from evo_p0p3.p3.sweep import Schedule
from evo_p0p3.p3.verdict import ClaimResult, Verdict, score


class MalformedReport(ValueError):
    """A report that would mislead. Raised rather than written."""


@dataclass(frozen=True, slots=True)
class AssetReport:
    """Everything one asset's evaluation produced."""

    record_id: str
    contract_id: str
    admission: Admission
    results: tuple[ClaimResult, ...] = ()
    schedule: Schedule | None = None
    provenance: dict = field(default_factory=dict)
    tolerance_digest: str = ""

    # -- the profile -------------------------------------------------------------------

    def metric(self, prefix: str) -> float | None:
        return score(tuple(r for r in self.results if r.predicate.startswith(prefix)))

    @property
    def profile(self) -> dict[str, float | None]:
        return {name: self.metric(name) for name in ("KF1", "KF2", "KF3")}

    @property
    def sub_scores(self) -> dict[str, float | None]:
        """KF1 split by how each claim was decided.

        Joint type and declared range parsed at 100% completeness in the previous corpus
        and matched in nearly every asset, so a flat proportion is dominated by claims that
        cannot vary. Splitting them out keeps the measured ones visible instead of letting
        them be outvoted.
        """
        declared = tuple(r for r in self.results if r.predicate in DECLARATION_READS)
        measured = tuple(
            r for r in self.results
            if r.predicate.startswith("KF1") and r.predicate not in DECLARATION_READS
        )
        return {"KF1.declared": score(declared), "KF1.measured": score(measured)}

    # -- what went wrong ---------------------------------------------------------------

    @property
    def failures(self) -> tuple[ClaimResult, ...]:
        return tuple(r for r in self.results if r.verdict is Verdict.FAIL)

    @property
    def abstentions(self) -> tuple[ClaimResult, ...]:
        return tuple(r for r in self.results if r.verdict is Verdict.NA)

    def applicability(self) -> dict[str, str]:
        """How much of each metric was actually evaluated, as `applicable/total`."""
        out = {}
        for name in ("KF1", "KF2", "KF3"):
            group = [r for r in self.results if r.predicate.startswith(name)]
            out[name] = f"{sum(1 for r in group if r.applicable)}/{len(group)}"
        return out

    # -- serialisation -----------------------------------------------------------------

    def to_dict(self) -> dict:
        validate(self)
        return {
            "record_id": self.record_id,
            "contract_id": self.contract_id,
            "admitted": self.admission.admitted,
            "gate": [
                {"check": c.name, "status": c.status.value, "detail": c.detail}
                for c in self.admission.checks
            ],
            "profile": self.profile,
            "sub_scores": self.sub_scores,
            "applicability": self.applicability(),
            "tolerance_digest": self.tolerance_digest,
            "provenance": self.provenance,
            "coverage": self.schedule.provenance if self.schedule else None,
            "claims": [
                {
                    "predicate": r.predicate,
                    "subject": r.subject,
                    "verdict": r.verdict.value,
                    "reason": r.reason,
                    "measured": dict(r.measured),
                    "threshold": dict(r.threshold),
                    "evidence": dict(r.evidence),
                }
                for r in self.results
            ],
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=False),
            encoding="utf-8",
        )
        return path

    # -- for a person ------------------------------------------------------------------

    def render(self, *, verbose: bool = False) -> str:
        lines = [f"{self.record_id}  ({self.contract_id})"]
        lines.append(f"  gate: {self.admission.summary()}")
        for check in self.admission.checks:
            if not check.passed:
                lines.append(f"    {check}")
        if not self.admission.admitted:
            return "\n".join(lines)

        shown = {k: ("  n/a" if v is None else f"{v:5.2f}") for k, v in self.profile.items()}
        lines.append(
            "  profile: " + "  ".join(f"{k}={v}" for k, v in shown.items())
            + "   applicable: "
            + "  ".join(f"{k}={v}" for k, v in self.applicability().items())
        )
        sub = self.sub_scores
        lines.append(
            "  KF1 split: "
            + "  ".join(
                f"{k.split('.')[1]}=" + ("n/a" if v is None else f"{v:.2f}")
                for k, v in sub.items()
            )
        )
        if self.schedule:
            lines.append(
                f"  swept {self.schedule.size} configurations "
                f"({', '.join(f'{k} {v}' for k, v in self.schedule.by_layer().items())})"
            )
        for note in self.provenance.get("notes", []):
            lines.append(f"  note: {note}")

        if self.failures:
            lines.append(f"  {len(self.failures)} failure(s):")
            for r in self.failures:
                lines.append(f"    ✗ {r.predicate} on {r.subject}")
                lines.append(f"        {r.reason}")
                if r.measured:
                    lines.append(f"        measured {dict(r.measured)}")
        else:
            lines.append("  no failures")
        if verbose and self.abstentions:
            lines.append(f"  {len(self.abstentions)} abstention(s):")
            for r in self.abstentions:
                lines.append(f"    - {r.predicate} on {r.subject}: {r.reason}")
        return "\n".join(lines)


def validate(report: AssetReport) -> None:
    """Refuse to emit a report that would mislead.

    Three shapes are rejected, each of which the previous project produced at some point:
    a failure with no stated reason, a verdict with nothing measured behind it, and an
    abstention that does not say what could not be read. The last is the one that matters
    most -- an unexplained N/A is indistinguishable from a pass to anyone reading a table.
    """
    problems = []
    for r in report.results:
        if r.verdict is Verdict.FAIL and not r.reason.strip():
            problems.append(f"{r.predicate} on {r.subject}: failure with no reason")
        if r.verdict is not Verdict.NA and not r.measured:
            problems.append(f"{r.predicate} on {r.subject}: verdict with nothing measured")
        if r.verdict is Verdict.NA and not r.reason.strip():
            problems.append(f"{r.predicate} on {r.subject}: abstention with no explanation")
    if report.admission.admitted and not report.results:
        problems.append("admitted asset produced no claims at all")
    if problems:
        raise MalformedReport("; ".join(problems))


# --------------------------------------------------------------------------------------
# across a run
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunReport:
    """A whole evaluation. Gate pass rate first, because the profile is conditional on it."""

    reports: tuple[AssetReport, ...]

    @property
    def gate_pass_rate(self) -> float | None:
        if not self.reports:
            return None
        return sum(1 for r in self.reports if r.admission.admitted) / len(self.reports)

    @property
    def admitted(self) -> tuple[AssetReport, ...]:
        return tuple(r for r in self.reports if r.admission.admitted)

    def mean_profile(self) -> dict[str, float | None]:
        """Averaged over the assets where each metric was applicable.

        Conditional on admission, and reported next to the pass rate for that reason: a
        generator whose assets mostly fail the Gate would otherwise show a flattering
        profile computed over the few that survived.
        """
        out: dict[str, float | None] = {}
        for name in ("KF1", "KF2", "KF3"):
            values = [
                r.metric(name) for r in self.admitted if r.metric(name) is not None
            ]
            out[name] = sum(values) / len(values) if values else None
        return out

    def applicable_counts(self) -> dict[str, str]:
        out = {}
        for name in ("KF1", "KF2", "KF3"):
            n = sum(1 for r in self.admitted if r.metric(name) is not None)
            out[name] = f"{n}/{len(self.admitted)}"
        return out

    def render(self) -> str:
        rate = self.gate_pass_rate
        lines = [
            f"Gate pass rate: "
            + ("n/a" if rate is None else f"{rate:.0%} ({len(self.admitted)}/{len(self.reports)})")
        ]
        profile = self.mean_profile()
        lines.append(
            "Kinematic Profile (mean over applicable, admitted assets): "
            + "  ".join(
                f"{k}=" + ("n/a" if v is None else f"{v:.3f}") for k, v in profile.items()
            )
        )
        lines.append(
            "Applicable assets per metric: "
            + "  ".join(f"{k}={v}" for k, v in self.applicable_counts().items())
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "gate_pass_rate": self.gate_pass_rate,
            "assets": len(self.reports),
            "admitted": len(self.admitted),
            "mean_profile": self.mean_profile(),
            "applicable_counts": self.applicable_counts(),
            "reports": [r.to_dict() for r in self.reports],
        }


def build(
    contract: Contract,
    admission: Admission,
    results: Iterable[ClaimResult] = (),
    schedule: Schedule | None = None,
) -> AssetReport:
    provenance = dict(admission.asset.provenance) if admission.asset else {}
    return AssetReport(
        record_id=admission.record_id,
        contract_id=contract.record_id,
        admission=admission,
        results=tuple(results),
        schedule=schedule,
        provenance=provenance,
        tolerance_digest=contract.kinematic_claims.tolerances.digest(),
    )
