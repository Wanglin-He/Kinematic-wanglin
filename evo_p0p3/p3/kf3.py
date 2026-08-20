"""KF3 -- feasible motion consistency.

Whether the asset can actually move the way P0 says, and what it runs into on the way.

Scored **per pair**, not per sample. One deep interpenetration at one configuration out of
a thousand is one broken pair, and dividing by the sample count would report it as
0.999 -- the same defect moving by three orders of magnitude depending on how finely the
sweep happened to be drawn. The configuration is evidence, not a denominator.

What the specification calls motion validity is mostly not implementable as written, and
the reason is worth stating rather than quietly working around. "Closed, open and
intermediate states are reachable", "the path is continuous", "relative poses follow the
declared semantics" -- on a compiled mjModel these are all properties of forward
kinematics, which is exact and continuous by construction. No asset can fail them. What
*can* fail is whether the asset runs into itself getting there, so reachability is scored
as "this state is reachable **without forbidden interference**", which is a question about
the asset rather than about MuJoCo.

Distances come from ``mj_geomDistance``, never from contacts: every Articraft geom
compiles with ``contype = conaffinity = 0``, so ``mjData.ncon`` is always zero and
contact-based detection returns nothing at all. The distance query also ignores MuJoCo's
parent-child filter, which matters more than it sounds -- that filter hides exactly the
pair a swept-interference check most needs to see, a door against its own frame.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from evo_p0p3.p0.schema import ContactClaim, Contract
from evo_p0p3.p3.binding import Binding, BindingError
from evo_p0p3.p3.mjcf import body_pair_distance
from evo_p0p3.p3.sweep import Sample, Schedule, Sweeper
from evo_p0p3.p3.verdict import ClaimResult, Verdict

DISTMAX = 0.5
"""Cutoff for the distance query. Anything further apart reads as ``DISTMAX``, so a
reported minimum clearance at this value means "at least this far" rather than a
measurement -- said explicitly in the evidence rather than left for a reader to assume."""


def _na(predicate: str, subject: str, reason: str, **evidence) -> ClaimResult:
    return ClaimResult(
        predicate=predicate, subject=subject, verdict=Verdict.NA, reason=reason,
        evidence=evidence,
    )


def _pair_label(claim: ContactClaim) -> str:
    return f"{claim.parts[0]}+{claim.parts[1]}@{claim.state}"


class Session:
    """One asset, one contract, one walk of the sweep.

    The sweep is walked once and every declared pair measured at every configuration,
    because walking it per predicate would multiply the cost by the number of predicates
    and, worse, let two predicates disagree about what they saw.
    """

    def __init__(self, contract: Contract, binding: Binding) -> None:
        self.contract = contract
        self.binding = binding
        self.asset = binding.asset
        self.sweeper = Sweeper(contract, binding)
        self.schedule: Schedule = self.sweeper.schedule()
        self._joint_of_part = {
            j.part: j for j in contract.kinematic_claims.joints
        }
        self._track: dict[str, dict] = {}
        self._walk()

    # -- state classification ----------------------------------------------------------

    def _joints_in_scope(self, claim: ContactClaim) -> list:
        """Which joints a named state constrains for this claim.

        ``named_parts`` -- the default -- takes only the joints of the parts the claim
        names. "The drawer is shut" should not require the other two drawers to be shut as
        well, and requiring it would leave the claim evaluated at almost no configuration.
        """
        states = self.contract.kinematic_claims.asset_states
        definition = states.get(claim.state)
        if definition is None:
            return []
        if definition.scope.value == "all_declaring":
            return [
                j for j in self.contract.kinematic_claims.joints
                if claim.state in (j.states or {})
            ]
        return [
            self._joint_of_part[p]
            for p in claim.parts
            if p in self._joint_of_part
            and claim.state in (self._joint_of_part[p].states or {})
        ]

    def _in_state(self, sample: Sample, claim: ContactClaim) -> bool:
        if claim.state == "all":
            return True
        definition = self.contract.kinematic_claims.asset_states.get(claim.state)
        if definition is None:
            return False
        joints = self._joints_in_scope(claim)
        if not joints:
            return False
        model = self.asset.model
        for joint in joints:
            k = self.sweeper._joints.get(joint.id)
            if k is None:
                return False
            target = joint.states[claim.state]
            actual = float(sample.qpos[int(model.jnt_qposadr[k])])
            if abs(actual - target) > definition.tolerance_relative * (joint.range.span or 1.0):
                return False
        return True

    # -- the walk ----------------------------------------------------------------------

    def _all_claims(self) -> list[tuple[str, ContactClaim]]:
        policy = self.contract.kinematic_claims.contact_policy
        return (
            [("required", c) for c in policy.required]
            + [("permitted", c) for c in policy.permitted]
            + [("forbidden", c) for c in policy.forbidden]
        )

    def _exempt(self, claim: ContactClaim, sample: Sample) -> bool:
        """Whether a required or permitted claim covers this forbidden pair here.

        Without this the canonical asset is undefined at exactly the configurations the
        sweep begins and ends in: a drawer resting against its stop is required to touch at
        the closed state and forbidden to touch at all states.
        """
        policy = self.contract.kinematic_claims.contact_policy
        pair = frozenset(claim.parts)
        for other in list(policy.required) + list(policy.permitted):
            if frozenset(other.parts) == pair and self._in_state(sample, other):
                return True
        return False

    def _walk(self) -> None:
        for bucket, claim in self._all_claims():
            key = f"{bucket}:{_pair_label(claim)}"
            self._track[key] = {
                "bucket": bucket, "claim": claim,
                "min_clearance": np.inf, "min_at": None,
                "max_penetration": 0.0, "max_at": None,
                "first_failing": None, "samples_in_state": 0,
                "unresolved": None,
            }

        for sample in self.schedule.samples:
            self.sweeper.visit(sample)
            for bucket, claim in self._all_claims():
                key = f"{bucket}:{_pair_label(claim)}"
                record = self._track[key]
                if record["unresolved"]:
                    continue
                if not self._in_state(sample, claim):
                    continue
                try:
                    a = self.binding.root_body(claim.parts[0])
                    b = self.binding.root_body(claim.parts[1])
                except BindingError as exc:
                    record["unresolved"] = str(exc)
                    continue
                record["samples_in_state"] += 1
                distance, _ = body_pair_distance(self.asset, a, b, DISTMAX)
                if distance < record["min_clearance"]:
                    record["min_clearance"] = distance
                    record["min_at"] = sample.label
                if bucket == "forbidden" and not self._exempt(claim, sample):
                    penetration = max(0.0, -distance)
                    if penetration > record["max_penetration"]:
                        record["max_penetration"] = penetration
                        record["max_at"] = sample.label
                    tol = (
                        claim.tolerance_override
                        or self.contract.kinematic_claims.tolerances.forbidden_penetration_m
                    )
                    if penetration > tol and record["first_failing"] is None:
                        record["first_failing"] = sample.label

    # -- predicates --------------------------------------------------------------------

    def forbidden_pairs(self) -> tuple[ClaimResult, ...]:
        """KF3.forbidden_pair -- a pair declared never to interfere never does.

        One verdict per pair over the whole swept set. The configuration where it first
        went wrong is reported as evidence, which is what a person needs to reproduce it,
        but it is not what the score is divided by.
        """
        results = []
        tolerances = self.contract.kinematic_claims.tolerances
        for key, record in self._track.items():
            if record["bucket"] != "forbidden":
                continue
            claim = record["claim"]
            subject = _pair_label(claim)
            if record["unresolved"]:
                results.append(_na("KF3.forbidden_pair", subject, record["unresolved"]))
                continue
            if record["samples_in_state"] == 0:
                results.append(_na(
                    "KF3.forbidden_pair", subject,
                    f"no swept configuration matched state {claim.state!r}, so the pair was "
                    f"never evaluated; a claim checked nowhere is not a claim satisfied",
                ))
                continue

            tol = claim.tolerance_override or tolerances.forbidden_penetration_m
            shared = {
                "measured": {
                    "max_penetration_m": round(record["max_penetration"], 6),
                    "min_clearance_m": round(float(record["min_clearance"]), 6),
                    "samples_evaluated": record["samples_in_state"],
                },
                "threshold": {"forbidden_penetration_m": tol},
                "evidence": {
                    "worst_configuration": record["max_at"],
                    "first_failing_configuration": record["first_failing"],
                    "closest_configuration": record["min_at"],
                    "distmax": DISTMAX,
                },
            }
            if record["max_penetration"] <= tol:
                results.append(ClaimResult(
                    "KF3.forbidden_pair", subject, Verdict.PASS,
                    f"{claim.parts[0]!r} and {claim.parts[1]!r} stay clear across "
                    f"{record['samples_in_state']} configurations, closest approach "
                    f"{float(record['min_clearance']):.4f} m", **shared,
                ))
            else:
                results.append(ClaimResult(
                    "KF3.forbidden_pair", subject, Verdict.FAIL,
                    f"{claim.parts[0]!r} passes {record['max_penetration']:.4f} m into "
                    f"{claim.parts[1]!r} at {record['max_at']}, first exceeding {tol:g} m at "
                    f"{record['first_failing']}", **shared,
                ))
        return tuple(results)

    def required_contacts(self) -> tuple[ClaimResult, ...]:
        """KF3.required_contact -- a declared stop or mesh actually happens.

        The half of the old ``allowed`` bucket that could never fail. Read as a permission
        it asserted nothing; read as an obligation it needed a tolerance and a state
        predicate, and had neither. As an obligation it catches a travel limit enforced
        only by a ``range`` attribute, with no geometry anywhere that stops the part --
        which is the difference between a mechanism and a declaration.
        """
        results = []
        tolerances = self.contract.kinematic_claims.tolerances
        for key, record in self._track.items():
            if record["bucket"] != "required":
                continue
            claim = record["claim"]
            subject = _pair_label(claim)
            if record["unresolved"]:
                results.append(_na("KF3.required_contact", subject, record["unresolved"]))
                continue
            if record["samples_in_state"] == 0:
                results.append(_na(
                    "KF3.required_contact", subject,
                    f"no swept configuration matched state {claim.state!r}, so the "
                    f"obligation was never evaluated",
                ))
                continue

            tol = claim.tolerance_override or tolerances.required_contact_m
            closest = float(record["min_clearance"])
            shared = {
                "measured": {"min_clearance_m": round(closest, 6),
                             "samples_evaluated": record["samples_in_state"]},
                "threshold": {"required_contact_m": tol},
                "evidence": {"relation": claim.relation.value if claim.relation else None,
                             "closest_configuration": record["min_at"],
                             "state": claim.state},
            }
            if closest <= tol:
                results.append(ClaimResult(
                    "KF3.required_contact", subject, Verdict.PASS,
                    f"{claim.parts[0]!r} and {claim.parts[1]!r} meet in state "
                    f"{claim.state!r}, closest {closest:.4f} m", **shared,
                ))
            else:
                results.append(ClaimResult(
                    "KF3.required_contact", subject, Verdict.FAIL,
                    f"{claim.parts[0]!r} never comes closer than {closest:.4f} m to "
                    f"{claim.parts[1]!r} in state {claim.state!r}, so the declared "
                    f"{claim.relation.value if claim.relation else 'contact'} is not "
                    f"realised by any geometry", **shared,
                ))
        return tuple(results)

    def state_reachability(self) -> tuple[ClaimResult, ...]:
        """KF3.state_reachability -- each named state is reachable without interference.

        Deliberately not "is the state reachable", which is a property of the sampler and
        cannot fail: writing a configuration into ``qpos`` always succeeds. What can fail
        is arriving there through the asset's own geometry, so the state is scored on
        whether any forbidden pair is violated at it.
        """
        results = []
        tolerances = self.contract.kinematic_claims.tolerances
        policy = self.contract.kinematic_claims.contact_policy
        for name in sorted(self.contract.kinematic_claims.asset_states):
            matching = [s for s in self.schedule.samples
                        if self._in_state(s, ContactClaim(parts=("", ""), state=name))
                        or self._state_samples(name, s)]
            samples = [s for s in self.schedule.samples if self._state_samples(name, s)]
            if not samples:
                results.append(_na(
                    "KF3.state_reachability", name,
                    f"no swept configuration matched state {name!r}",
                ))
                continue

            worst, worst_at, worst_pair = 0.0, None, None
            for sample in samples:
                self.sweeper.visit(sample)
                for claim in policy.forbidden:
                    if self._exempt(claim, sample):
                        continue
                    try:
                        a = self.binding.root_body(claim.parts[0])
                        b = self.binding.root_body(claim.parts[1])
                    except BindingError:
                        continue
                    distance, _ = body_pair_distance(self.asset, a, b, DISTMAX)
                    if -distance > worst:
                        worst, worst_at, worst_pair = -distance, sample.label, claim.parts
            tol = tolerances.forbidden_penetration_m
            shared = {
                "measured": {"max_penetration_m": round(worst, 6),
                             "configurations_in_state": len(samples)},
                "threshold": {"forbidden_penetration_m": tol},
                "evidence": {"worst_configuration": worst_at,
                             "worst_pair": list(worst_pair) if worst_pair else None},
            }
            if worst <= tol:
                results.append(ClaimResult(
                    "KF3.state_reachability", name, Verdict.PASS,
                    f"state {name!r} is held with no forbidden interference", **shared,
                ))
            else:
                results.append(ClaimResult(
                    "KF3.state_reachability", name, Verdict.FAIL,
                    f"state {name!r} cannot be held cleanly: {worst_pair[0]!r} and "
                    f"{worst_pair[1]!r} overlap by {worst:.4f} m at {worst_at}", **shared,
                ))
        return tuple(results)

    def _state_samples(self, name: str, sample: Sample) -> bool:
        """Whether a sample sits in a named state, using every joint that declares it."""
        model = self.asset.model
        joints = [
            j for j in self.contract.kinematic_claims.joints if name in (j.states or {})
        ]
        if not joints:
            return False
        definition = self.contract.kinematic_claims.asset_states.get(name)
        tol = definition.tolerance_relative if definition else 0.02
        for joint in joints:
            k = self.sweeper._joints.get(joint.id)
            if k is None:
                return False
            actual = float(sample.qpos[int(model.jnt_qposadr[k])])
            if abs(actual - joint.states[name]) > tol * (joint.range.span or 1.0):
                return False
        return True

    def evaluate(self) -> tuple[ClaimResult, ...]:
        return (
            self.state_reachability()
            + self.forbidden_pairs()
            + self.required_contacts()
        )


def evaluate(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    return Session(contract, binding).evaluate()
