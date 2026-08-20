"""Admission: is this contract checkable at all?

These run before the asset is generated. A contract that fails here is sent back to its
author; nothing is scored and no asset is produced. That timing is the whole point.

The rule that earns the rest of them is A1, referential integrity. In the previous
attempt at these metrics, "part not found" was simultaneously the largest source of false
alarms (42 of 63) and a carrier of real signal (4 of 7 true positives), because it could
mean either "the asset is missing a part" or "the contract used a name nothing binds to".
A paired bootstrap showed no better matcher could pull those apart -- they were the same
event. Deciding the second meaning out of existence *before generation* leaves the flag
with exactly one interpretation.

Every finding names the field path and the offending value, because the reader is a human
about to edit a YAML file, not a metric.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from evo_p0p3.p0.schema import Contract, JointType, Role


class Severity(StrEnum):
    ERROR = "error"
    """The contract cannot be checked. Reject it."""

    WARNING = "warning"
    """The contract is checkable but something in it will never be exercised."""


@dataclass(frozen=True, slots=True)
class Finding:
    rule: str
    path: str
    message: str
    severity: Severity = Severity.ERROR

    def __str__(self) -> str:
        return f"[{self.rule}] {self.path}: {self.message}"


@dataclass(frozen=True, slots=True)
class AdmissionReport:
    contract_id: str
    findings: tuple[Finding, ...]

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    @property
    def admitted(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        if self.admitted and not self.warnings:
            return f"{self.contract_id}: admitted"
        bits = [f"{len(self.errors)} error(s)"]
        if self.warnings:
            bits.append(f"{len(self.warnings)} warning(s)")
        verdict = "admitted" if self.admitted else "REJECTED"
        return f"{self.contract_id}: {verdict} -- {', '.join(bits)}"


def check(contract: Contract) -> AdmissionReport:
    findings: list[Finding] = []
    for rule in _RULES:
        findings.extend(rule(contract))
    return AdmissionReport(contract_id=contract.record_id, findings=tuple(findings))


# --------------------------------------------------------------------------------------
# A1 referential integrity
# --------------------------------------------------------------------------------------


def _referenced_parts(c: Contract) -> Iterator[tuple[str, str]]:
    """Every part id used anywhere, with the path that used it."""
    for pid in c.part_geometry:
        yield pid, f"part_geometry.{pid}"
    for i, rel in enumerate(c.part_relations):
        for key in ("subject", "object"):
            if rel.get(key):
                yield str(rel[key]), f"part_relations[{i}].{key}"
    for i, pc in enumerate(c.proportion_claims):
        for key in ("subject", "object"):
            if pc.get(key):
                yield str(pc[key]), f"proportion_claims[{i}].{key}"
        for j, p in enumerate(pc.get("parts") or []):
            yield str(p), f"proportion_claims[{i}].parts[{j}]"

    k = c.kinematic_claims
    if k.canonical_frame.front:
        yield k.canonical_frame.front, "kinematic_claims.canonical_frame.front"
    for j in k.joints:
        base = f"kinematic_claims.joints[{j.id}]"
        yield j.part, f"{base}.part"
        yield j.parent, f"{base}.parent"
        if j.axis.relational:
            for name in j.axis.relational.referenced_parts():
                yield name, f"{base}.axis.relational"
        if j.anchor:
            for name in j.anchor.referenced_parts():
                yield name, f"{base}.anchor"
    for a in k.rigid_attachments:
        yield a.follower, f"kinematic_claims.rigid_attachments[{a.follower}].follower"
        yield a.leader, f"kinematic_claims.rigid_attachments[{a.follower}].leader"
    for bucket in ("required", "permitted", "forbidden"):
        for i, claim in enumerate(getattr(k.contact_policy, bucket)):
            for p in claim.parts:
                yield p, f"kinematic_claims.contact_policy.{bucket}[{i}].parts"


def _a1_referential_integrity(c: Contract) -> Iterator[Finding]:
    declared = c.part_ids
    for name, path in _referenced_parts(c):
        if name not in declared:
            yield Finding(
                "A1",
                path,
                f"{name!r} is not declared in required_parts. Every id used anywhere must "
                f"be declared, so that a 'part not found' at evaluation time can only mean "
                f"the asset is at fault.",
            )
    joints = c.joint_ids
    k = c.kinematic_claims
    for jid in k.independent_dofs:
        if jid not in joints:
            yield Finding("A1", "kinematic_claims.independent_dofs", f"{jid!r} is not a declared joint")
    for cp in k.couplings:
        for role, jid in (("dependent", cp.relation.dependent), ("independent", cp.relation.independent)):
            if jid not in joints:
                yield Finding(
                    "A1",
                    f"kinematic_claims.couplings[{cp.id}].relation.{role}",
                    f"{jid!r} is not a declared joint",
                )


# --------------------------------------------------------------------------------------
# A2..A12
# --------------------------------------------------------------------------------------


def _a2_part_geometry(c: Contract) -> Iterator[Finding]:
    for p in c.required_parts:
        if p.id not in c.part_geometry:
            yield Finding("A2", f"part_geometry.{p.id}", "every required part needs a part_geometry entry")


def _a3_movable_has_one_joint(c: Contract) -> Iterator[Finding]:
    counts: dict[str, list[str]] = {}
    for j in c.kinematic_claims.joints:
        counts.setdefault(j.part, []).append(j.id)
    for p in c.required_parts:
        owned = counts.get(p.id, [])
        if p.role is Role.MOVABLE and len(owned) != 1:
            yield Finding(
                "A3",
                f"kinematic_claims.joints (part={p.id})",
                f"a movable part needs exactly one joint claim, found {len(owned)}"
                + (f": {', '.join(owned)}" if owned else ""),
            )
        if p.role in (Role.FIXED, Role.ATTACHED) and owned:
            yield Finding(
                "A3",
                f"kinematic_claims.joints (part={p.id})",
                f"a {p.role.value} part must not own a joint, but {', '.join(owned)} claims it. "
                f"Gate G2 would then demand a motion chain the correct asset must not have.",
            )


def _a4_attached_has_leader(c: Contract) -> Iterator[Finding]:
    followers: dict[str, int] = {}
    for a in c.kinematic_claims.rigid_attachments:
        followers[a.follower] = followers.get(a.follower, 0) + 1
    for p in c.required_parts:
        n = followers.get(p.id, 0)
        if p.role is Role.ATTACHED and n != 1:
            yield Finding(
                "A4",
                f"kinematic_claims.rigid_attachments (follower={p.id})",
                f"an attached part must appear as a follower exactly once, found {n}. "
                f"Without it, nothing says which leader it is supposed to ride.",
            )
        if p.role is not Role.ATTACHED and n:
            yield Finding(
                "A4",
                f"kinematic_claims.rigid_attachments (follower={p.id})",
                f"only parts with role 'attached' may be followers; {p.id!r} is {p.role.value}",
            )


def _a5_a6_independent_dofs(c: Contract) -> Iterator[Finding]:
    k = c.kinematic_claims
    indep = set(k.independent_dofs)
    for cp in k.couplings:
        dep = cp.relation.dependent
        if dep in indep:
            yield Finding(
                "A6",
                f"kinematic_claims.couplings[{cp.id}].relation.dependent",
                f"{dep!r} is also listed in independent_dofs; a dependent state is solved "
                f"from the coupling and cannot also be driven independently",
            )
        if cp.relation.independent not in indep:
            yield Finding(
                "A6",
                f"kinematic_claims.couplings[{cp.id}].relation.independent",
                f"{cp.relation.independent!r} is not in independent_dofs, so the sweep would "
                f"never drive it and the coupling would never be exercised",
                Severity.WARNING,
            )
    driven = indep | {cp.relation.dependent for cp in k.couplings}
    for j in k.joints:
        if j.id not in driven:
            yield Finding(
                "A5",
                f"kinematic_claims.independent_dofs (joint={j.id})",
                f"{j.id!r} is neither an independent DOF nor the dependent side of a coupling, "
                f"so no sweep sample will ever move it",
                Severity.WARNING,
            )


def _a7_states_defined(c: Contract) -> Iterator[Finding]:
    k = c.kinematic_claims
    defined = set(k.asset_states) | {"all"}
    for bucket in ("required", "permitted", "forbidden"):
        for i, claim in enumerate(getattr(k.contact_policy, bucket)):
            if claim.state not in defined:
                yield Finding(
                    "A7",
                    f"kinematic_claims.contact_policy.{bucket}[{i}].state",
                    f"{claim.state!r} is not defined in asset_states. An undefined state "
                    f"cannot classify any sweep sample, and a claim that is never evaluated "
                    f"reads as passing.",
                )
    declared_labels = {label for j in k.joints for label in j.states}
    for name in k.asset_states:
        if name not in declared_labels:
            yield Finding(
                "A7",
                f"kinematic_claims.asset_states.{name}",
                f"no joint declares a {name!r} state value, so this state matches nothing",
                Severity.WARNING,
            )


def _a8_a9_axis(c: Contract) -> Iterator[Finding]:
    frame = c.kinematic_claims.canonical_frame
    for j in c.kinematic_claims.joints:
        path = f"kinematic_claims.joints[{j.id}].axis"
        if not j.axis.forms:
            yield Finding("A8", path, "give at least one of semantic, relational or numeric")
        if j.axis.numeric is not None and not frame.front_anchored:
            yield Finding(
                "A9",
                f"{path}.numeric",
                "a numeric axis needs canonical_frame.front anchored to a part, otherwise "
                "the object's yaw is undetermined and this claim can only be reported N/A",
            )


def _a10_anchor_matches_type(c: Contract) -> Iterator[Finding]:
    for j in c.kinematic_claims.joints:
        path = f"kinematic_claims.joints[{j.id}].anchor"
        if j.type.has_anchor and j.anchor is None:
            yield Finding(
                "A10",
                path,
                f"a {j.type.value} rotates about its anchor, so it needs an anchor predicate "
                f"(on_edge_of or through_center_of); it is the field that separates a door "
                f"hinged on its edge from one through its middle",
            )
        if not j.type.has_anchor and j.anchor is not None:
            yield Finding(
                "A10",
                path,
                f"a {j.type.value} joint has no pivot point that affects its kinematics -- "
                f"displacing pos moves the child by exactly zero -- so this claim can never "
                f"be checked",
            )


def _a11_units_and_states(c: Contract) -> Iterator[Finding]:
    for j in c.kinematic_claims.joints:
        base = f"kinematic_claims.joints[{j.id}]"
        if j.range.unit != j.type.unit:
            yield Finding(
                "A11",
                f"{base}.range.unit",
                f"a {j.type.value} joint measures in {j.type.unit!r}, not {j.range.unit!r}",
            )
        for label, value in j.states.items():
            if not (j.range.min <= value <= j.range.max):
                yield Finding(
                    "A11",
                    f"{base}.states.{label}",
                    f"{value} lies outside the declared range "
                    f"[{j.range.min}, {j.range.max}]",
                )


def _a12_contact_shape(c: Contract) -> Iterator[Finding]:
    k = c.kinematic_claims
    for bucket in ("required", "permitted", "forbidden"):
        for i, claim in enumerate(getattr(k.contact_policy, bucket)):
            path = f"kinematic_claims.contact_policy.{bucket}[{i}]"
            if claim.parts[0] == claim.parts[1]:
                yield Finding("A12", f"{path}.parts", "a contact pair needs two distinct parts")
            if bucket == "required" and claim.relation and not claim.relation.is_obligation:
                yield Finding(
                    "A12",
                    f"{path}.relation",
                    f"{claim.relation.value!r} is a permission and cannot sit in 'required'",
                )
            if bucket == "permitted" and claim.relation and claim.relation.is_obligation:
                yield Finding(
                    "A12",
                    f"{path}.relation",
                    f"{claim.relation.value!r} is an obligation and cannot sit in 'permitted'",
                )
            if bucket == "forbidden" and claim.relation is not None:
                yield Finding(
                    "A12",
                    f"{path}.relation",
                    "a forbidden pair asserts absence of penetration and takes no relation token",
                )


def _a13_joint_parent_declared(c: Contract) -> Iterator[Finding]:
    """A joint whose parent is itself, or whose part equals its parent, is unresolvable."""
    for j in c.kinematic_claims.joints:
        if j.part == j.parent:
            yield Finding(
                "A13",
                f"kinematic_claims.joints[{j.id}]",
                f"part and parent are both {j.part!r}; a joint connects two different bodies",
            )


def _a14_coupling_members_are_claimed(c: Contract) -> Iterator[Finding]:
    """Both coupling members must also be joint claims, or KF1 never checks them.

    The dependent joint is not double counted by this: KF1 scores its configuration
    (parent, type, axis, range), KF2 scores the relation between the two. Different
    claims, different failure physics, different repairs.
    """
    joints = c.joint_ids
    for cp in c.kinematic_claims.couplings:
        for role, jid in (
            ("dependent", cp.relation.dependent),
            ("independent", cp.relation.independent),
        ):
            if jid in joints:
                continue
            yield Finding(
                "A14",
                f"kinematic_claims.couplings[{cp.id}].relation.{role}",
                f"{jid!r} must also appear in joints so its own configuration is checked",
            )


def _a15_expected_dof_sane(c: Contract) -> Iterator[Finding]:
    for cp in c.kinematic_claims.couplings:
        if cp.expected_dof < 0:
            yield Finding(
                "A15",
                f"kinematic_claims.couplings[{cp.id}].expected_dof",
                f"{cp.expected_dof} is negative",
            )
        if cp.relation.coefficient == 0.0:
            yield Finding(
                "A15",
                f"kinematic_claims.couplings[{cp.id}].relation.coefficient",
                "a zero coefficient makes the dependent joint constant, which is a weld, "
                "not a coupling ratio",
            )


def _a16_free_joints(c: Contract) -> Iterator[Finding]:
    """A free joint has six DOFs and no single declared range, so range claims are moot."""
    for j in c.kinematic_claims.joints:
        if j.type is JointType.FREE:
            yield Finding(
                "A16",
                f"kinematic_claims.joints[{j.id}].type",
                "a free joint has six degrees of freedom, so its range, states and axis "
                "claims describe only one of them; declare what is actually intended",
                Severity.WARNING,
            )


_RULES = (
    _a1_referential_integrity,
    _a2_part_geometry,
    _a3_movable_has_one_joint,
    _a4_attached_has_leader,
    _a5_a6_independent_dofs,
    _a7_states_defined,
    _a8_a9_axis,
    _a10_anchor_matches_type,
    _a11_units_and_states,
    _a12_contact_shape,
    _a13_joint_parent_declared,
    _a14_coupling_members_are_claimed,
    _a15_expected_dof_sane,
    _a16_free_joints,
)
