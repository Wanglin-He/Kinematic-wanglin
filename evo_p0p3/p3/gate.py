"""A stand-in Initialization Gate.

The Gate is P1's phase, not this one. It is implemented here because P3 cannot run without
two things it produces -- the part-to-body binding, and a decision about which assets are
admitted at all -- and waiting for someone else's module would mean either blocking or
inventing the binding inside P3, which is precisely the inference this architecture
removes. When the real Gate lands, the interface is :class:`Admission` and this file goes.

It is faithful to P1's four checks with one deliberate exception, stated here rather than
buried. **G3 does not gate P3.** MuJoCo refuses to compile a moving body with no mass, so
the loader synthesises inertia for the 341 of 546 real assets that declare none -- and P3's
scope excludes mass, friction, damping and actuation entirely, so nothing it measures could
depend on that value anyway. Failing those assets out of P3 would discard two thirds of the
corpus over a property P3 never reads. The check still runs and its verdict is reported;
it simply is not a precondition here. Whether an asset's real inertial properties are valid
remains a genuine question, and it is P1's.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import mujoco

from evo_p0p3.p0.schema import Contract, Role
from evo_p0p3.p3 import binding as binding_mod
from evo_p0p3.p3.mjcf import AssetLoadError, LoadedAsset, load


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"
    """An earlier check failed, so this one had nothing to run against.

    P1's own scope note asks that each failure be classified under the earliest matching
    gate, and this is what that costs in code. Without it a single missing part produces
    three failures -- G1 for the part, G2 for the motion chain it cannot resolve, G3 for
    the physics it never reached -- and a reader cannot tell which is the cause and which
    are its shadow.
    """


@dataclass(frozen=True, slots=True)
class GateCheck:
    name: str
    status: Status
    detail: str

    @property
    def passed(self) -> bool:
        return self.status is Status.PASS

    def __str__(self) -> str:
        tag = {Status.PASS: "pass", Status.FAIL: "FAIL", Status.NOT_RUN: "----"}[self.status]
        return f"[{tag}] {self.name}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Admission:
    """What the Gate hands to P3."""

    record_id: str
    checks: tuple[GateCheck, ...]
    asset: LoadedAsset | None
    binding: binding_mod.Binding | None

    @property
    def admitted(self) -> bool:
        """Whether P3 may score this asset.

        G3 is excluded on purpose -- see the module docstring. Every other check is a
        precondition: without geometry there is nothing to measure, and without a
        resolvable motion chain there is nothing to measure it against.
        """
        return all(
            c.status is not Status.FAIL for c in self.checks if c.name != "G3"
        ) and all(c.status is Status.PASS for c in self.checks if c.name in ("G0", "G1", "G2"))

    @property
    def failed(self) -> tuple[GateCheck, ...]:
        """Only genuine failures, not the checks they prevented from running."""
        return tuple(c for c in self.checks if c.status is Status.FAIL)

    @property
    def root_cause(self) -> GateCheck | None:
        """The earliest failure. What a person should be told to fix."""
        return self.failed[0] if self.failed else None

    def summary(self) -> str:
        if self.admitted:
            g3 = next((c for c in self.checks if c.name == "G3"), None)
            note = "" if (g3 and g3.passed) else " (G3 reported, not gating)"
            return f"{self.record_id}: admitted{note}"
        cause = self.root_cause
        return f"{self.record_id}: REJECTED at {cause.name}" if cause else (
            f"{self.record_id}: REJECTED"
        )


def admit(
    urdf_path: str | Path,
    contract: Contract,
    *,
    binding_table: str | Path | None = None,
    diagnostic: bool = False,
) -> Admission:
    """Run the four checks and, if possible, produce the binding.

    ``binding_table`` supplies the Gate's part-to-link mapping explicitly, which is what
    the real Gate will emit. Without one, part ids are taken as link names -- true of the
    hand-authored gold assets by construction, and never safe to assume of a generated one.

    ``diagnostic`` downgrades an unbound required part from a G1 failure to an unbound
    entry, so P3 runs on what did bind and reports the rest as N/A. It is not a score. The
    faithful reading of P1 is that a missing required part fails the Gate, and that stays
    the default; this exists so a Gate-level blocker does not make P3 unobservable.
    """
    urdf_path = Path(urdf_path)
    record_id = urdf_path.parent.name if urdf_path.name == "model.urdf" else urdf_path.stem
    checks: list[GateCheck] = []

    # -- G0 readability ---------------------------------------------------------------
    try:
        asset = load(urdf_path, record_id=record_id)
    except AssetLoadError as exc:
        return Admission(
            record_id=record_id,
            checks=(GateCheck("G0", Status.FAIL, str(exc)),),
            asset=None,
            binding=None,
        )
    checks.append(GateCheck(
        "G0", Status.PASS,
        f"compiled: {asset.model.nbody} bodies, {asset.model.njnt} joints, "
        f"{asset.model.ngeom} geoms",
    ))

    # -- G1 geometry existence ---------------------------------------------------------
    table = None
    if binding_table is not None:
        import yaml
        raw = yaml.safe_load(Path(binding_table).read_text(encoding="utf-8")) or {}
        table = {str(k): v for k, v in raw.items()}

    missing_body, empty = [], []
    for part in contract.required_parts:
        link = table.get(part.id) if table is not None else part.id
        body = asset.body_id(str(link)) if link else None
        if body is None:
            missing_body.append(part.id)
        elif not asset.geoms_of(body):
            empty.append(part.id)
    if (missing_body or empty) and not (diagnostic and not empty):
        detail = []
        if missing_body:
            detail.append(f"no body for {missing_body}")
        if empty:
            detail.append(f"no geometry on {empty}")
        checks.append(GateCheck("G1", Status.FAIL, "; ".join(detail)))
    elif missing_body:
        checks.append(GateCheck(
            "G1", Status.PASS,
            f"diagnostic run: {len(contract.required_parts) - len(missing_body)} of "
            f"{len(contract.required_parts)} required parts bound; {missing_body} are "
            f"unbound and every claim about them is reported N/A. NOT A SCORE -- the "
            f"faithful reading is that these fail the Gate",
        ))
    else:
        checks.append(GateCheck(
            "G1", Status.PASS,
            f"all {len(contract.required_parts)} required parts have geometry",
        ))

    binding = None
    if checks[-1].passed:
        bound = {}
        for part in contract.required_parts:
            link = table.get(part.id) if table is not None else part.id
            body = asset.body_id(str(link)) if link else None
            if body is not None:
                bound[part.id] = (body,)
        binding = binding_mod.Binding(
            parts=bound,
            source=(binding_mod.BindingSource.TABLE if table is not None
                    else binding_mod.BindingSource.IDENTITY),
            asset=asset,
        )

    # -- G2 kinematic validity ---------------------------------------------------------
    if binding is None:
        checks.append(GateCheck(
            "G2", Status.NOT_RUN,
            "G1 failed, so there is no binding and no motion chain to resolve",
        ))
        checks.append(GateCheck(
            "G3", Status.NOT_RUN, "an earlier check failed"
        ))
        return Admission(record_id=record_id, checks=tuple(checks), asset=asset, binding=None)

    problems = []
    for part in contract.required_parts:
        if part.id not in binding.parts:
            continue  # unbound in a diagnostic run; G1 already reported it
        body = binding.root_body(part.id)
        count = int(asset.model.body_jntnum[body])
        if part.role is Role.MOVABLE and count != 1:
            problems.append(f"{part.id} is movable but its body carries {count} joints")
        if part.role in (Role.FIXED, Role.ATTACHED) and count:
            problems.append(
                f"{part.id} is {part.role.value} but its body carries {count} joint(s)"
            )
    for coupling in contract.kinematic_claims.couplings:
        for role in (coupling.relation.dependent, coupling.relation.independent):
            joint = contract.kinematic_claims.joint(role)
            # Resolved through the binding, not by looking up the contract's part name as
            # a link name. Bypassing the table here made a coupling look unresolvable
            # whenever the asset simply used different names, which is the confusion
            # between contract fault and asset fault the binding exists to remove.
            if joint is None or joint.part not in binding.parts:
                problems.append(f"coupling {coupling.id} references unresolvable {role}")
    checks.append(GateCheck(
        "G2",
        Status.FAIL if problems else Status.PASS,
        "; ".join(problems)
        or "every required part has a motion chain matching its declared role",
    ))

    # -- G3 physical properties --------------------------------------------------------
    if asset.inertia_synthesized:
        checks.append(GateCheck(
            "G3", Status.FAIL,
            "the source declares no <inertial>; mass and inertia were synthesised to "
            "satisfy the compiler. Reported, but not a precondition for P3, whose scope "
            "excludes mass entirely",
        ))
    else:
        bad = [
            asset.body_name(b)
            for b in range(1, asset.model.nbody)
            if float(asset.model.body_mass[b]) <= mujoco.mjMINVAL
        ]
        checks.append(GateCheck(
            "G3", Status.FAIL if bad else Status.PASS,
            f"non-positive mass on {bad}" if bad else "declared masses and inertias compile",
        ))

    return Admission(
        record_id=record_id,
        checks=tuple(checks),
        asset=asset,
        binding=binding if all(c.passed for c in checks if c.name in ("G0", "G1", "G2")) else None,
    )
