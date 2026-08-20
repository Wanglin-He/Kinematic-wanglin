"""KF2 -- coupling fidelity.

Does a mechanism in the model actually enforce the linkage P0 declared, or do two joints
merely happen to be drawn near each other?

The printed formula cannot answer that. ``KF2 = mean_g 1[max_q |r| <= eps]`` reads a
residual, and a residual exists only if the model instantiated a constraint. A generator
that declares a gearbox and links nothing produces no residual at all; a maximum over an
empty set makes the indicator true, and the single most severe coupling failure there is
takes full marks. So the formula carries a binding factor:

    KF2 = mean_g  1[bound(g)] * 1[max_q |r_g,norm(q)| <= eps_g]

The specification's prose bullet "check that coupling is active" was reaching for this,
but prose does not run. ``gearbox_missing_coupling`` is the asset that settles it.

Everything here reads MuJoCo's own equality residual after ``mj_forward`` at a written
configuration -- position level, no stepping, no dynamics, inside P3's declared scope.
"""

from __future__ import annotations

import mujoco
import numpy as np

from evo_p0p3.p0.schema import Contract, Coupling, ResidualNorm
from evo_p0p3.p3.binding import Binding, BindingError
from evo_p0p3.p3.verdict import ClaimResult, Verdict

_DOFS_PER_TYPE = {
    mujoco.mjtJoint.mjJNT_FREE: 6,
    mujoco.mjtJoint.mjJNT_BALL: 3,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}
"""Degrees of freedom a joint contributes.

Derived from the type rather than read from a field: MuJoCo exposes ``jnt_dofadr`` but no
``jnt_dofnum``, and reaching for the second is how an invented API gets into code that
otherwise looks right.
"""

SAMPLES = 33
"""Points across the independent joint's declared range.

Endpoint-inclusive and odd, so the reference configuration falls exactly on a sample. A
wrong ratio produces a residual proportional to how far the input has turned, so the
endpoints carry the most signal -- but a wrong *offset* shows up at the reference and
nowhere else, which is why the middle is sampled too.
"""


def _na(subject: str, predicate: str, reason: str, **evidence) -> ClaimResult:
    return ClaimResult(
        predicate=predicate, subject=subject, verdict=Verdict.NA, reason=reason,
        evidence=evidence,
    )


def _mj_joint(binding: Binding, contract: Contract, joint_id: str) -> int | None:
    """The MuJoCo joint implementing a declared joint, resolved through its part's body."""
    joint = contract.kinematic_claims.joint(joint_id)
    if joint is None:
        return None
    try:
        body = binding.root_body(joint.part)
    except BindingError:
        return None
    model = binding.asset.model
    start, count = int(model.body_jntadr[body]), int(model.body_jntnum[body])
    return start if count == 1 else None


def _equalities_for(binding: Binding, dep: int, ind: int) -> list[tuple[int, bool]]:
    """Every joint-equality linking these two joints, with whether the roles are swapped.

    A model may write ``q_ind = a0 + a1 q_dep`` instead of ``q_dep = a0 + a1 q_ind``. The
    two describe the same mechanism, so both are accepted and the coefficient comparison
    inverts accordingly rather than failing an asset over which way round it was typed.
    """
    model = binding.asset.model
    out = []
    for e in range(model.neq):
        if model.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        o1, o2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        if (o1, o2) == (dep, ind):
            out.append((e, False))
        elif (o1, o2) == (ind, dep):
            out.append((e, True))
    return out


def _resolve(binding: Binding, contract: Contract, coupling: Coupling):
    dep = _mj_joint(binding, contract, coupling.relation.dependent)
    ind = _mj_joint(binding, contract, coupling.relation.independent)
    if dep is None or ind is None:
        return None, None, None, (
            f"could not resolve {coupling.relation.dependent!r} or "
            f"{coupling.relation.independent!r} to a single MuJoCo joint"
        )
    found = _equalities_for(binding, dep, ind)
    return dep, ind, (found[0] if found else None), None


# --------------------------------------------------------------------------------------


def bound(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """KF2.bound -- some mechanism in the model actually implements this coupling.

    The factor the printed formula is missing. Without it an asset whose gears spin
    independently scores a perfect one, because there is no constraint to produce a
    residual and a maximum over nothing is vacuously within tolerance.

    Note what this does *not* do: it never infers a coupling from two joints that happen
    to move together. A relation nothing enforces is not a relation, whatever the numbers
    look like at a particular configuration.
    """
    results = []
    for coupling in contract.kinematic_claims.couplings:
        dep, ind, found, problem = _resolve(binding, contract, coupling)
        if problem:
            results.append(_na(coupling.id, "KF2.bound", problem))
            continue

        model = binding.asset.model
        equalities = _equalities_for(binding, dep, ind)
        active = [e for e, _ in equalities if bool(model.eq_active0[e])]
        shared = {
            "measured": {"constraints_found": len(equalities), "active": len(active)},
            "evidence": {
                "dependent": coupling.relation.dependent,
                "independent": coupling.relation.independent,
                "total_equalities_in_model": int(model.neq),
                "mimics_recovered": [m.dependent for m in binding.asset.mimics],
            },
        }
        if active:
            results.append(ClaimResult(
                "KF2.bound", coupling.id, Verdict.PASS,
                f"an active joint equality links {coupling.relation.dependent!r} and "
                f"{coupling.relation.independent!r}", **shared,
            ))
        elif equalities:
            results.append(ClaimResult(
                "KF2.bound", coupling.id, Verdict.FAIL,
                f"a constraint links the two joints but is inactive, so nothing enforces "
                f"the declared relation", **shared,
            ))
        else:
            results.append(ClaimResult(
                "KF2.bound", coupling.id, Verdict.FAIL,
                f"nothing in the model links {coupling.relation.dependent!r} to "
                f"{coupling.relation.independent!r}; the two joints move independently, so "
                f"the declared coupling exists only in the contract", **shared,
            ))
    return tuple(results)


def coefficient(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """KF2.coefficient -- the ratio and offset the model enforces are the declared ones.

    Read straight off ``eq_data``, whose polynomial form
    ``q_1 = p0 + p1 q_2 + p2 q_2^2 + ...`` is exactly URDF's mimic semantics and exactly
    P0's ``dependent = coefficient * independent + offset``. Nothing is interpreted; the
    fields line up one to one.

    Sign is part of the ratio, not a convention. External gears counter-rotate, so a
    gearbox declaring -3 and built +3 describes a mechanism that cannot exist, and this
    fires on it.
    """
    results = []
    for coupling in contract.kinematic_claims.couplings:
        dep, ind, found, problem = _resolve(binding, contract, coupling)
        if problem:
            results.append(_na(coupling.id, "KF2.coefficient", problem))
            continue
        if found is None:
            results.append(_na(
                coupling.id, "KF2.coefficient",
                "no constraint implements this coupling, so there are no coefficients to "
                "compare; KF2.bound owns that failure",
            ))
            continue

        index, swapped = found
        poly = np.asarray(binding.asset.model.eq_data[index][:5], dtype=float)
        want_c, want_o = coupling.relation.coefficient, coupling.relation.offset

        if swapped:
            # The model wrote q_ind = a0 + a1 q_dep; the same mechanism inverted.
            a0, a1 = float(poly[0]), float(poly[1])
            if abs(a1) < 1e-12:
                results.append(ClaimResult(
                    "KF2.coefficient", coupling.id, Verdict.FAIL,
                    "the constraint pins the independent joint to a constant rather than "
                    "coupling it", measured={"polycoef": poly.round(6).tolist()},
                ))
                continue
            got_c, got_o = 1.0 / a1, -a0 / a1
        else:
            got_c, got_o = float(poly[1]), float(poly[0])

        higher = float(np.abs(poly[2:]).max())
        tol = coupling.epsilon or contract.kinematic_claims.tolerances.coupling_residual
        rel = abs(got_c - want_c) / (abs(want_c) or 1.0)
        shared = {
            "measured": {"coefficient": round(got_c, 6), "offset": round(got_o, 6),
                         "polycoef": poly.round(6).tolist(),
                         "roles_swapped": swapped,
                         "higher_order_terms": round(higher, 9)},
            "threshold": {"coefficient": want_c, "offset": want_o, "epsilon": tol},
            "evidence": {"dependent": coupling.relation.dependent,
                         "independent": coupling.relation.independent},
        }
        if rel <= tol and abs(got_o - want_o) <= tol and higher <= 1e-9:
            results.append(ClaimResult(
                "KF2.coefficient", coupling.id, Verdict.PASS,
                f"the model enforces {got_c:+.4g} as declared", **shared,
            ))
        elif np.sign(got_c) != np.sign(want_c):
            results.append(ClaimResult(
                "KF2.coefficient", coupling.id, Verdict.FAIL,
                f"the model enforces {got_c:+.4g} where P0 declares {want_c:+.4g}; the sign "
                f"is inverted, so the members turn together where they should oppose",
                **shared,
            ))
        else:
            results.append(ClaimResult(
                "KF2.coefficient", coupling.id, Verdict.FAIL,
                f"the model enforces {got_c:+.4g} where P0 declares {want_c:+.4g}, off by "
                f"{rel:.0%}", **shared,
            ))
    return tuple(results)


def expected_dof(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """KF2.expected_dof -- the members have as many degrees of freedom left as declared.

    What separates "constrained to each other" from "moving together by coincidence".
    Counted as member joint DOFs minus the active equalities acting among them: two hinges
    and one joint equality leave one.

    Counting rows rather than taking the rank of the constraint Jacobian assumes those
    rows are independent. For a single equality between two joints they trivially are.
    Two redundant equalities on one pair would be over-counted here and under-report the
    remaining freedom -- recorded as a limitation rather than papered over, and it does not
    arise in the corpus, where couplings come from ``<mimic>`` and MuJoCo permits one per
    joint.
    """
    results = []
    for coupling in contract.kinematic_claims.couplings:
        dep, ind, _, problem = _resolve(binding, contract, coupling)
        if problem:
            results.append(_na(coupling.id, "KF2.expected_dof", problem))
            continue

        model = binding.asset.model
        members = {dep, ind}
        dofs = sum(_DOFS_PER_TYPE[model.jnt_type[j]] for j in members)
        constraints = sum(
            1
            for e in range(model.neq)
            if model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT
            and bool(model.eq_active0[e])
            and {int(model.eq_obj1id[e]), int(model.eq_obj2id[e])} <= members
        )
        observed = dofs - constraints
        shared = {
            "measured": {"member_dofs": dofs, "active_constraints": constraints,
                         "remaining_dof": observed},
            "threshold": {"expected_dof": coupling.expected_dof},
            "evidence": {"members": [coupling.relation.dependent,
                                     coupling.relation.independent]},
        }
        if observed == coupling.expected_dof:
            results.append(ClaimResult(
                "KF2.expected_dof", coupling.id, Verdict.PASS,
                f"{dofs} member degrees of freedom less {constraints} constraint leaves "
                f"{observed}, as declared", **shared,
            ))
        else:
            results.append(ClaimResult(
                "KF2.expected_dof", coupling.id, Verdict.FAIL,
                f"P0 declares {coupling.expected_dof} remaining degree(s) of freedom and "
                f"the model leaves {observed}: {dofs} member DOFs with {constraints} "
                f"constraint(s) among them", **shared,
            ))
    return tuple(results)


def residual(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """KF2.residual -- placed on the declared manifold, the model's own constraint agrees.

    The independent joint is swept across its declared range, the dependent one is written
    from the target relation, and MuJoCo's equality violation is read at each sample. If
    the model enforces what P0 declared, the two agree everywhere and the residual is zero.
    If it enforces a different ratio, the disagreement grows with how far the input has
    turned; a wrong offset shows up even at the reference.

    Position level only: ``mj_forward`` fills ``efc_pos`` without any stepping, so no mass,
    friction, damping or actuation enters -- which is what P3's scope requires.

    Measured on the gold gearbox: 0.000000 at the declared ratio, and up to 6.28 rad on the
    same asset built 2:1.
    """
    results = []
    for coupling in contract.kinematic_claims.couplings:
        dep, ind, found, problem = _resolve(binding, contract, coupling)
        if problem:
            results.append(_na(coupling.id, "KF2.residual", problem))
            continue
        if found is None:
            results.append(_na(
                coupling.id, "KF2.residual",
                "no constraint implements this coupling, so there is no residual to read; "
                "KF2.bound owns that failure",
            ))
            continue

        asset = binding.asset
        model, data = asset.model, asset.data
        index, _ = found
        declared = contract.kinematic_claims.joint(coupling.relation.independent)
        lo, hi = declared.range.min, declared.range.max

        worst, worst_q = 0.0, None
        first_failure = None
        tol = coupling.epsilon or contract.kinematic_claims.tolerances.coupling_residual

        for value in np.linspace(lo, hi, SAMPLES):
            mujoco.mj_resetData(model, data)
            data.qpos[int(model.jnt_qposadr[ind])] = value
            data.qpos[int(model.jnt_qposadr[dep])] = (
                coupling.relation.coefficient * value + coupling.relation.offset
            )
            mujoco.mj_forward(model, data)
            rows = [
                i for i in range(data.nefc)
                if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
                and int(data.efc_id[i]) == index
            ]
            if not rows:
                continue
            magnitude = float(np.abs(data.efc_pos[rows]).max())
            if coupling.residual_norm is ResidualNorm.DEPENDENT_RANGE_SPAN:
                magnitude /= contract.kinematic_claims.joint(
                    coupling.relation.dependent
                ).range.span or 1.0
            elif coupling.residual_norm is ResidualNorm.INDEPENDENT_RANGE_SPAN:
                magnitude /= declared.range.span or 1.0
            if magnitude > worst:
                worst, worst_q = magnitude, float(value)
            if magnitude > tol and first_failure is None:
                first_failure = float(value)
        mujoco.mj_forward(model, data)

        shared = {
            "measured": {"max_residual": round(worst, 9),
                         "at_independent_q": worst_q,
                         "first_failing_q": first_failure,
                         "samples": SAMPLES},
            "threshold": {"epsilon": tol, "normalisation": coupling.residual_norm.value},
            "evidence": {"dependent": coupling.relation.dependent,
                         "independent": coupling.relation.independent,
                         "swept_range": [lo, hi]},
        }
        if worst <= tol:
            results.append(ClaimResult(
                "KF2.residual", coupling.id, Verdict.PASS,
                f"the model's constraint agrees with the declared relation across the whole "
                f"range, worst disagreement {worst:.2e}", **shared,
            ))
        else:
            results.append(ClaimResult(
                "KF2.residual", coupling.id, Verdict.FAIL,
                f"placed on the declared manifold the model's own constraint is violated by "
                f"up to {worst:.4g}, first exceeding {tol:g} at "
                f"{coupling.relation.independent} = {first_failure:.4g}", **shared,
            ))
    return tuple(results)


PREDICATES = (bound, coefficient, expected_dof, residual)


def evaluate(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """Every KF2 result. Empty when the contract declares no coupling, which the profile
    reports as N/A rather than as either extreme -- an unmeasured dimension is not a
    perfect one."""
    results: list[ClaimResult] = []
    for predicate in PREDICATES:
        results.extend(predicate(contract, binding))
    return tuple(results)
