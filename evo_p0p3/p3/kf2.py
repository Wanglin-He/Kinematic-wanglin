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

from dataclasses import dataclass

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


@dataclass(frozen=True, slots=True)
class _Relation:
    """The affine relation the model enforces between two joints, however it spells it.

    ``q_dep = coefficient * q_ind + offset``, recovered by composing every equality along
    the shortest chain that connects them. A directly written equality is the one-edge case.
    """

    coefficient: float
    offset: float
    edges: tuple[int, ...]
    active: bool
    higher_order: float
    waypoints: tuple[tuple[int, float, float], ...]
    """Joints between the two ends, each with the model's own ``(slope, offset)`` from the
    independent joint. The residual sweep needs them: a chain routed through a third joint
    is only meaningful once that joint sits where the model says it should."""


def _equality_graph(model, *, active_only: bool = False) -> dict[int, list]:
    """Adjacency for the joint-equality graph.

    An edge ``u -> (v, m, b, e)`` reads "equality e constrains q_v = m*q_u + b". MuJoCo
    writes ``q_obj1 = a0 + a1 q_obj2``, which is an edge obj2 -> obj1; the inverse edge
    exists whenever a1 is non-zero, because the same constraint read the other way round
    describes the same mechanism. That is also what makes a model writing the relation
    backwards pass rather than fail over which way round it was typed.
    """
    graph: dict[int, list] = {}
    for e in range(model.neq):
        if model.eq_type[e] != mujoco.mjtEq.mjEQ_JOINT:
            continue
        if active_only and not bool(model.eq_active0[e]):
            continue
        o1, o2 = int(model.eq_obj1id[e]), int(model.eq_obj2id[e])
        a0, a1 = float(model.eq_data[e][0]), float(model.eq_data[e][1])
        graph.setdefault(o2, []).append((o1, a1, a0, e))
        if abs(a1) > 1e-12:
            graph.setdefault(o1, []).append((o2, 1.0 / a1, -a0 / a1, e))
    return graph


def _relation(binding: Binding, dep: int, ind: int) -> _Relation | None:
    """What the model enforces between these two joints, or None if nothing does.

    Chains are followed, not only equalities written directly between the pair, because P0
    declares a *relation* and says ``mechanism: any``. The claim is about how two parts move
    together, not about which constraint object implements it, and an earlier version that
    demanded a direct equality contradicted the contract's own stated semantics.

    The glove compartment is the case that settled it. Its contract declares the twin
    limiter links move 1:1 -- the only ratio the prompt supports, since it never gives a
    door-to-link figure -- and the asset slaves both limiters to the door at 0.55. The two
    limiters therefore do move exactly 1:1; requiring an equality written between them
    reported a correct mechanism as absent and scored the asset zero.

    Composing gives nothing away. The composed coefficient still has to match the declared
    one, the residual sweep still runs, and a chain routed through an unconstrained joint
    yields no path at all. It stays arithmetic: multiply the slopes, carry the offsets.
    """
    model = binding.asset.model
    graph = _equality_graph(model)
    # Breadth-first, so the shortest chain wins: a model carrying both a direct equality
    # and a longer route is read the direct way.
    frontier = [(ind, 1.0, 0.0, (), ())]
    seen = {ind}
    while frontier:
        node, m, b, edges, nodes = frontier.pop(0)
        for nxt, edge_m, edge_b, e in graph.get(node, ()):
            if nxt in seen:
                continue
            slope, offset, path = edge_m * m, edge_m * b + edge_b, edges + (e,)
            if nxt == dep:
                return _Relation(
                    coefficient=slope,
                    offset=offset,
                    edges=path,
                    active=all(bool(model.eq_active0[x]) for x in path),
                    higher_order=max(
                        float(np.abs(np.asarray(model.eq_data[x][2:5], dtype=float)).max())
                        for x in path
                    ),
                    waypoints=nodes,
                )
            seen.add(nxt)
            frontier.append((nxt, slope, offset, path, nodes + ((nxt, slope, offset),)))
    return None


def _resolve(binding: Binding, contract: Contract, coupling: Coupling):
    dep = _mj_joint(binding, contract, coupling.relation.dependent)
    ind = _mj_joint(binding, contract, coupling.relation.independent)
    if dep is None or ind is None:
        return None, None, None, (
            f"could not resolve {coupling.relation.dependent!r} or "
            f"{coupling.relation.independent!r} to a single MuJoCo joint"
        )
    return dep, ind, _relation(binding, dep, ind), None


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
        chain = len(found.edges) if found else 0
        shared = {
            "measured": {"chain_length": chain,
                         "active": bool(found and found.active)},
            "evidence": {
                "dependent": coupling.relation.dependent,
                "independent": coupling.relation.independent,
                "equality_ids": list(found.edges) if found else [],
                "total_equalities_in_model": int(model.neq),
                "mimics_recovered": [m.dependent for m in binding.asset.mimics],
            },
        }
        if found and found.active:
            how = (
                "a joint equality" if chain == 1
                else f"a chain of {chain} joint equalities"
            )
            results.append(ClaimResult(
                "KF2.bound", coupling.id, Verdict.PASS,
                f"{how} actively links {coupling.relation.dependent!r} and "
                f"{coupling.relation.independent!r}", **shared,
            ))
        elif found:
            results.append(ClaimResult(
                "KF2.bound", coupling.id, Verdict.FAIL,
                "a constraint links the two joints but is inactive, so nothing enforces "
                "the declared relation", **shared,
            ))
        else:
            results.append(ClaimResult(
                "KF2.bound", coupling.id, Verdict.FAIL,
                f"nothing in the model links {coupling.relation.dependent!r} to "
                f"{coupling.relation.independent!r}, directly or through any chain of "
                f"equalities; the two joints move independently, so the declared coupling "
                f"exists only in the contract", **shared,
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

        got_c, got_o = found.coefficient, found.offset
        higher = found.higher_order
        want_c, want_o = coupling.relation.coefficient, coupling.relation.offset
        if abs(got_c) < 1e-12:
            results.append(ClaimResult(
                "KF2.coefficient", coupling.id, Verdict.FAIL,
                "the constraint pins the dependent joint to a constant rather than "
                "coupling it to the independent one",
                measured={"coefficient": got_c, "offset": got_o},
            ))
            continue

        tol = coupling.epsilon or contract.kinematic_claims.tolerances.coupling_residual
        rel = abs(got_c - want_c) / (abs(want_c) or 1.0)
        shared = {
            "measured": {"coefficient": round(got_c, 6), "offset": round(got_o, 6),
                         "chain_length": len(found.edges),
                         "higher_order_terms": round(higher, 9)},
            "threshold": {"coefficient": want_c, "offset": want_o, "epsilon": tol},
            "evidence": {"dependent": coupling.relation.dependent,
                         "independent": coupling.relation.independent,
                         "equality_ids": list(found.edges)},
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

    Counted as member joint DOFs minus the freedoms the active equalities remove, where
    "removes a freedom" means two members land in the same connected component of the
    equality graph -- so a linkage routed through a third joint counts, matching what
    KF2.bound accepts. Grouping rather than counting rows also removes an over-count the
    row version had: two redundant equalities on one pair used to subtract two freedoms
    where physically they remove one.
    """
    results = []
    for coupling in contract.kinematic_claims.couplings:
        dep, ind, _, problem = _resolve(binding, contract, coupling)
        if problem:
            results.append(_na(coupling.id, "KF2.expected_dof", problem))
            continue

        model = binding.asset.model
        members = [dep, ind]
        dofs = sum(_DOFS_PER_TYPE[model.jnt_type[j]] for j in members)

        # How many independent motions the members retain *among themselves*. Two members
        # in the same connected component of the active equality graph have one motion
        # between them, whether the constraint is written directly between them or routed
        # through a third joint. Counting components rather than equality rows also drops
        # the old over-count: two redundant equalities on one pair form one component and
        # remove one freedom, which is what they physically do.
        graph = _equality_graph(model, active_only=True)

        def _reachable(start: int) -> set[int]:
            seen, stack = {start}, [start]
            while stack:
                node = stack.pop()
                for nxt, *_ in graph.get(node, ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            return seen

        remaining, components = set(members), 0
        while remaining:
            remaining -= _reachable(remaining.pop())
            components += 1
        constraints = len(members) - components
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
        declared = contract.kinematic_claims.joint(coupling.relation.independent)
        lo, hi = declared.range.min, declared.range.max

        worst, worst_q = 0.0, None
        first_failure = None
        tol = coupling.epsilon or contract.kinematic_claims.tolerances.coupling_residual

        for value in np.linspace(lo, hi, SAMPLES):
            mujoco.mj_resetData(model, data)
            data.qpos[int(model.jnt_qposadr[ind])] = value
            # Intermediate joints go where the *model* says, so the reading below is about
            # the declared relation and not about a waypoint left at zero.
            for node, slope, offset in found.waypoints:
                data.qpos[int(model.jnt_qposadr[node])] = slope * value + offset
            # The dependent end goes where the *contract* says. The gap between the two
            # descriptions is exactly what the equality residual then reports.
            data.qpos[int(model.jnt_qposadr[dep])] = (
                coupling.relation.coefficient * value + coupling.relation.offset
            )
            mujoco.mj_forward(model, data)
            on_chain = set(found.edges)
            rows = [
                i for i in range(data.nefc)
                if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY
                and int(data.efc_id[i]) in on_chain
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
