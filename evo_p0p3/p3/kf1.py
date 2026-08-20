"""KF1 -- articulation specification fidelity.

Does the model implement the motion chain and joint configuration P0 froze? Each predicate
lands together with the gold-standard asset that makes it fail. That pairing is not
ceremony: a predicate no input can fail passes everything, and against a corpus of
mostly-correct assets that is indistinguishable from a corpus of correct assets. This
project has been caught by exactly that twice -- a fault injection displacing a joint
origin by a whole link diagonal detected nothing, because the origin IS the child frame
origin and moving it carries the geometry along.

Two of the specification's four KF1 bullets could not be implemented as written, and both
failures were of that kind. "A slide introduces no extra rotation" is true of every MuJoCo
slide by construction; "a hinge anchor remains fixed under a small perturbation" is true of
every hinge, because mj_kinematics derives the anchor from the declared jnt_pos. Neither
admits a failing model. What replaces them is :func:`dof_composition` and :func:`anchor`,
which compare the model against something outside itself.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from evo_p0p3.p0.schema import Contract, Joint, JointType
from evo_p0p3.p3.binding import Binding, BindingError
from evo_p0p3.p3.mjcf import aabb_diagonal, body_pair_distance, subtree_aabb
from evo_p0p3.p3.verdict import ClaimResult, Verdict

_MJ_TYPE = {
    JointType.SLIDE: mujoco.mjtJoint.mjJNT_SLIDE,
    JointType.HINGE: mujoco.mjtJoint.mjJNT_HINGE,
    JointType.BALL: mujoco.mjtJoint.mjJNT_BALL,
    JointType.FREE: mujoco.mjtJoint.mjJNT_FREE,
}

_BOX_CORNERS = np.array(
    [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float
)


# --------------------------------------------------------------------------------------
# shared machinery
# --------------------------------------------------------------------------------------


def _na(predicate: str, subject: str, reason: str, **evidence) -> ClaimResult:
    return ClaimResult(
        predicate=predicate, subject=subject, verdict=Verdict.NA, reason=reason,
        evidence=evidence,
    )


def _joints_of(binding: Binding, body: int) -> tuple[int, ...]:
    model = binding.asset.model
    start = int(model.body_jntadr[body])
    count = int(model.body_jntnum[body])
    return tuple(range(start, start + count)) if count > 0 else ()


def _resolve(binding: Binding, joint: Joint) -> tuple[int | None, int | None, str | None]:
    """The child body and the single MuJoCo joint that moves it.

    Resolved through the body rather than by joint name: names in a generated asset are
    the generator's, and matching them is the inference this architecture exists to remove.
    A body carrying more than one joint is left unresolved here and reported by
    :func:`dof_composition`, which is the predicate that owns that failure.
    """
    try:
        child = binding.root_body(joint.part)
    except BindingError as exc:
        return None, None, str(exc)
    owned = _joints_of(binding, child)
    if len(owned) != 1:
        return child, None, (
            f"body {binding.asset.body_name(child)!r} carries {len(owned)} joints, so "
            f"which one implements {joint.id!r} is undetermined"
        )
    return child, owned[0], None


def _reference_pose(contract: Contract, binding: Binding) -> None:
    """Put the model at the configuration P0 calls closed, or home, or the range minimum.

    Every geometric KF1 predicate reads from one frozen pose, so that two predicates
    disagreeing is a disagreement about the claim rather than about where the asset was
    standing at the time.
    """
    asset = binding.asset
    mujoco.mj_resetData(asset.model, asset.data)
    for joint in contract.kinematic_claims.joints:
        _, k, _ = _resolve(binding, joint)
        if k is None:
            continue
        if asset.model.jnt_type[k] not in (
            mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE
        ):
            continue
        value = joint.states.get("closed", joint.states.get("home", joint.range.min))
        asset.data.qpos[int(asset.model.jnt_qposadr[k])] = value
    mujoco.mj_forward(asset.model, asset.data)


def _points_of(binding: Binding, bodies: tuple[int, ...]) -> np.ndarray:
    """A deterministic world-frame point cloud for a set of bodies.

    Box corners exactly; other primitives and meshes fall back to their local AABB corners,
    which is conservative in the only direction that matters here -- it can only make a
    part look bigger, never smaller, so a part is never judged to sit on one side of a line
    because the sampling missed the part of it that does not.
    """
    asset = binding.asset
    out = []
    for body in bodies:
        for g in asset.geoms_of(body):
            rot = np.asarray(asset.data.geom_xmat[g], dtype=float).reshape(3, 3)
            origin = np.asarray(asset.data.geom_xpos[g], dtype=float)
            if asset.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                half = np.asarray(asset.model.geom_size[g], dtype=float)
                local = _BOX_CORNERS * half
            else:
                centre = np.asarray(asset.model.geom_aabb[g][:3], dtype=float)
                half = np.asarray(asset.model.geom_aabb[g][3:], dtype=float)
                local = centre + _BOX_CORNERS * half
            out.extend(origin + local @ rot.T)
    return np.asarray(out, dtype=float) if out else np.zeros((0, 3))


# --------------------------------------------------------------------------------------
# predicates
# --------------------------------------------------------------------------------------


def parent(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """KF1.parent -- each joint moves its part relative to the part P0 named.

    Decided on the **nearest declared ancestor**, not on ancestry alone. A lower drawer
    hung off an upper drawer still has the carcass somewhere up its chain, so "is the
    declared parent an ancestor?" passes an asset where opening the upper drawer drags the
    lower one out with it. The claim is about what the part moves *relative to*, and that
    is the first declared part above it.

    Bodies the contract never declared are transparent: generators insert unnamed
    intermediate links freely, and treating one as a parent would fail correct assets over
    an authoring style the contract says nothing about.
    """
    results = []
    for joint in contract.kinematic_claims.joints:
        try:
            child = binding.root_body(joint.part)
        except BindingError as exc:
            results.append(_na("KF1.parent", joint.id, str(exc), part=joint.part))
            continue

        observed, steps = binding.nearest_declared_ancestor(child)
        model = binding.asset.model
        chain, walker = [], int(model.body_parentid[child])
        for _ in range(steps):
            chain.append(binding.asset.body_name(walker))
            if walker == 0:
                break
            walker = int(model.body_parentid[walker])

        shared = {
            "measured": {"nearest_declared_ancestor": observed, "links_up": steps},
            "evidence": {
                "part": joint.part,
                "child_body": binding.asset.body_name(child),
                "body_chain_upward": chain,
                "binding_source": binding.source.value,
            },
        }
        if observed == joint.parent:
            results.append(ClaimResult(
                "KF1.parent", joint.id, Verdict.PASS,
                f"{joint.part!r} moves relative to {joint.parent!r} as declared", **shared,
            ))
        elif observed is None:
            results.append(ClaimResult(
                "KF1.parent", joint.id, Verdict.FAIL,
                f"{joint.part!r} hangs off no declared part at all; P0 says its parent is "
                f"{joint.parent!r}", **shared,
            ))
        else:
            results.append(ClaimResult(
                "KF1.parent", joint.id, Verdict.FAIL,
                f"{joint.part!r} moves relative to {observed!r}, but P0 declares "
                f"{joint.parent!r}; driving {observed!r} therefore carries {joint.part!r} "
                f"with it", **shared,
            ))
    return tuple(results)


def joint_type(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """KF1.type -- the joint is of the kind P0 declared.

    A declaration read, and honestly the weakest predicate here: in the previous corpus
    joint type parsed at 100% completeness and matched the requirement in nearly every
    asset, and six assets declared verbatim the type the prompt asked for while human
    annotators still failed them. It is kept because it is cheap and its failure is
    unambiguous, and it is reported in the declaration-reading sub-score rather than mixed
    in with the measured ones, so it cannot dilute them.
    """
    results = []
    for joint in contract.kinematic_claims.joints:
        _, k, problem = _resolve(binding, joint)
        if k is None:
            results.append(_na("KF1.type", joint.id, problem or "unresolved", part=joint.part))
            continue
        observed = binding.asset.model.jnt_type[k]
        expected = _MJ_TYPE[joint.type]
        name = {v: k2.value for k2, v in _MJ_TYPE.items()}.get(observed, str(observed))
        shared = {
            "measured": {"joint_type": name},
            "threshold": {"declared": joint.type.value},
            "evidence": {"mj_joint": binding.asset.joint_name(k)},
        }
        if observed == expected:
            results.append(ClaimResult(
                "KF1.type", joint.id, Verdict.PASS, f"declared and built as {joint.type.value}",
                **shared,
            ))
        else:
            results.append(ClaimResult(
                "KF1.type", joint.id, Verdict.FAIL,
                f"P0 declares {joint.type.value}, the model builds {name}", **shared,
            ))
    return tuple(results)


def dof_composition(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """KF1.dof_composition -- one declared degree of freedom is one MuJoCo joint.

    This is half of what the specification's fourth KF1 bullet was reaching for. "A slide
    introduces no extra rotation" cannot fail -- a MuJoCo slide is a pure translation by
    construction -- but the failure it was aimed at is real and is here: a part given a
    free joint, or a slide stacked with a hinge through a dummy link, moves in ways the
    single declared DOF does not describe. Counting the joints on the body is a test the
    model can fail; asking whether a slide rotates is not.
    """
    results = []
    for joint in contract.kinematic_claims.joints:
        try:
            child = binding.root_body(joint.part)
        except BindingError as exc:
            results.append(_na("KF1.dof_composition", joint.id, str(exc), part=joint.part))
            continue
        owned = _joints_of(binding, child)
        names = [binding.asset.joint_name(j) for j in owned]
        shared = {
            "measured": {"joints_on_body": len(owned)},
            "threshold": {"expected": 1},
            "evidence": {"joint_names": names, "body": binding.asset.body_name(child)},
        }
        if len(owned) == 1:
            results.append(ClaimResult(
                "KF1.dof_composition", joint.id, Verdict.PASS,
                "one declared degree of freedom, one joint", **shared,
            ))
        else:
            results.append(ClaimResult(
                "KF1.dof_composition", joint.id, Verdict.FAIL,
                f"P0 declares one degree of freedom for {joint.part!r}, the model gives its "
                f"body {len(owned)}: {names}", **shared,
            ))
    return tuple(results)


def axis_semantic(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """KF1.axis_semantic -- the axis is vertical, or horizontal, as declared.

    The only direction claim that needs no object frame. Gravity fixes the vertical, so
    ``|cos(axis, up)|`` settles it: near one for vertical, near zero for horizontal.
    Rotating the whole asset about the vertical does not change either, which is the point
    -- a correct cabinet authored ninety degrees round must not fail for it.

    Coarse on purpose, and it does not pretend otherwise. A drawer sliding sideways into
    the carcass wall instead of out through its opening is horizontal either way; that
    error is invisible here and belongs to a swept test.
    """
    results = []
    up = np.asarray(binding.asset.model.opt.gravity, dtype=float)
    up = -up / np.linalg.norm(up) if np.linalg.norm(up) > 1e-9 else np.array([0.0, 0.0, 1.0])
    tol_deg = contract.kinematic_claims.tolerances.axis_angle_deg

    _reference_pose(contract, binding)
    for joint in contract.kinematic_claims.joints:
        want = joint.axis.semantic
        if want is None:
            results.append(_na(
                "KF1.axis_semantic", joint.id,
                "no semantic axis declared; the relational and numeric forms are scored "
                "separately",
            ))
            continue
        _, k, problem = _resolve(binding, joint)
        if k is None:
            results.append(_na("KF1.axis_semantic", joint.id, problem or "unresolved"))
            continue

        a = np.asarray(binding.asset.data.xaxis[k], dtype=float)
        a = a / (np.linalg.norm(a) or 1.0)
        cos = abs(float(np.dot(a, up)))
        angle_to_up = math.degrees(math.acos(min(1.0, cos)))
        if want.value == "vertical":
            ok, wanted = cos >= math.cos(math.radians(tol_deg)), "parallel to up"
            off = angle_to_up
        else:
            ok, wanted = cos <= math.sin(math.radians(tol_deg)), "perpendicular to up"
            off = abs(90.0 - angle_to_up)

        shared = {
            "measured": {"angle_to_up_deg": round(angle_to_up, 3),
                         "deviation_deg": round(off, 3)},
            "threshold": {"axis_angle_deg": tol_deg},
            "evidence": {"axis_world": [round(float(v), 6) for v in a],
                         "up": [round(float(v), 6) for v in up]},
        }
        if ok:
            results.append(ClaimResult(
                "KF1.axis_semantic", joint.id, Verdict.PASS,
                f"axis is {want.value} ({wanted}), off by {off:.1f} deg", **shared,
            ))
        else:
            results.append(ClaimResult(
                "KF1.axis_semantic", joint.id, Verdict.FAIL,
                f"P0 declares a {want.value} axis, but the axis sits {angle_to_up:.1f} deg "
                f"from up, {off:.1f} deg outside the {tol_deg} deg tolerance", **shared,
            ))
    return tuple(results)


def axis_admits_motion(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """KF1.axis_admits_motion -- moving along the declared axis is geometrically possible.

    The finer half of the direction claim, and the only frame-free way to get at it. A
    drawer that slides sideways instead of outward is horizontal either way, so no static
    comparison against gravity can separate them; what separates them is that one direction
    leaves through the opening and the other drives straight into a wall.

    So the child is displaced a short distance along the declared travel direction and the
    signed distance to its declared parent is read. Correct: the gap does not collapse.
    Wrong axis: the part is already inside the parent's material.

    Slide joints only. For a hinge the equivalent question is about a swept arc rather than
    a direction, and that belongs to KF3 rather than being approximated here.
    """
    results = []
    tol = contract.kinematic_claims.tolerances.forbidden_penetration_m
    for joint in contract.kinematic_claims.joints:
        if joint.type is not JointType.SLIDE:
            results.append(_na(
                "KF1.axis_admits_motion", joint.id,
                f"only meaningful for a slide; {joint.type.value} travel is an arc and is "
                f"KF3's to sweep",
            ))
            continue
        child_body, k, problem = _resolve(binding, joint)
        if k is None:
            results.append(_na("KF1.axis_admits_motion", joint.id, problem or "unresolved"))
            continue
        try:
            parent_body = binding.root_body(joint.parent)
        except BindingError as exc:
            results.append(_na("KF1.axis_admits_motion", joint.id, str(exc)))
            continue

        asset = binding.asset
        step = min(0.25 * joint.range.span, 0.05)
        _reference_pose(contract, binding)
        before, _ = body_pair_distance(asset, parent_body, child_body, distmax=1.0)
        adr = int(asset.model.jnt_qposadr[k])
        asset.data.qpos[adr] += step
        mujoco.mj_forward(asset.model, asset.data)
        after, _ = body_pair_distance(asset, parent_body, child_body, distmax=1.0)
        _reference_pose(contract, binding)

        # Static-overlap gate. Asking whether moving a part in some direction causes
        # interference is meaningless when the part is already inside its parent at the
        # reference pose; every direction "causes" interference then. That overlap is a
        # real defect, but it is static geometry's to report, not this predicate's.
        if before < -tol:
            results.append(_na(
                "KF1.axis_admits_motion", joint.id,
                f"{joint.part!r} already overlaps {joint.parent!r} by {-before:.4f} m at the "
                f"reference pose, so no direction can be judged clear; that overlap is a "
                f"static-geometry finding",
                clearance_before=round(before, 6),
            ))
            _reference_pose(contract, binding)
            continue

        gained = before - after
        shared = {
            "measured": {"clearance_before": round(before, 6),
                         "clearance_after": round(after, 6),
                         "penetration_gained": round(max(0.0, gained), 6),
                         "step_m": round(step, 6)},
            "threshold": {"forbidden_penetration_m": tol},
            "evidence": {"parent": joint.parent, "part": joint.part},
        }
        if after >= -tol:
            results.append(ClaimResult(
                "KF1.axis_admits_motion", joint.id, Verdict.PASS,
                f"moving {step:.3f} m along the declared axis keeps {joint.part!r} clear of "
                f"{joint.parent!r}", **shared,
            ))
        else:
            results.append(ClaimResult(
                "KF1.axis_admits_motion", joint.id, Verdict.FAIL,
                f"moving {step:.3f} m along the declared axis drives {joint.part!r} "
                f"{-after:.4f} m into {joint.parent!r}; the declared direction is not the "
                f"one the geometry allows", **shared,
            ))
    return tuple(results)


def anchor(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """KF1.anchor -- the line of rotation sits where P0 says relative to the part.

    Stated as a geometric relation rather than a coordinate, because P0 is authored before
    the asset exists and cannot know how large the door will be or where it will sit.
    ``envelope_m`` gives a size range, not a geometry, and a hard-coded ``[0.4, 0, 0]`` is
    only checkable if the door happens to land where the author guessed.

    Two forms, and they are opposites:

    ``on_edge_of``
        The part's vertices must lie on one side of a plane through the axis. A door
        hinged on its edge scores 1.0; hinged through its middle, 0.5.
    ``through_center_of``
        The axis must pass near the part's centroid, normalised by the part's own size. A
        gear on its shaft scores near 0; mounted off-centre it does not.

    Scoring a gear with ``on_edge_of`` would fail it by construction -- every plane through
    a disc's symmetry axis halves it -- which is why one form cannot serve both.

    N/A for slide and free, where the anchor does not enter the kinematics: displacing
    ``pos`` moves the child by exactly zero.
    """
    results = []
    tolerances = contract.kinematic_claims.tolerances
    _reference_pose(contract, binding)

    for joint in contract.kinematic_claims.joints:
        if not joint.type.has_anchor or joint.anchor is None:
            results.append(_na(
                "KF1.anchor", joint.id,
                f"a {joint.type.value} joint has no pivot the kinematics depend on"
                if not joint.type.has_anchor else "no anchor claim declared",
            ))
            continue
        _, k, problem = _resolve(binding, joint)
        if k is None:
            results.append(_na("KF1.anchor", joint.id, problem or "unresolved"))
            continue

        target = joint.anchor.on_edge_of or joint.anchor.through_center_of
        try:
            bodies = (binding.root_body(target),)
        except BindingError as exc:
            results.append(_na("KF1.anchor", joint.id, str(exc)))
            continue

        asset = binding.asset
        points = _points_of(binding, bodies)
        if len(points) == 0:
            results.append(_na("KF1.anchor", joint.id, f"{target!r} has no geometry to measure"))
            continue

        p0 = np.asarray(asset.data.xanchor[k], dtype=float)
        a = np.asarray(asset.data.xaxis[k], dtype=float)
        a = a / (np.linalg.norm(a) or 1.0)
        perp = lambda v: v - np.dot(v, a) * a  # noqa: E731
        radial = np.array([perp(p - p0) for p in points])

        if joint.anchor.on_edge_of:
            # The direction to split along. The centroid offset is the natural choice
            # and is well conditioned for a door hinged on its edge. When the axis runs
            # through the part's centroid that offset vanishes -- which is precisely the
            # defect this predicate exists to catch, not a reason to abstain -- so fall
            # back to the direction the part spreads furthest in, which always exists and
            # gives the honest 50/50 split.
            centre = perp(points.mean(axis=0) - p0)
            norm = float(np.linalg.norm(centre))
            if norm > 1e-9:
                direction = centre / norm
            else:
                _, _, vh = np.linalg.svd(radial - radial.mean(axis=0), full_matrices=False)
                direction = vh[0] - np.dot(vh[0], a) * a
                direction = direction / (np.linalg.norm(direction) or 1.0)
            s = radial @ direction
            positive = int((s > 1e-6).sum())
            negative = int((s < -1e-6).sum())
            on_line = len(s) - positive - negative
            fraction = max(positive + on_line, negative + on_line) / len(s)
            floor = tolerances.anchor_side_fraction_min
            shared = {
                "measured": {"side_fraction": round(fraction, 4),
                             "positive": positive, "negative": negative,
                             "on_axis": on_line},
                "threshold": {"anchor_side_fraction_min": floor},
                "evidence": {"part": target, "anchor_world": [round(float(v), 6) for v in p0]},
            }
            if fraction >= floor:
                results.append(ClaimResult(
                    "KF1.anchor", joint.id, Verdict.PASS,
                    f"{fraction:.0%} of {target!r} lies on one side of the axis, so the line "
                    f"runs along its edge", **shared,
                ))
            else:
                results.append(ClaimResult(
                    "KF1.anchor", joint.id, Verdict.FAIL,
                    f"the axis splits {target!r} {positive}/{negative}; only {fraction:.0%} "
                    f"lies on one side, so it turns about its middle rather than its edge",
                    **shared,
                ))
        else:
            lo, hi = subtree_aabb(asset, bodies)
            size = aabb_diagonal(lo, hi) or 1.0
            offset = float(np.linalg.norm(perp(points.mean(axis=0) - p0)))
            ratio = offset / size
            ceiling = tolerances.anchor_center_offset_max
            shared = {
                "measured": {"offset_over_diagonal": round(ratio, 4),
                             "offset_m": round(offset, 6), "part_diagonal_m": round(size, 6)},
                "threshold": {"anchor_center_offset_max": ceiling},
                "evidence": {"part": target, "anchor_world": [round(float(v), 6) for v in p0]},
            }
            if ratio <= ceiling:
                results.append(ClaimResult(
                    "KF1.anchor", joint.id, Verdict.PASS,
                    f"the axis passes within {ratio:.1%} of {target!r}'s own size of its "
                    f"centroid", **shared,
                ))
            else:
                results.append(ClaimResult(
                    "KF1.anchor", joint.id, Verdict.FAIL,
                    f"the axis misses {target!r}'s centroid by {ratio:.1%} of the part's own "
                    f"size, past the {ceiling:.0%} allowed", **shared,
                ))
    return tuple(results)


def range_and_reference(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """KF1.range_and_reference -- travel and the named states, scored once.

    One claim, not four. ``states.closed``, the reference configuration and ``range.min``
    are typically the same declared number, and the specification lists them as separate
    things to verify. Counting one number three times inflates the denominator with claims
    that cannot vary independently, which is the double counting P0's own core rule forbids.

    A ``continuous`` joint arrives as a hinge with ``jnt_limited`` false and no meaningful
    range, so the travel half is N/A while the named states are still checked.
    """
    results = []
    for joint in contract.kinematic_claims.joints:
        _, k, problem = _resolve(binding, joint)
        if k is None:
            results.append(_na("KF1.range_and_reference", joint.id, problem or "unresolved"))
            continue
        model = binding.asset.model
        if not model.jnt_limited[k]:
            results.append(_na(
                "KF1.range_and_reference", joint.id,
                "the joint is unlimited (URDF 'continuous'), so there is no declared travel "
                "to compare against",
            ))
            continue

        lo, hi = (float(v) for v in model.jnt_range[k])
        span = hi - lo
        want = joint.range.span
        relative = abs(span - want) / (want or 1.0)
        floor = contract.kinematic_claims.tolerances.state_match_relative
        shared = {
            "measured": {"model_span": round(span, 6), "model_range": [round(lo, 6), round(hi, 6)]},
            "threshold": {"declared_span": round(want, 6),
                          "state_match_relative": floor},
            "evidence": {"unit": joint.range.unit,
                         "declared_states": dict(joint.states)},
        }
        if relative <= floor:
            results.append(ClaimResult(
                "KF1.range_and_reference", joint.id, Verdict.PASS,
                f"travel {span:.4g} {joint.range.unit} matches the declared "
                f"{want:.4g}", **shared,
            ))
        else:
            results.append(ClaimResult(
                "KF1.range_and_reference", joint.id, Verdict.FAIL,
                f"P0 declares {want:.4g} {joint.range.unit} of travel and the model gives "
                f"{span:.4g}, off by {relative:.0%}", **shared,
            ))
    return tuple(results)


def rigid_follower(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """KF1.rigid_follower -- a part declared to ride another really is fixed to it.

    Structural, on the body tree: the follower must be reachable from its leader without
    crossing a joint. That is what "rigidly attached" means operationally, and it is the
    only formulation that catches the case worth catching -- a handle bolted to the carcass
    instead of the drawer renders identically at the reference pose and only reveals itself
    when the drawer opens. A check evaluated at one configuration cannot see it; a graph
    walk can, at no cost.
    """
    results = []
    for attachment in contract.kinematic_claims.rigid_attachments:
        subject = f"{attachment.follower}<-{attachment.leader}"
        try:
            follower = binding.root_body(attachment.follower)
            leader = binding.root_body(attachment.leader)
        except BindingError as exc:
            results.append(_na("KF1.rigid_follower", subject, str(exc)))
            continue

        rigid = binding.asset.rigid_subtree(leader)
        names = [binding.asset.body_name(b) for b in rigid]
        observed, _ = binding.nearest_declared_ancestor(follower)
        shared = {
            "measured": {"in_leader_rigid_subtree": follower in rigid,
                         "nearest_declared_ancestor": observed},
            "evidence": {"leader_rigid_subtree": names,
                         "follower_body": binding.asset.body_name(follower)},
        }
        if follower in rigid:
            results.append(ClaimResult(
                "KF1.rigid_follower", subject, Verdict.PASS,
                f"{attachment.follower!r} is fixed to {attachment.leader!r} and moves with it",
                **shared,
            ))
        else:
            results.append(ClaimResult(
                "KF1.rigid_follower", subject, Verdict.FAIL,
                f"{attachment.follower!r} is not rigidly attached to {attachment.leader!r} "
                f"-- it hangs off {observed!r} instead, so it stays behind when "
                f"{attachment.leader!r} moves", **shared,
            ))
    return tuple(results)


def travel_scale(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """KF1.travel_scale -- the body that moves is big enough to be the part it claims to be.

    This is the admissible form of the specification's "no changing qpos with a visually
    stationary part". That bullet names the plainest kinematic failure there is -- you open
    it and nothing moves -- but "visually" wants a renderer, which P3's own scope forbids.

    The failure it is aimed at is a body carrying the declared joint and a correct range
    while owning almost no geometry, with the drawer a viewer sees belonging to the static
    body. Displacement alone does not catch it: the decoy stub travels the full declared
    distance. What catches it is size. A part that slides 0.18 m and is 8 mm across is not
    the drawer.

    Normalised by the joint's own declared travel, never by a whole-asset diagonal -- only
    19 of 607 corpus assets could produce one of those.

    Slide joints only. A hinge's range is an angle and has no length to compare against,
    and inventing one out of a lever arm reduces to comparing the part to itself.
    """
    results = []
    floor = contract.kinematic_claims.tolerances.travel_scale_min
    _reference_pose(contract, binding)

    for joint in contract.kinematic_claims.joints:
        if joint.type is not JointType.SLIDE:
            results.append(_na(
                "KF1.travel_scale", joint.id,
                f"a {joint.type.value} range is an angle, with no length to compare a size "
                f"against",
            ))
            continue
        try:
            child = binding.root_body(joint.part)
        except BindingError as exc:
            results.append(_na("KF1.travel_scale", joint.id, str(exc)))
            continue

        # The part's own bodies, not everything riding it. A decoy stub with a real
        # handle bolted to it has a rigid subtree 0.39 m across while the part the
        # contract names is 8 mm; the claim is about the part.
        bodies = binding.bodies(joint.part)
        lo, hi = subtree_aabb(binding.asset, bodies)
        size = aabb_diagonal(lo, hi)
        travel = joint.range.span
        ratio = size / (travel or 1.0)
        shared = {
            "measured": {"moving_geometry_diagonal_m": round(size, 6),
                         "size_over_travel": round(ratio, 4)},
            "threshold": {"travel_scale_min": floor, "declared_travel_m": round(travel, 6)},
            "evidence": {"measured_bodies": [binding.asset.body_name(b) for b in bodies]},
        }
        if ratio >= floor:
            results.append(ClaimResult(
                "KF1.travel_scale", joint.id, Verdict.PASS,
                f"the geometry that moves is {size:.3f} m across against {travel:.3f} m of "
                f"declared travel", **shared,
            ))
        else:
            results.append(ClaimResult(
                "KF1.travel_scale", joint.id, Verdict.FAIL,
                f"the body driven by {joint.id!r} is only {size:.4f} m across while "
                f"declaring {travel:.3f} m of travel; whatever a viewer would call "
                f"{joint.part!r} is not what moves", **shared,
            ))
    return tuple(results)


PREDICATES = (
    parent,
    joint_type,
    dof_composition,
    axis_semantic,
    axis_admits_motion,
    anchor,
    range_and_reference,
    rigid_follower,
    travel_scale,
)
"""Every KF1 predicate, in report order."""

DECLARATION_READS = frozenset({"KF1.type", "KF1.range_and_reference"})
"""Predicates decided by reading a field back rather than by measuring the model.

Reported as their own sub-score. In the previous corpus this class matched in nearly every
asset, so folding it in with the measured predicates buries the ones that vary.
"""


def evaluate(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    results: list[ClaimResult] = []
    for predicate in PREDICATES:
        results.extend(predicate(contract, binding))
    return tuple(results)
