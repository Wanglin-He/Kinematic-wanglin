"""Where to look, in a way that is affordable and the same every time.

A full grid is out. Three independent drawers at 33 points each is 35,937 configurations,
and the count is exponential in the degrees of freedom -- six of them would be 1.3 billion.
Sweeping one joint at a time is affordable and blind to every interaction: two drawers that
collide only when both are open are individually clean.

So four layers, each buying something the others cannot:

1. **Per-DOF scan.** Every joint across its declared range with the others at rest. Finds
   anything a single motion runs into, which is most of it.
2. **Pair grid.** A coarse grid over pairs of joints, but only pairs whose swept volumes
   could reach each other. The adjacency test is what keeps this affordable: joints on
   opposite ends of an asset never need to be swept together, and a cheap conservative
   bound proves it.
3. **Declared states.** Every combination of the named semantic states. These are the
   configurations a human actually names, so they are visited exactly rather than
   approximately.
4. **Space-filling fill.** A low-discrepancy sequence over the whole cube, to catch what a
   structured schedule would step over.

Cost grows linearly in the joint count instead of exponentially: the gold cabinet's three
joints come to a few hundred configurations rather than 35,937.

Everything is deterministic. There is no seed to set because nothing is drawn -- the fill
is a Halton sequence, whose k-th point is a fixed function of k. A verifier whose answer
moves between runs cannot be argued with: nobody could separate a real regression from
noise, which is the same reason a model must not decide anything here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

import mujoco
import numpy as np

from evo_p0p3.p0.schema import Contract
from evo_p0p3.p3.binding import Binding, BindingError
from evo_p0p3.p3.mjcf import subtree_aabb

SCAN_POINTS = 33
"""Samples per joint in layer 1. Odd and endpoint-inclusive, so both declared limits and
the midpoint all land exactly on a sample."""

PAIR_POINTS = 9
"""Grid resolution per joint in layer 2. Coarse on purpose: the layer exists to find
whether two motions can reach each other at all, not to localise the contact."""

FILL_PER_DOF = 256
"""Layer 4 points per degree of freedom."""

STATE_COMBINATION_CAP = 12
"""Above this many joints, 2^d named-state combinations stop being affordable and only the
all-same combinations are visited. Reported rather than silently truncated."""

INFLATION = 0.05
"""How much to grow each swept bound before testing two joints for adjacency, as a
fraction of the larger bound's diagonal. Errs toward sweeping a pair that could not have
touched, because the opposite error is missing the interference the layer exists to find."""

_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53)


@dataclass(frozen=True, slots=True)
class Sample:
    """One configuration, and which layer asked for it."""

    qpos: np.ndarray
    layer: str
    label: str
    """Human-readable, e.g. ``drawer_1_slide=0.180`` or ``closed``."""


@dataclass(frozen=True, slots=True)
class Schedule:
    """Every configuration to visit, plus what it cost and what was skipped."""

    samples: tuple[Sample, ...]
    driven: tuple[str, ...]
    dependent: tuple[str, ...]
    pairs_swept: tuple[tuple[str, str], ...]
    pairs_skipped: tuple[tuple[str, str], ...]
    notes: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return len(self.samples)

    def by_layer(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.samples:
            counts[s.layer] = counts.get(s.layer, 0) + 1
        return counts

    @property
    def provenance(self) -> dict[str, object]:
        """What a report must carry so the coverage is auditable rather than assumed."""
        return {
            "samples": self.size,
            "by_layer": self.by_layer(),
            "driven_dofs": list(self.driven),
            "dependent_dofs": list(self.dependent),
            "pairs_swept": [list(p) for p in self.pairs_swept],
            "pairs_skipped": [list(p) for p in self.pairs_skipped],
            "scan_points": SCAN_POINTS,
            "pair_points": PAIR_POINTS,
            "fill_per_dof": FILL_PER_DOF,
            "notes": list(self.notes),
        }


def _halton(index: int, base: int) -> float:
    """The index-th point of the van der Corput sequence in ``base``.

    A fixed function of its arguments, so the fill is reproducible without a seed. Halton
    correlates badly in high dimensions, which is why the fill is a supplement to the
    structured layers rather than the whole schedule.
    """
    f, result = 1.0, 0.0
    while index > 0:
        f /= base
        result += f * (index % base)
        index //= base
    return result


class Sweeper:
    """Builds and walks a schedule for one asset under one contract."""

    def __init__(self, contract: Contract, binding: Binding) -> None:
        self.contract = contract
        self.binding = binding
        self.asset = binding.asset
        self.model = binding.asset.model
        self.data = binding.asset.data
        self._joints = self._resolve_joints()

    # -- resolution --------------------------------------------------------------------

    def _resolve_joints(self) -> dict[str, int]:
        out = {}
        for joint in self.contract.kinematic_claims.joints:
            try:
                body = self.binding.root_body(joint.part)
            except BindingError:
                continue
            start, count = int(self.model.body_jntadr[body]), int(self.model.body_jntnum[body])
            if count == 1:
                out[joint.id] = start
        return out

    def _bounds(self, joint_id: str) -> tuple[float, float]:
        """The interval to sweep.

        The contract's declared range, not the model's. A joint whose model range is wrong
        is KF1's finding, and sweeping the model's range instead would quietly narrow the
        search to whatever the asset happened to allow -- so an asset that under-declares
        its travel would be swept less thoroughly for it.
        """
        joint = self.contract.kinematic_claims.joint(joint_id)
        return joint.range.min, joint.range.max

    def _reference(self) -> np.ndarray:
        q = np.array(self.model.qpos0, copy=True)
        for joint in self.contract.kinematic_claims.joints:
            k = self._joints.get(joint.id)
            if k is None:
                continue
            value = joint.states.get("closed", joint.states.get("home", joint.range.min))
            q[int(self.model.jnt_qposadr[k])] = value
        return q

    def _apply_couplings(self, q: np.ndarray) -> np.ndarray:
        """Set every dependent joint from its declared relation.

        The contract's relation, not the model's constraint. That is the point of the
        sweep: the model is placed where P0 says it should be, and whether its own
        mechanism agrees there is KF2's residual.
        """
        for coupling in self.contract.kinematic_claims.couplings:
            dep = self._joints.get(coupling.relation.dependent)
            ind = self._joints.get(coupling.relation.independent)
            if dep is None or ind is None:
                continue
            value = q[int(self.model.jnt_qposadr[ind])]
            q[int(self.model.jnt_qposadr[dep])] = (
                coupling.relation.coefficient * value + coupling.relation.offset
            )
        return q

    # -- layers ------------------------------------------------------------------------

    def _driven(self) -> tuple[str, ...]:
        declared = tuple(self.contract.kinematic_claims.independent_dofs)
        return tuple(j for j in declared if j in self._joints)

    def _dependent(self) -> tuple[str, ...]:
        return tuple(
            c.relation.dependent
            for c in self.contract.kinematic_claims.couplings
            if c.relation.dependent in self._joints
        )

    def _set(self, q: np.ndarray, joint_id: str, value: float) -> np.ndarray:
        q = np.array(q, copy=True)
        q[int(self.model.jnt_qposadr[self._joints[joint_id]])] = value
        return self._apply_couplings(q)

    def _layer1(self, reference: np.ndarray) -> list[Sample]:
        out = []
        for joint_id in self._driven():
            lo, hi = self._bounds(joint_id)
            for value in np.linspace(lo, hi, SCAN_POINTS):
                out.append(Sample(
                    qpos=self._set(reference, joint_id, float(value)),
                    layer="scan",
                    label=f"{joint_id}={value:+.4g}",
                ))
        return out

    def _swept_bound(self, joint_id: str, reference: np.ndarray):
        """A conservative world bound on everything this joint moves, over its whole range.

        The union of the moved subtree's bounds at the ends and middle of the range, taken
        with MuJoCo's bounding-sphere radii. Over-stating is the safe direction: it can
        only cause a pair to be swept that need not have been.
        """
        joint = next(
            j for j in self.contract.kinematic_claims.joints if j.id == joint_id
        )
        body = self.binding.root_body(joint.part)
        bodies = self.asset.subtree_bodies(body)
        lo, hi = self._bounds(joint_id)
        low = np.full(3, np.inf)
        high = np.full(3, -np.inf)
        for value in (lo, 0.5 * (lo + hi), hi):
            self.data.qpos[:] = self._set(reference, joint_id, float(value))
            mujoco.mj_forward(self.model, self.data)
            a, b = subtree_aabb(self.asset, bodies, conservative=True)
            low, high = np.minimum(low, a), np.maximum(high, b)
        return low, high

    def _layer2(self, reference: np.ndarray):
        driven = self._driven()
        bounds = {j: self._swept_bound(j, reference) for j in driven}
        out, swept, skipped = [], [], []
        for a, b in combinations(driven, 2):
            (alo, ahi), (blo, bhi) = bounds[a], bounds[b]
            pad = INFLATION * max(
                float(np.linalg.norm(ahi - alo)), float(np.linalg.norm(bhi - blo))
            )
            if np.any(ahi + pad < blo) or np.any(bhi + pad < alo):
                skipped.append((a, b))
                continue
            swept.append((a, b))
            alo_v, ahi_v = self._bounds(a)
            blo_v, bhi_v = self._bounds(b)
            for va in np.linspace(alo_v, ahi_v, PAIR_POINTS):
                for vb in np.linspace(blo_v, bhi_v, PAIR_POINTS):
                    q = self._set(reference, a, float(va))
                    q = self._set(q, b, float(vb))
                    out.append(Sample(
                        qpos=q, layer="pair",
                        label=f"{a}={va:+.3g},{b}={vb:+.3g}",
                    ))
        return out, tuple(swept), tuple(skipped)

    def _layer3(self, reference: np.ndarray) -> tuple[list[Sample], tuple[str, ...]]:
        driven = self._driven()
        labels = sorted({
            label
            for j in driven
            for label in (self.contract.kinematic_claims.joint(j).states or {})
        })
        if not labels:
            return [], ()
        notes: tuple[str, ...] = ()
        if len(driven) > STATE_COMBINATION_CAP:
            combos = [(label,) * len(driven) for label in labels]
            notes = (
                f"{len(driven)} driven joints exceeds the {STATE_COMBINATION_CAP} cap, so "
                f"only the uniform state combinations were visited, not all "
                f"{len(labels)}^{len(driven)}",
            )
        else:
            combos = list(product(labels, repeat=len(driven)))

        out = []
        for combo in combos:
            q = np.array(reference, copy=True)
            usable = True
            for joint_id, label in zip(driven, combo, strict=True):
                states = self.contract.kinematic_claims.joint(joint_id).states or {}
                if label not in states:
                    usable = False
                    break
                q = self._set(q, joint_id, float(states[label]))
            if usable:
                out.append(Sample(qpos=q, layer="states", label=",".join(combo)))
        return out, notes

    def _layer4(self, reference: np.ndarray) -> list[Sample]:
        driven = self._driven()
        if not driven:
            return []
        count = FILL_PER_DOF * len(driven)
        out = []
        for i in range(1, count + 1):
            q = np.array(reference, copy=True)
            for axis, joint_id in enumerate(driven):
                lo, hi = self._bounds(joint_id)
                t = _halton(i, _PRIMES[axis % len(_PRIMES)])
                q = self._set(q, joint_id, float(lo + t * (hi - lo)))
            out.append(Sample(qpos=q, layer="fill", label=f"halton[{i}]"))
        return out

    # -- assembly ----------------------------------------------------------------------

    def schedule(self) -> Schedule:
        reference = self._reference()
        samples = [Sample(qpos=reference, layer="reference", label="reference")]
        samples += self._layer1(reference)
        pair_samples, swept, skipped = self._layer2(reference)
        samples += pair_samples
        state_samples, notes = self._layer3(reference)
        samples += state_samples
        samples += self._layer4(reference)

        if skipped:
            notes = notes + (
                f"{len(skipped)} joint pair(s) were not swept together because their "
                f"swept bounds cannot reach each other: "
                f"{', '.join(f'{a}+{b}' for a, b in skipped)}",
            )
        return Schedule(
            samples=tuple(samples),
            driven=self._driven(),
            dependent=self._dependent(),
            pairs_swept=swept,
            pairs_skipped=skipped,
            notes=notes,
        )

    def visit(self, sample: Sample) -> None:
        """Place the model at one configuration. Position level only, no stepping."""
        self.data.qpos[:] = sample.qpos
        mujoco.mj_forward(self.model, self.data)
