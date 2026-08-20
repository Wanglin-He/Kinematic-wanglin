"""KF1 -- articulation specification fidelity.

Does the model implement the motion chain and joint configuration P0 froze? Predicates are
added one at a time, each landing together with the gold-standard asset that makes it fail.
That pairing is the point rather than the ceremony: a predicate no input can fail passes
everything, and against a corpus of mostly-correct assets that looks identical to a corpus
of correct assets. This project has already been caught by that twice -- a fault injection
displacing a joint origin by a whole link diagonal detected nothing, because the origin IS
the child frame origin and moving it carries the geometry along.

So each predicate here has, in the same commit, an asset whose answer is known because we
broke it ourselves.
"""

from __future__ import annotations

from evo_p0p3.p0.schema import Contract
from evo_p0p3.p3.binding import Binding, BindingError
from evo_p0p3.p3.verdict import ClaimResult, Verdict


def parent(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    """KF1.parent -- each joint moves its part relative to the part P0 named.

    Decided on the **nearest declared ancestor**, not on ancestry alone. A lower drawer
    hung off an upper drawer still has the carcass somewhere up its chain, so "is the
    declared parent an ancestor?" passes an asset where opening the upper drawer drags the
    lower one out with it. The claim is about what the part moves *relative to*, and that
    is the first declared part above it.

    Bodies the contract never declared are transparent. Generators insert unnamed
    intermediate links freely, and treating one as a parent would fail correct assets for
    an authoring style the contract says nothing about.

    Pure graph work on ``body_parentid``: no geometry, no tolerance, no threshold.
    """
    results = []
    for joint in contract.kinematic_claims.joints:
        subject = joint.id
        try:
            child_body = binding.root_body(joint.part)
        except BindingError as exc:
            results.append(
                ClaimResult(
                    predicate="KF1.parent",
                    subject=subject,
                    verdict=Verdict.NA,
                    reason=str(exc),
                    evidence={"part": joint.part},
                )
            )
            continue

        observed, steps = binding.nearest_declared_ancestor(child_body)
        model = binding.asset.model
        chain = []
        walker = int(model.body_parentid[child_body])
        for _ in range(steps):
            chain.append(binding.asset.body_name(walker))
            if walker == 0:
                break
            walker = int(model.body_parentid[walker])

        common = {
            "measured": {"nearest_declared_ancestor": observed, "links_up": steps},
            "evidence": {
                "part": joint.part,
                "child_body": binding.asset.body_name(child_body),
                "body_chain_upward": chain,
                "binding_source": binding.source.value,
            },
        }

        if observed is None:
            results.append(
                ClaimResult(
                    predicate="KF1.parent",
                    subject=subject,
                    verdict=Verdict.FAIL,
                    reason=(
                        f"{joint.part!r} hangs off no declared part at all; P0 says its "
                        f"parent is {joint.parent!r}"
                    ),
                    **common,
                )
            )
        elif observed == joint.parent:
            results.append(
                ClaimResult(
                    predicate="KF1.parent",
                    subject=subject,
                    verdict=Verdict.PASS,
                    reason=f"{joint.part!r} moves relative to {joint.parent!r} as declared",
                    **common,
                )
            )
        else:
            results.append(
                ClaimResult(
                    predicate="KF1.parent",
                    subject=subject,
                    verdict=Verdict.FAIL,
                    reason=(
                        f"{joint.part!r} moves relative to {observed!r}, but P0 declares "
                        f"{joint.parent!r}; driving {observed!r} therefore carries "
                        f"{joint.part!r} with it"
                    ),
                    **common,
                )
            )
    return tuple(results)


PREDICATES = (parent,)
"""Every KF1 predicate, in report order. Grows one entry per commit."""


def evaluate(contract: Contract, binding: Binding) -> tuple[ClaimResult, ...]:
    results: list[ClaimResult] = []
    for predicate in PREDICATES:
        results.extend(predicate(contract, binding))
    return tuple(results)
