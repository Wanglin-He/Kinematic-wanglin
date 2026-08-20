"""KF1, judged entirely by assets whose answers we built in.

The manifest carries a full verdict for every predicate on every defective asset -- 70
expectations across eight assets -- and this file asserts the whole matrix. Stating only
the verdict a defect is *aimed* at would leave the interesting half unchecked: whether the
other eight predicates stayed quiet. A check that fires on everything broken has not
diagnosed a wrong parent, it has noticed that something is different, and a report built
on it cannot tell anyone what to fix.

Two defects legitimately break two claims each, and the manifest says so rather than
massaging the assets until each breaks exactly one. A revolute turned prismatic changes
the unit its range is measured in; a hinge axis turned horizontal stops running along any
edge. Forcing isolation there would mean authoring absurd assets to flatter the tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evo_p0p3.p0.loader import parse_contract
from evo_p0p3.p3 import binding as binding_mod
from evo_p0p3.p3 import gold, kf1, mjcf
from evo_p0p3.p3.verdict import Verdict, score

ROOT = Path(__file__).resolve().parents[1]
CABINET = gold.defects("cabinet_correct.urdf")
"""Only the cabinet family. The gearbox exists for the coupling claims and is judged by
tests/test_kf2.py against its own contract; running it through the cabinet's contract
would ask KF1 about parts that asset does not have."""


@pytest.fixture(scope="module")
def contract():
    raw = yaml.safe_load((ROOT / "contracts" / "gold_cabinet.yaml").read_text(encoding="utf-8"))
    return parse_contract(raw, record_id="gold_cabinet")


@pytest.fixture(scope="module")
def materialised(tmp_path_factory) -> dict[str, Path]:
    return gold.write_all(tmp_path_factory.mktemp("kf1"))


def verdicts_by_predicate(name, materialised, contract) -> dict[str, set[Verdict]]:
    asset = mjcf.load(materialised[name], record_id=name)
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    out: dict[str, set[Verdict]] = {}
    for result in kf1.evaluate(contract, bound):
        out.setdefault(result.predicate, set()).add(result.verdict)
    return out


def collapse(verdicts: set[Verdict]) -> str:
    if Verdict.FAIL in verdicts:
        return "fail"
    if verdicts == {Verdict.NA}:
        return "na"
    return "pass"


# --------------------------------------------------------------------------------------
# the control
# --------------------------------------------------------------------------------------


def test_the_control_passes_every_predicate(materialised, contract):
    verdicts = verdicts_by_predicate("cabinet_correct", materialised, contract)
    failed = {p: v for p, v in verdicts.items() if Verdict.FAIL in v}
    assert not failed, failed


def test_the_control_scores_one(materialised, contract):
    asset = mjcf.load(materialised["cabinet_correct"], record_id="control")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    assert score(kf1.evaluate(contract, bound)) == 1.0


def test_the_control_leaves_nothing_unevaluated_that_should_be_evaluated(
    materialised, contract
):
    # Every predicate must reach a real verdict on at least one subject of the control.
    # A predicate that is N/A everywhere has not been exercised at all, and its first real
    # use would be against an asset whose answer nobody knows.
    verdicts = verdicts_by_predicate("cabinet_correct", materialised, contract)
    dead = [p for p, v in verdicts.items() if v == {Verdict.NA}]
    assert not dead, f"never evaluated on the control: {dead}"


# --------------------------------------------------------------------------------------
# the full matrix
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("defect", CABINET, ids=lambda d: d.name)
def test_the_verdict_matrix_matches_the_manifest(defect, materialised, contract):
    observed = {
        p: collapse(v)
        for p, v in verdicts_by_predicate(defect.name, materialised, contract).items()
    }
    mismatches = {
        predicate: (expected, observed.get(predicate))
        for predicate, expected in defect.expect.items()
        if predicate.startswith("KF1.") and observed.get(predicate) != expected
    }
    assert not mismatches, f"expected vs observed: {mismatches}"


@pytest.mark.parametrize(
    "defect", [d for d in CABINET if d.targets.startswith("KF1.")], ids=lambda d: d.name
)
def test_every_defect_kf1_owns_is_detected_by_kf1(defect, materialised, contract):
    # Only the defects KF1 is responsible for. swept_interference is KF3's, and expecting
    # KF1 to notice it would be expecting the wrong metric to do another's job.
    verdicts = verdicts_by_predicate(defect.name, materialised, contract)
    assert any(Verdict.FAIL in v for v in verdicts.values()), (
        f"{defect.name} passes every KF1 predicate, so the defect it carries is invisible"
    )


@pytest.mark.parametrize("defect", CABINET, ids=lambda d: d.name)
def test_the_manifest_covers_every_predicate(defect):
    # An expectation table with holes lets a predicate drift into firing on an asset it
    # has nothing to do with, without any test noticing.
    named = {f"KF1.{p.__name__}" for p in kf1.PREDICATES}
    named = {"KF1.type" if p == "KF1.joint_type" else p for p in named}
    assert named <= set(defect.expect), named - set(defect.expect)


@pytest.mark.parametrize("defect", CABINET, ids=lambda d: d.name)
def test_each_defect_is_caught_by_the_predicate_it_targets(defect, materialised, contract):
    verdicts = verdicts_by_predicate(defect.name, materialised, contract)
    targeted = [p for p, v in defect.expect.items()
                if v == "fail" and p.startswith("KF1.")]
    for predicate in targeted:
        assert Verdict.FAIL in verdicts.get(predicate, set()), (
            f"{defect.name} was built to break {predicate} and it did not fire"
        )


# --------------------------------------------------------------------------------------
# properties of individual predicates
# --------------------------------------------------------------------------------------


def test_parent_uses_the_nearest_declared_ancestor_not_mere_ancestry(materialised, contract):
    # In wrong_parent the carcass is still an ancestor of drawer_2, just not the nearest
    # declared one, so a check asking "is the declared parent somewhere above?" passes it.
    asset = mjcf.load(materialised["wrong_parent"], record_id="wrong_parent")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    drawer_2 = bound.root_body("drawer_2")
    assert bound.nearest_declared_ancestor(drawer_2)[0] == "drawer_1"

    chain, body = [], drawer_2
    while body != 0:
        body = int(asset.model.body_parentid[body])
        chain.append(asset.body_name(body))
    assert "cabinet_body" in chain


def test_anchor_does_not_abstain_when_the_axis_runs_through_the_centroid(
    materialised, contract
):
    # The degenerate case is the defect, not a reason to abstain: an axis through the
    # door's centroid is precisely a hinge through its middle.
    asset = mjcf.load(materialised["hinge_through_middle"], record_id="middle")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    result = next(r for r in kf1.anchor(contract, bound) if r.subject == "door_hinge")
    assert result.verdict is Verdict.FAIL
    assert result.measured["side_fraction"] == pytest.approx(0.5, abs=0.01)


def test_anchor_measures_the_control_as_entirely_on_one_side(materialised, contract):
    asset = mjcf.load(materialised["cabinet_correct"], record_id="control")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    result = next(r for r in kf1.anchor(contract, bound) if r.subject == "door_hinge")
    assert result.measured["side_fraction"] == pytest.approx(1.0)


def test_travel_scale_measures_the_part_not_everything_riding_it(materialised, contract):
    # The decoy stub carries a real 0.16 m handle, so its rigid subtree is 0.39 m across
    # while the part the contract names is 8 mm. The claim is about the part.
    asset = mjcf.load(materialised["fake_joint_decoy_geom"], record_id="fake")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    result = next(r for r in kf1.travel_scale(contract, bound) if r.subject == "drawer_1_slide")
    assert result.verdict is Verdict.FAIL
    assert result.measured["moving_geometry_diagonal_m"] < 0.02
    assert result.evidence["measured_bodies"] == ["drawer_1"]


def test_axis_admits_motion_abstains_on_a_part_already_overlapping_its_parent(
    materialised, contract
):
    # Asking whether a direction causes interference is meaningless for a part already
    # inside its parent; every direction "causes" it. That overlap is a real defect but a
    # static-geometry one, and charging it here would blame the wrong claim.
    asset = mjcf.load(materialised["fake_joint_decoy_geom"], record_id="fake")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    result = next(
        r for r in kf1.axis_admits_motion(contract, bound) if r.subject == "drawer_1_slide"
    )
    assert result.verdict is Verdict.NA
    assert "static-geometry" in result.reason


def test_axis_semantic_cannot_see_a_sideways_drawer_and_says_so(materialised, contract):
    # Sliding into the wall is exactly as horizontal as sliding out of the opening. The
    # predicate abstaining from a claim it cannot make is the point; axis_admits_motion
    # is what catches this.
    asset = mjcf.load(materialised["axis_rotated_90"], record_id="rotated")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    semantic = next(
        r for r in kf1.axis_semantic(contract, bound) if r.subject == "drawer_1_slide"
    )
    motion = next(
        r for r in kf1.axis_admits_motion(contract, bound) if r.subject == "drawer_1_slide"
    )
    assert semantic.verdict is Verdict.PASS
    assert motion.verdict is Verdict.FAIL


def test_an_unbound_part_is_na_rather_than_a_failure(materialised, contract):
    # The reader's gap must never be charged to the asset -- the rule the previous project
    # broke by scoring an unreadable field as an absent one.
    asset = mjcf.load(materialised["cabinet_correct"], record_id="control")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    stripped = binding_mod.Binding(
        parts={k: v for k, v in bound.parts.items() if k != "drawer_1"},
        source=bound.source,
        asset=bound.asset,
    )
    verdicts = {r.subject: r.verdict for r in kf1.parent(contract, stripped)}
    assert verdicts["drawer_1_slide"] is Verdict.NA
    assert verdicts["drawer_2_slide"] is Verdict.PASS


def test_every_result_carries_the_number_it_was_decided_by(materialised, contract):
    # A verdict nobody can argue with is an assertion, not evidence.
    asset = mjcf.load(materialised["cabinet_correct"], record_id="control")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    for result in kf1.evaluate(contract, bound):
        if result.verdict is Verdict.NA:
            assert result.reason
        else:
            assert result.measured, result
            assert result.reason


def test_evaluation_is_reproducible(materialised, contract):
    asset = mjcf.load(materialised["cabinet_correct"], record_id="control")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    runs = {tuple(str(r) for r in kf1.evaluate(contract, bound)) for _ in range(3)}
    assert len(runs) == 1
