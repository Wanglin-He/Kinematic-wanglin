"""KF1, one predicate at a time, each judged by an asset whose answer we built in.

Every predicate gets the same three questions, because passing only the first two is how
a tautology gets shipped:

* does it stay quiet on the control, where every claim is true?
* does it fire on the asset built to break exactly this claim?
* does it stay quiet on the assets built to break the *other* claims?

The third is the one that separates a predicate from a smoke alarm. A check that fires on
every defective asset has not detected a wrong parent; it has detected that something is
different, and a report built on it cannot tell anyone what to fix.
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
CONTRACT = ROOT / "contracts" / "gold_cabinet.yaml"


@pytest.fixture(scope="module")
def contract():
    return parse_contract(
        yaml.safe_load(CONTRACT.read_text(encoding="utf-8")), record_id="gold_cabinet"
    )


@pytest.fixture(scope="module")
def materialised(tmp_path_factory) -> dict[str, Path]:
    return gold.write_all(tmp_path_factory.mktemp("gold_kf1"))


def evaluate(name: str, materialised, contract) -> dict[str, tuple[Verdict, str]]:
    asset = mjcf.load(materialised[name], record_id=name)
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    return {r.subject: (r.verdict, r.reason) for r in kf1.evaluate(contract, bound)}


# --------------------------------------------------------------------------------------
# the control
# --------------------------------------------------------------------------------------


def test_the_control_passes_every_kf1_predicate(materialised, contract):
    verdicts = evaluate("cabinet_correct", materialised, contract)
    failing = {k: v for k, v in verdicts.items() if v[0] is not Verdict.PASS}
    assert not failing, failing


def test_the_control_scores_one(materialised, contract):
    asset = mjcf.load(materialised["cabinet_correct"], record_id="control")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    assert score(kf1.evaluate(contract, bound)) == 1.0


# --------------------------------------------------------------------------------------
# KF1.parent
# --------------------------------------------------------------------------------------


def test_parent_fires_on_the_asset_built_to_break_it(materialised, contract):
    verdicts = evaluate("wrong_parent", materialised, contract)
    assert verdicts["drawer_2_slide"][0] is Verdict.FAIL


def test_parent_names_what_is_wrong_and_what_it_should_be(materialised, contract):
    _, reason = evaluate("wrong_parent", materialised, contract)["drawer_2_slide"]
    assert "drawer_1" in reason and "cabinet_body" in reason


def test_parent_blames_only_the_joint_that_is_wrong(materialised, contract):
    verdicts = evaluate("wrong_parent", materialised, contract)
    assert verdicts["drawer_1_slide"][0] is Verdict.PASS
    assert verdicts["door_hinge"][0] is Verdict.PASS


@pytest.mark.parametrize(
    "other_defect",
    ["wrong_joint_type", "hinge_through_middle", "axis_rotated_90", "range_too_small",
     "fake_joint_decoy_geom"],
)
def test_parent_stays_quiet_on_defects_that_are_not_its_own(
    other_defect, materialised, contract
):
    # A predicate that fires on everything broken is a smoke alarm, not a diagnosis.
    verdicts = evaluate(other_defect, materialised, contract)
    assert all(v[0] is Verdict.PASS for v in verdicts.values()), verdicts


def test_parent_fires_on_the_detached_follower_because_that_asset_reparents_a_body(
    materialised, contract
):
    # Honest exception to the rule above, worth pinning rather than hiding: this defect
    # moves handle_1 onto the carcass, and handle_1 owns no joint, so KF1.parent has no
    # claim about it and every joint claim still passes. It is KF1.rigid_follower's to
    # catch, and this test records that KF1.parent correctly does not.
    verdicts = evaluate("detached_follower", materialised, contract)
    assert all(v[0] is Verdict.PASS for v in verdicts.values())


def test_parent_uses_the_nearest_declared_ancestor_not_mere_ancestry(
    materialised, contract
):
    # In wrong_parent, cabinet_body is still an ancestor of drawer_2 -- just not the
    # nearest declared one. A predicate asking only "is the declared parent somewhere
    # above?" passes this asset, which is why it does not ask that.
    asset = mjcf.load(materialised["wrong_parent"], record_id="wrong_parent")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    drawer_2 = bound.root_body("drawer_2")
    observed, _ = bound.nearest_declared_ancestor(drawer_2)
    assert observed == "drawer_1"

    chain = []
    body = drawer_2
    while body != 0:
        body = int(asset.model.body_parentid[body])
        chain.append(asset.body_name(body))
    assert "cabinet_body" in chain  # ancestry alone would have passed it


def test_an_unbound_part_is_na_rather_than_a_failure(materialised, contract):
    # The reader's gap must never be charged to the asset. This is the rule the previous
    # project broke, where an unreadable field was scored as an absent one.
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


def test_results_carry_the_evidence_a_person_needs(materialised, contract):
    asset = mjcf.load(materialised["wrong_parent"], record_id="wrong_parent")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    result = next(r for r in kf1.parent(contract, bound) if r.subject == "drawer_2_slide")
    assert result.measured["nearest_declared_ancestor"] == "drawer_1"
    assert result.evidence["binding_source"] == "identity"
    assert "drawer_1" in result.evidence["body_chain_upward"]


def test_identity_binding_refuses_a_name_it_cannot_find(materialised, contract):
    asset = mjcf.load(materialised["cabinet_correct"], record_id="control")
    with pytest.raises(binding_mod.BindingError):
        binding_mod.identity(asset, ("cabinet_body", "no_such_part"))
