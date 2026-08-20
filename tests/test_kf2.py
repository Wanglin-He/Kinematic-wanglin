"""KF2, judged by a gearbox whose coupling we broke three different ways.

The asset that matters most here is ``gearbox_missing_coupling``: geometrically perfect
gears with nothing linking them. The specification's printed formula scores it a perfect
one -- no constraint means no residual, and a maximum over an empty set satisfies any
tolerance -- so the single most severe coupling failure takes full marks. The ``bound()``
factor exists for that asset, and this file is where it is held to it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evo_p0p3.p0 import admission
from evo_p0p3.p0.loader import parse_contract
from evo_p0p3.p3 import binding as binding_mod
from evo_p0p3.p3 import gold, kf2, mjcf
from evo_p0p3.p3.verdict import Verdict, score

ROOT = Path(__file__).resolve().parents[1]
GEARBOX = gold.defects("gearbox_correct.urdf")


@pytest.fixture(scope="module")
def contract():
    raw = yaml.safe_load((ROOT / "contracts" / "gold_gearbox.yaml").read_text(encoding="utf-8"))
    return parse_contract(raw, record_id="gold_gearbox")


@pytest.fixture(scope="module")
def materialised(tmp_path_factory) -> dict[str, Path]:
    return gold.write_all(tmp_path_factory.mktemp("kf2"))


def evaluate(name, materialised, contract):
    asset = mjcf.load(materialised[name], record_id=name)
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    return {r.predicate: r for r in kf2.evaluate(contract, bound)}


# --------------------------------------------------------------------------------------
# the contract and the control
# --------------------------------------------------------------------------------------


def test_the_gearbox_contract_is_admitted(contract):
    report = admission.check(contract)
    assert report.admitted, [str(f) for f in report.errors]
    assert not report.warnings, [str(f) for f in report.warnings]


def test_the_control_passes_every_kf2_predicate(materialised, contract):
    results = evaluate("gearbox_correct", materialised, contract)
    failed = {p: r.reason for p, r in results.items() if r.verdict is Verdict.FAIL}
    assert not failed, failed


def test_the_control_scores_one(materialised, contract):
    asset = mjcf.load(materialised["gearbox_correct"], record_id="control")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    assert score(kf2.evaluate(contract, bound)) == 1.0


def test_the_control_residual_is_exactly_zero(materialised, contract):
    # Placed on the declared manifold, a model enforcing the declared relation has nothing
    # to violate. Anything else would mean the two descriptions disagree.
    result = evaluate("gearbox_correct", materialised, contract)["KF2.residual"]
    assert result.measured["max_residual"] == pytest.approx(0.0, abs=1e-9)


def test_the_gears_actually_mesh_in_the_control(materialised, contract):
    # A ratio is a claim about teeth that touch. Two discs at the right ratio but far
    # apart would satisfy every constraint check while meshing nothing, so the control has
    # to be tangent for the coupling claims to mean what they say.
    asset = mjcf.load(materialised["gearbox_correct"], record_id="control")
    distance, _ = mjcf.body_pair_distance(
        asset, asset.body_id("gear_large"), asset.body_id("gear_small"), distmax=1.0
    )
    assert distance == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------------------
# the defects
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("defect", GEARBOX, ids=lambda d: d.name)
def test_the_verdict_matrix_matches_the_manifest(defect, materialised, contract):
    observed = {p: r.verdict.value for p, r in evaluate(defect.name, materialised, contract).items()}
    mismatches = {
        predicate: (expected, observed.get(predicate))
        for predicate, expected in defect.expect.items()
        if predicate.startswith("KF2.") and observed.get(predicate) != expected
    }
    assert not mismatches, f"expected vs observed: {mismatches}"


@pytest.mark.parametrize("defect", GEARBOX, ids=lambda d: d.name)
def test_the_manifest_covers_every_kf2_predicate(defect):
    named = {f"KF2.{p.__name__}" for p in kf2.PREDICATES}
    assert named <= set(defect.expect), named - set(defect.expect)


def test_a_declared_coupling_the_model_never_built_scores_zero(materialised, contract):
    # The whole reason the printed formula needed changing. Without bound(), this asset --
    # two gears spinning entirely independently -- is a perfect one.
    asset = mjcf.load(materialised["gearbox_missing_coupling"], record_id="missing")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    results = kf2.evaluate(contract, bound)
    assert score(results) == 0.0
    assert asset.model.neq == 0


def test_the_missing_coupling_is_caught_from_two_directions(materialised, contract):
    # bound() reads the model's constraint list; expected_dof counts what is left free.
    # A model that fooled one would have to fool the other by a different mechanism.
    results = evaluate("gearbox_missing_coupling", materialised, contract)
    assert results["KF2.bound"].verdict is Verdict.FAIL
    assert results["KF2.expected_dof"].verdict is Verdict.FAIL
    assert results["KF2.expected_dof"].measured["remaining_dof"] == 2


def test_coefficient_and_residual_abstain_when_there_is_nothing_to_read(
    materialised, contract
):
    # Reporting "the ratio is wrong" for a coupling that does not exist would name the
    # wrong defect. The absence is bound()'s finding, and these two say so.
    results = evaluate("gearbox_missing_coupling", materialised, contract)
    assert results["KF2.coefficient"].verdict is Verdict.NA
    assert results["KF2.residual"].verdict is Verdict.NA
    assert "KF2.bound" in results["KF2.residual"].reason


def test_a_wrong_ratio_shows_up_as_a_residual_that_grows_with_the_input(
    materialised, contract
):
    # Declared -3, built -2. At a full turn of the input the two descriptions disagree by
    # exactly one turn of the output.
    result = evaluate("gearbox_wrong_ratio", materialised, contract)["KF2.residual"]
    assert result.verdict is Verdict.FAIL
    assert result.measured["max_residual"] == pytest.approx(6.283, abs=1e-3)


def test_a_flipped_sign_is_a_larger_disagreement_than_a_wrong_magnitude(
    materialised, contract
):
    # -3 built as +3 is six turns of disagreement against the two of -3 built as -2, which
    # is right: gears turning the same way is a worse error than gears at the wrong ratio.
    wrong_ratio = evaluate("gearbox_wrong_ratio", materialised, contract)["KF2.residual"]
    wrong_sign = evaluate("gearbox_wrong_sign", materialised, contract)["KF2.residual"]
    assert wrong_sign.measured["max_residual"] > wrong_ratio.measured["max_residual"]
    assert wrong_sign.measured["max_residual"] == pytest.approx(37.698, abs=1e-2)


def test_a_flipped_sign_is_named_as_a_sign_error(materialised, contract):
    # External gears counter-rotate, so a positive ratio here describes a mechanism that
    # cannot exist. The report should say that rather than quoting a percentage.
    result = evaluate("gearbox_wrong_sign", materialised, contract)["KF2.coefficient"]
    assert result.verdict is Verdict.FAIL
    assert "sign" in result.reason


def test_the_coupling_survives_the_urdf_import(materialised, contract):
    # URDF can only declare a coupling with <mimic>, which MuJoCo's importer discards.
    # Without the loader recovering it, every one of these assets would look uncoupled and
    # KF2 would blame the asset for the reader's gap.
    asset = mjcf.load(materialised["gearbox_correct"], record_id="control")
    assert asset.model.neq == 1
    assert asset.provenance["mimics_translated"][0]["multiplier"] == -3.0


def test_the_ratio_the_model_enforces_is_read_not_inferred(materialised, contract):
    result = evaluate("gearbox_correct", materialised, contract)["KF2.coefficient"]
    assert result.measured["coefficient"] == pytest.approx(-3.0)
    assert result.measured["polycoef"][:2] == [0.0, -3.0]


def test_results_carry_the_configuration_a_failure_first_appeared_at(
    materialised, contract
):
    result = evaluate("gearbox_wrong_ratio", materialised, contract)["KF2.residual"]
    assert result.measured["first_failing_q"] is not None
    assert result.measured["at_independent_q"] is not None


def test_evaluation_is_reproducible(materialised, contract):
    asset = mjcf.load(materialised["gearbox_wrong_ratio"], record_id="wr")
    bound = binding_mod.identity(asset, tuple(contract.part_ids))
    runs = {tuple(str(r) for r in kf2.evaluate(contract, bound)) for _ in range(3)}
    assert len(runs) == 1


def test_an_asset_with_no_coupling_claim_produces_no_kf2_results(materialised):
    # The cabinet declares none, so KF2 is N/A for it -- excluded from the profile rather
    # than folded in as either extreme. An unmeasured dimension is not a perfect one.
    raw = yaml.safe_load((ROOT / "contracts" / "gold_cabinet.yaml").read_text(encoding="utf-8"))
    cabinet = parse_contract(raw, record_id="gold_cabinet")
    asset = mjcf.load(materialised["cabinet_correct"], record_id="cabinet")
    bound = binding_mod.identity(asset, tuple(cabinet.part_ids))
    assert kf2.evaluate(cabinet, bound) == ()
    assert score(kf2.evaluate(cabinet, bound)) is None
