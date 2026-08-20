"""KF3, and the asset that justifies sweeping at all.

``swept_interference`` puts a bracket in the door's arc. With the door closed it stands
well clear; with the door fully open the panel has swung past it; the collision lives only
between them, around thirty degrees. Every check evaluated at the declared states -- which
is every check that does not sweep -- reports that asset as clean.

The sweep costs 1,119 configurations for three degrees of freedom where a full grid would
be 35,937, and the tests below hold it to both halves of that bargain: cheap enough to run,
and thorough enough to find the thing it exists to find.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evo_p0p3.p0.loader import parse_contract
from evo_p0p3.p3 import binding as binding_mod
from evo_p0p3.p3 import gold, kf3, mjcf
from evo_p0p3.p3.sweep import SCAN_POINTS, Sweeper
from evo_p0p3.p3.verdict import Verdict, score

ROOT = Path(__file__).resolve().parents[1]
CABINET = gold.defects("cabinet_correct.urdf")


@pytest.fixture(scope="module")
def contract():
    raw = yaml.safe_load((ROOT / "contracts" / "gold_cabinet.yaml").read_text(encoding="utf-8"))
    return parse_contract(raw, record_id="gold_cabinet")


@pytest.fixture(scope="module")
def materialised(tmp_path_factory) -> dict[str, Path]:
    return gold.write_all(tmp_path_factory.mktemp("kf3"))


def bound_asset(name, materialised, contract):
    asset = mjcf.load(materialised[name], record_id=name)
    return binding_mod.identity(asset, tuple(contract.part_ids))


def verdicts(name, materialised, contract) -> dict[str, set[Verdict]]:
    out: dict[str, set[Verdict]] = {}
    for r in kf3.evaluate(contract, bound_asset(name, materialised, contract)):
        out.setdefault(r.predicate, set()).add(r.verdict)
    return out


def collapse(v: set[Verdict]) -> str:
    if Verdict.FAIL in v:
        return "fail"
    return "na" if v == {Verdict.NA} else "pass"


# --------------------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------------------


def test_the_sweep_is_far_cheaper_than_a_full_grid(materialised, contract):
    schedule = Sweeper(contract, bound_asset("cabinet_correct", materialised, contract)).schedule()
    assert schedule.size < SCAN_POINTS**3 / 10


def test_every_layer_contributes(materialised, contract):
    # A layer producing nothing is a layer whose cost is paid and whose coverage is not.
    schedule = Sweeper(contract, bound_asset("cabinet_correct", materialised, contract)).schedule()
    assert set(schedule.by_layer()) == {"reference", "scan", "pair", "states", "fill"}
    assert all(count > 0 for count in schedule.by_layer().values())


def test_the_schedule_is_identical_across_runs(materialised, contract):
    # Nothing is drawn, so there is no seed. A verifier whose answer moves between runs
    # cannot be argued with: nobody could separate a regression from noise.
    import numpy as np

    bound = bound_asset("cabinet_correct", materialised, contract)
    runs = [Sweeper(contract, bound).schedule() for _ in range(2)]
    assert runs[0].size == runs[1].size
    for a, b in zip(runs[0].samples, runs[1].samples, strict=True):
        assert a.label == b.label
        assert np.array_equal(a.qpos, b.qpos)


def test_the_schedule_records_which_pairs_it_skipped(materialised, contract):
    # Silent truncation reads as "everything was covered". What the adjacency gate pruned
    # has to be visible in the report.
    schedule = Sweeper(contract, bound_asset("cabinet_correct", materialised, contract)).schedule()
    assert "pairs_swept" in schedule.provenance
    assert "pairs_skipped" in schedule.provenance


def test_dependent_joints_are_driven_from_the_contract(materialised):
    # The sweep places the model where P0 says it should be. Whether the model's own
    # constraint agrees there is KF2's residual, not the sampler's business.
    raw = yaml.safe_load((ROOT / "contracts" / "gold_gearbox.yaml").read_text(encoding="utf-8"))
    gearbox = parse_contract(raw, record_id="gearbox")
    asset = mjcf.load(materialised["gearbox_correct"], record_id="gearbox")
    bound = binding_mod.identity(asset, tuple(gearbox.part_ids))
    sweeper = Sweeper(gearbox, bound)
    schedule = sweeper.schedule()
    assert schedule.driven == ("gear_large_spin",)
    assert schedule.dependent == ("gear_small_spin",)

    model = asset.model
    big = int(model.jnt_qposadr[asset.joint_id("gear_large_spin")])
    small = int(model.jnt_qposadr[asset.joint_id("gear_small_spin")])
    for sample in schedule.samples:
        assert sample.qpos[small] == pytest.approx(-3.0 * sample.qpos[big], abs=1e-9)


# --------------------------------------------------------------------------------------
# the control
# --------------------------------------------------------------------------------------


def test_the_control_passes_every_kf3_predicate(materialised, contract):
    failed = {p: v for p, v in verdicts("cabinet_correct", materialised, contract).items()
              if Verdict.FAIL in v}
    assert not failed, failed


def test_the_control_scores_one(materialised, contract):
    results = kf3.evaluate(contract, bound_asset("cabinet_correct", materialised, contract))
    assert score(results) == 1.0


def test_the_control_actually_meets_its_declared_stops(materialised, contract):
    # A limit enforced only by a range attribute, with no geometry stopping the part, is a
    # declaration rather than a mechanism. The control has a front lip so the door has
    # something to rest against, and the drawers bottom out against the carcass.
    session = kf3.Session(contract, bound_asset("cabinet_correct", materialised, contract))
    for result in session.required_contacts():
        assert result.verdict is Verdict.PASS, result.reason


# --------------------------------------------------------------------------------------
# the defects
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("defect", CABINET, ids=lambda d: d.name)
def test_the_verdict_matrix_matches_the_manifest(defect, materialised, contract):
    observed = {p: collapse(v) for p, v in verdicts(defect.name, materialised, contract).items()}
    mismatches = {
        predicate: (expected, observed.get(predicate))
        for predicate, expected in defect.expect.items()
        if predicate.startswith("KF3.") and observed.get(predicate) != expected
    }
    assert not mismatches, f"expected vs observed: {mismatches}"


@pytest.mark.parametrize("defect", CABINET, ids=lambda d: d.name)
def test_the_manifest_covers_every_kf3_predicate(defect):
    assert {
        "KF3.forbidden_pair", "KF3.required_contact", "KF3.state_reachability"
    } <= set(defect.expect)


def test_the_swept_interference_is_invisible_at_both_declared_states(
    materialised, contract
):
    # The claim this whole layer rests on, checked rather than asserted: at closed and at
    # open the asset is clean, so a verifier that only visited the named states would pass
    # it. If this ever stopped being true the defect would stop proving anything.
    import math

    import mujoco

    bound = bound_asset("swept_interference", materialised, contract)
    asset = bound.asset
    adr = int(asset.model.jnt_qposadr[asset.joint_id("door_hinge")])
    for angle in (0.0, math.pi / 2):
        asset.data.qpos[:] = asset.model.qpos0
        asset.data.qpos[adr] = angle
        mujoco.mj_forward(asset.model, asset.data)
        distance, _ = mjcf.body_pair_distance(
            asset, bound.root_body("cabinet_body"), bound.root_body("door"), distmax=0.5
        )
        assert distance >= -1e-6, f"already interfering at {angle} rad"


def test_the_swept_interference_is_found_in_between(materialised, contract):
    session = kf3.Session(contract, bound_asset("swept_interference", materialised, contract))
    failures = [r for r in session.forbidden_pairs() if r.verdict is Verdict.FAIL]
    assert failures, "the sweep missed a collision it was built to find"
    worst = max(failures, key=lambda r: r.measured["max_penetration_m"])
    assert worst.measured["max_penetration_m"] > 0.02


def test_a_failure_names_the_configuration_to_reproduce_it(materialised, contract):
    session = kf3.Session(contract, bound_asset("swept_interference", materialised, contract))
    failure = next(r for r in session.forbidden_pairs() if r.verdict is Verdict.FAIL)
    assert failure.evidence["first_failing_configuration"]
    assert failure.evidence["worst_configuration"]
    assert failure.measured["samples_evaluated"] > 100


def test_scoring_is_per_pair_not_per_sample(materialised, contract):
    # One deep interpenetration among a thousand configurations is one broken pair. Divided
    # by the sample count it would read 0.999, and the same defect would move by three
    # orders of magnitude with the sweep's resolution.
    session = kf3.Session(contract, bound_asset("swept_interference", materialised, contract))
    results = session.forbidden_pairs()
    assert len(results) == len(contract.kinematic_claims.contact_policy.forbidden)
    failing = [r for r in results if r.verdict is Verdict.FAIL]
    assert len(failing) == 1
    assert failing[0].measured["samples_evaluated"] > 1000


# --------------------------------------------------------------------------------------
# precedence
# --------------------------------------------------------------------------------------


def test_a_required_contact_exempts_the_same_pair_from_the_forbidden_rule(
    materialised, contract
):
    # The control declares door-against-carcass both required at the closed state and
    # forbidden at all states. Without precedence the asset would be undefined at exactly
    # the configuration the sweep starts from.
    policy = contract.kinematic_claims.contact_policy
    pairs_required = {frozenset(c.parts) for c in policy.required}
    pairs_forbidden = {frozenset(c.parts) for c in policy.forbidden}
    assert pairs_required & pairs_forbidden

    results = kf3.evaluate(contract, bound_asset("cabinet_correct", materialised, contract))
    assert all(r.verdict is not Verdict.FAIL for r in results)


def test_evaluation_is_reproducible(materialised, contract):
    bound = bound_asset("cabinet_correct", materialised, contract)
    runs = {tuple(str(r) for r in kf3.evaluate(contract, bound)) for _ in range(2)}
    assert len(runs) == 1


def test_results_carry_both_clearance_and_penetration(materialised, contract):
    session = kf3.Session(contract, bound_asset("cabinet_correct", materialised, contract))
    for result in session.forbidden_pairs():
        assert "min_clearance_m" in result.measured
        assert "max_penetration_m" in result.measured
        assert result.evidence["distmax"] == kf3.DISTMAX
