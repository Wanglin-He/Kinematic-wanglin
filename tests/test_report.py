"""The Gate, the report, and the whole pipeline end to end.

The report is where a run either becomes evidence or becomes a number. Three shapes are
refused outright rather than written, each of which the previous project produced at some
point: a failure with no reason, a verdict with nothing measured behind it, and an
abstention that does not say what could not be read. The last is the dangerous one -- an
unexplained N/A is indistinguishable from a pass to anyone reading a table.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from evo_p0p3.p0.loader import parse_contract
from evo_p0p3.p3 import gate, gold, report
from evo_p0p3.p3.cli import evaluate, main
from evo_p0p3.p3.verdict import ClaimResult, Verdict

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def materialised(tmp_path_factory) -> dict[str, Path]:
    return gold.write_all(tmp_path_factory.mktemp("report"))


@pytest.fixture(scope="module")
def cabinet():
    raw = yaml.safe_load((ROOT / "contracts" / "gold_cabinet.yaml").read_text(encoding="utf-8"))
    return parse_contract(raw, record_id="gold_cabinet")


# --------------------------------------------------------------------------------------
# the Gate
# --------------------------------------------------------------------------------------


def test_the_control_is_admitted(materialised, cabinet):
    admitted = gate.admit(materialised["cabinet_correct"], cabinet)
    assert admitted.admitted
    assert admitted.binding is not None


def test_g3_reports_but_does_not_gate(materialised, cabinet):
    # The loader synthesises inertia for the 341 of 546 real assets declaring none, and
    # P3's scope excludes mass entirely. Failing those out would discard two thirds of the
    # corpus over a property nothing here reads -- so the check runs and is reported, and
    # is not a precondition.
    admitted = gate.admit(materialised["cabinet_correct"], cabinet)
    g3 = next(c for c in admitted.checks if c.name == "G3")
    assert not g3.passed
    assert admitted.admitted
    assert "not a precondition" in g3.detail


def test_a_contract_naming_a_part_the_asset_lacks_fails_g1(materialised, cabinet):
    from evo_p0p3.p0.schema import Part, Role

    broken = replace(
        cabinet, required_parts=cabinet.required_parts + (Part(id="plinth", role=Role.FIXED),)
    )
    admitted = gate.admit(materialised["cabinet_correct"], broken)
    assert not admitted.admitted
    assert [c.name for c in admitted.failed] == ["G1"]
    assert admitted.binding is None


def test_an_unreadable_asset_fails_g0_and_stops_there(tmp_path, cabinet):
    path = tmp_path / "model.urdf"
    path.write_text("<robot name='x'><link name='a'/></robot>", encoding="utf-8")
    admitted = gate.admit(path, cabinet)
    assert not admitted.admitted
    assert admitted.checks[0].name == "G0"
    assert len(admitted.checks) == 1  # nothing downstream is guessed at


def test_a_rejected_asset_produces_a_report_with_no_claims(materialised, cabinet):
    from evo_p0p3.p0.schema import Part, Role

    broken = replace(
        cabinet, required_parts=cabinet.required_parts + (Part(id="plinth", role=Role.FIXED),)
    )
    result = evaluate(broken, materialised["cabinet_correct"])
    assert not result.admission.admitted
    assert result.results == ()
    assert result.profile == {"KF1": None, "KF2": None, "KF3": None}


# --------------------------------------------------------------------------------------
# the report contract
# --------------------------------------------------------------------------------------


def test_a_failure_without_a_reason_is_refused(materialised, cabinet):
    result = evaluate(cabinet, materialised["cabinet_correct"])
    bad = ClaimResult("KF1.parent", "x", Verdict.FAIL, "", measured={"a": 1})
    with pytest.raises(report.MalformedReport, match="no reason"):
        report.validate(replace(result, results=result.results + (bad,)))


def test_a_verdict_with_nothing_measured_is_refused(materialised, cabinet):
    result = evaluate(cabinet, materialised["cabinet_correct"])
    bad = ClaimResult("KF1.parent", "x", Verdict.PASS, "looks fine")
    with pytest.raises(report.MalformedReport, match="nothing measured"):
        report.validate(replace(result, results=result.results + (bad,)))


def test_an_unexplained_abstention_is_refused(materialised, cabinet):
    # The one that matters most: to a reader scanning a table, an N/A with no explanation
    # is indistinguishable from a pass.
    result = evaluate(cabinet, materialised["cabinet_correct"])
    bad = ClaimResult("KF1.anchor", "x", Verdict.NA, "")
    with pytest.raises(report.MalformedReport, match="no explanation"):
        report.validate(replace(result, results=result.results + (bad,)))


def test_every_gold_report_is_well_formed(materialised, cabinet):
    for name in ["cabinet_correct"] + [d.name for d in gold.defects("cabinet_correct.urdf")]:
        report.validate(evaluate(cabinet, materialised[name]))


# --------------------------------------------------------------------------------------
# the profile
# --------------------------------------------------------------------------------------


def test_the_control_scores_one_on_every_applicable_metric(materialised, cabinet):
    profile = evaluate(cabinet, materialised["cabinet_correct"]).profile
    assert profile["KF1"] == 1.0
    assert profile["KF3"] == 1.0
    assert profile["KF2"] is None  # the cabinet declares no coupling


def test_an_inapplicable_metric_is_none_rather_than_either_extreme(materialised, cabinet):
    # Not 1.0, which would credit an asset for a dimension nobody measured, and not 0.0,
    # which would punish it for the same.
    assert evaluate(cabinet, materialised["cabinet_correct"]).metric("KF2") is None


def test_kf1_is_split_by_how_each_claim_was_decided(materialised, cabinet):
    # Declaration reads matched in nearly every asset of the previous corpus, so a flat
    # proportion is dominated by claims that cannot vary.
    declared = evaluate(cabinet, materialised["range_too_small"]).sub_scores
    measured = evaluate(cabinet, materialised["hinge_through_middle"]).sub_scores
    assert declared["KF1.declared"] < 1.0 and declared["KF1.measured"] == 1.0
    assert measured["KF1.measured"] < 1.0 and measured["KF1.declared"] == 1.0


def test_applicability_is_reported_alongside_the_score(materialised, cabinet):
    result = evaluate(cabinet, materialised["cabinet_correct"])
    assert result.applicability()["KF2"] == "0/0"
    assert result.applicability()["KF3"].endswith("/9")


def test_provenance_travels_with_every_report(materialised, cabinet):
    # Nobody should have to take on trust that the synthesised inertia and the recovered
    # couplings did not influence a score.
    result = evaluate(cabinet, materialised["cabinet_correct"])
    assert result.provenance["inertia_synthesized"] is True
    assert result.provenance["distance_backend"] == "mj_geomDistance"
    assert result.tolerance_digest


def test_coverage_travels_with_every_report(materialised, cabinet):
    # A schedule that silently skipped half the configuration space would otherwise read
    # exactly like one that covered it.
    coverage = evaluate(cabinet, materialised["cabinet_correct"]).to_dict()["coverage"]
    assert coverage["samples"] > 1000
    assert set(coverage["by_layer"]) == {"reference", "scan", "pair", "states", "fill"}
    assert "pairs_skipped" in coverage


# --------------------------------------------------------------------------------------
# across a run
# --------------------------------------------------------------------------------------


def test_gate_pass_rate_is_reported_with_the_profile(materialised, cabinet):
    from evo_p0p3.p0.schema import Part, Role

    broken = replace(
        cabinet, required_parts=cabinet.required_parts + (Part(id="plinth", role=Role.FIXED),)
    )
    run = report.RunReport((
        evaluate(cabinet, materialised["cabinet_correct"]),
        evaluate(broken, materialised["cabinet_correct"]),
    ))
    assert run.gate_pass_rate == 0.5
    assert len(run.admitted) == 1
    assert "Gate pass rate" in run.render()


def test_the_mean_profile_is_taken_only_over_applicable_assets(materialised, cabinet):
    run = report.RunReport(tuple(
        evaluate(cabinet, materialised[n])
        for n in ["cabinet_correct", "wrong_parent", "hinge_through_middle"]
    ))
    assert run.mean_profile()["KF2"] is None
    assert run.applicable_counts()["KF2"] == "0/3"
    assert run.applicable_counts()["KF1"] == "3/3"


# --------------------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------------------


def test_the_gold_command_runs_the_whole_set(capsys):
    code = main(["gold"])
    out = capsys.readouterr().out
    assert code == 1  # defective assets fail, which is the point
    assert "Gate pass rate: 100%" in out
    assert "cabinet_correct" in out and "gearbox_correct" in out


def test_a_contract_that_fails_admission_stops_the_run(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "record_id: bad\nrequired_parts:\n  - {id: a, role: movable}\n"
        "kinematic_claims: {}\n",
        encoding="utf-8",
    )
    code = main(["run", str(bad), str(tmp_path / "nothing.urdf")])
    assert code == 2
    assert "blamed for the contract" in capsys.readouterr().err


def test_reports_serialise_and_round_trip(tmp_path, materialised, cabinet):
    import json

    result = evaluate(cabinet, materialised["swept_interference"])
    path = result.write(tmp_path / "r.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["profile"]["KF3"] < 1.0
    failing = [c for c in data["claims"] if c["verdict"] == "fail"]
    assert failing and failing[0]["evidence"]["first_failing_configuration"]
