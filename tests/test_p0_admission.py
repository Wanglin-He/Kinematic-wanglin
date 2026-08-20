"""Every admission rule must be able to fire.

A rule nothing can trigger is worse than no rule: it reads as coverage while checking
nothing. This is the same failure the kinematic metrics already walked into once, where a
fault injection that displaced a joint origin by a full link diagonal produced zero
detection because the quantity was derived from the field being checked.

So each test below breaks the gold contract in exactly one way and asserts the intended
rule fires. The first test asserts the reverse: unmodified, the gold contract is clean.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from evo_p0p3.p0 import admission
from evo_p0p3.p0.loader import ContractSyntaxError, parse_contract
from evo_p0p3.p0.schema import ContactClaim

GOLD = Path(__file__).resolve().parents[1] / "contracts" / "gold_cabinet.yaml"


@pytest.fixture
def raw() -> dict:
    return yaml.safe_load(GOLD.read_text(encoding="utf-8"))


def rules_fired(raw: dict) -> set[str]:
    report = admission.check(parse_contract(raw, record_id="test"))
    return {f.rule for f in report.findings if f.severity is admission.Severity.ERROR}


def joint(raw: dict, jid: str) -> dict:
    return next(j for j in raw["kinematic_claims"]["joints"] if j["id"] == jid)


# --------------------------------------------------------------------------------------


def test_gold_contract_is_admitted_with_no_findings(raw):
    report = admission.check(parse_contract(raw, record_id="gold"))
    assert report.admitted, [str(f) for f in report.errors]
    assert not report.warnings, [str(f) for f in report.warnings]


def test_a1_fires_on_an_undeclared_part(raw):
    raw["kinematic_claims"]["contact_policy"]["forbidden"].append(
        {"parts": ["drawer_side", "cabinet_body"], "state": "all"}
    )
    assert "A1" in rules_fired(raw)


def test_a1_fires_on_an_undeclared_joint_in_independent_dofs(raw):
    raw["kinematic_claims"]["independent_dofs"].append("phantom_slide")
    assert "A1" in rules_fired(raw)


def test_a2_fires_when_a_part_has_no_geometry(raw):
    raw["part_geometry"] = [g for g in raw["part_geometry"] if g["id"] != "door"]
    assert "A2" in rules_fired(raw)


def test_a3_fires_when_a_movable_part_has_no_joint(raw):
    raw["kinematic_claims"]["joints"] = [
        j for j in raw["kinematic_claims"]["joints"] if j["id"] != "door_hinge"
    ]
    assert "A3" in rules_fired(raw)


def test_a3_fires_when_an_attached_part_owns_a_joint(raw):
    joint(raw, "door_hinge")["part"] = "door_knob"
    assert "A3" in rules_fired(raw)


def test_a4_fires_when_an_attached_part_has_no_leader(raw):
    raw["kinematic_claims"]["rigid_attachments"] = [
        a for a in raw["kinematic_claims"]["rigid_attachments"] if a["follower"] != "handle_1"
    ]
    assert "A4" in rules_fired(raw)


def test_a4_fires_when_a_movable_part_is_declared_a_follower(raw):
    raw["kinematic_claims"]["rigid_attachments"].append(
        {"follower": "drawer_1", "leader": "cabinet_body"}
    )
    assert "A4" in rules_fired(raw)


def test_a6_fires_when_a_dependent_joint_is_also_driven(raw):
    raw["kinematic_claims"]["couplings"] = [
        {
            "id": "bogus",
            "relation": {
                "dependent": "drawer_2_slide",
                "independent": "drawer_1_slide",
                "coefficient": 1.0,
            },
            "expected_dof": 1,
        }
    ]
    assert "A6" in rules_fired(raw)


def test_a7_fires_on_an_undefined_state(raw):
    raw["kinematic_claims"]["contact_policy"]["required"][0]["state"] = "ajar"
    assert "A7" in rules_fired(raw)


def test_a8_fires_when_a_joint_declares_no_axis_form(raw):
    joint(raw, "door_hinge")["axis"] = {}
    assert "A8" in rules_fired(raw)


def test_a9_fires_on_numeric_axis_without_an_anchored_front(raw):
    joint(raw, "door_hinge")["axis"]["numeric"] = [0, 0, 1]
    assert "A9" in rules_fired(raw)


def test_a9_is_satisfied_once_front_is_anchored(raw):
    joint(raw, "door_hinge")["axis"]["numeric"] = [0, 0, 1]
    raw["kinematic_claims"]["canonical_frame"]["front"] = {"outward_normal_of": "door"}
    assert "A9" not in rules_fired(raw)


def test_a10_fires_when_a_hinge_has_no_anchor(raw):
    joint(raw, "door_hinge")["anchor"] = None
    assert "A10" in rules_fired(raw)


def test_a10_fires_when_a_slide_declares_an_anchor(raw):
    # A slide's pos does not enter the kinematics at all, so the claim is uncheckable.
    joint(raw, "drawer_1_slide")["anchor"] = {"on_edge_of": "drawer_1"}
    assert "A10" in rules_fired(raw)


def test_a11_fires_on_a_unit_that_does_not_match_the_joint_type(raw):
    joint(raw, "door_hinge")["range"]["unit"] = "m"
    assert "A11" in rules_fired(raw)


def test_a11_fires_when_a_state_sits_outside_its_range(raw):
    joint(raw, "drawer_1_slide")["states"]["open"] = 0.99
    assert "A11" in rules_fired(raw)


def test_a12_still_guards_a_self_pair_on_a_directly_built_contract(raw):
    # The loader rejects a literal self-pair outright (see the syntax test below), so this
    # covers the other way a Contract can come into being: constructed in code.
    contract = parse_contract(raw, record_id="gold")
    policy = contract.kinematic_claims.contact_policy
    broken = replace(
        contract,
        kinematic_claims=replace(
            contract.kinematic_claims,
            contact_policy=replace(
                policy, forbidden=policy.forbidden + (ContactClaim(parts=("door", "door")),)
            ),
        ),
    )
    assert "A12" in {f.rule for f in admission.check(broken).errors}


def test_a_literal_self_pair_is_a_syntax_error_not_a_silent_drop(raw):
    raw["kinematic_claims"]["contact_policy"]["forbidden"].append(
        {"parts": ["door", "door"], "state": "all"}
    )
    with pytest.raises(ContractSyntaxError) as exc:
        parse_contract(raw)
    assert "distinct" in str(exc.value)


def test_a12_fires_when_a_permission_sits_in_required(raw):
    raw["kinematic_claims"]["contact_policy"]["required"][0]["relation"] = "rest_contact"
    assert "A12" in rules_fired(raw)


def test_a13_fires_when_a_joint_connects_a_part_to_itself(raw):
    joint(raw, "door_hinge")["parent"] = "door"
    assert "A13" in rules_fired(raw)


def test_a15_fires_on_a_zero_coupling_coefficient(raw):
    raw["kinematic_claims"]["couplings"] = [
        {
            "id": "bogus",
            "relation": {
                "dependent": "drawer_2_slide",
                "independent": "drawer_1_slide",
                "coefficient": 0.0,
            },
            "expected_dof": 1,
        }
    ]
    raw["kinematic_claims"]["independent_dofs"].remove("drawer_2_slide")
    assert "A15" in rules_fired(raw)


# --------------------------------------------------------------------------------------
# loader: shape errors are a different failure from admission findings
# --------------------------------------------------------------------------------------


def test_count_is_rejected_with_an_explanation(raw):
    raw["required_parts"] = [{"id": "drawer", "count": 3, "role": "movable"}]
    with pytest.raises(ContractSyntaxError) as exc:
        parse_contract(raw)
    assert "enumerate instances" in str(exc.value)


def test_allowed_bucket_is_rejected_with_an_explanation(raw):
    raw["kinematic_claims"]["contact_policy"]["allowed"] = []
    with pytest.raises(ContractSyntaxError) as exc:
        parse_contract(raw)
    assert "ambiguous" in str(exc.value)


def test_anchor_demands_exactly_one_form(raw):
    joint(raw, "door_hinge")["anchor"] = {"on_edge_of": "door", "through_center_of": "door"}
    with pytest.raises(ContractSyntaxError):
        parse_contract(raw)


def test_relational_axis_needs_its_operands(raw):
    joint(raw, "door_hinge")["axis"]["relational"] = {"type": "interface_line"}
    with pytest.raises(ContractSyntaxError) as exc:
        parse_contract(raw)
    assert "between" in str(exc.value)


def test_unknown_tolerance_key_is_rejected(raw):
    raw["kinematic_claims"]["tolerances"]["axis_angle"] = 15.0
    with pytest.raises(ContractSyntaxError) as exc:
        parse_contract(raw)
    assert "axis_angle" in str(exc.value)


def test_inverted_range_is_rejected(raw):
    joint(raw, "drawer_1_slide")["range"] = {"unit": "m", "min": 0.4, "max": 0.0}
    with pytest.raises(ContractSyntaxError):
        parse_contract(raw)


# --------------------------------------------------------------------------------------
# group expansion
# --------------------------------------------------------------------------------------


def test_group_pair_expands_to_the_cross_product_without_self_pairs(raw):
    contract = parse_contract(raw, record_id="gold")
    forbidden = contract.kinematic_claims.contact_policy.forbidden
    drawer_pairs = [c for c in forbidden if c.expanded_from == "[group:drawer, group:drawer]"]
    assert [c.parts for c in drawer_pairs] == [("drawer_1", "drawer_2")]


def test_group_against_a_single_part_expands_to_one_pair_each(raw):
    contract = parse_contract(raw, record_id="gold")
    permitted = contract.kinematic_claims.contact_policy.permitted
    assert sorted(c.parts for c in permitted) == [
        ("handle_1", "cabinet_body"),
        ("handle_2", "cabinet_body"),
    ]


def test_expansion_is_recorded_so_the_pair_count_is_auditable(raw):
    contract = parse_contract(raw, record_id="gold")
    permitted = contract.kinematic_claims.contact_policy.permitted
    assert all(c.expanded_from == "[group:handle, cabinet_body]" for c in permitted)


def test_tolerance_digest_changes_when_a_threshold_changes(raw):
    before = parse_contract(raw, record_id="g").kinematic_claims.tolerances.digest()
    raw["kinematic_claims"]["tolerances"]["axis_angle_deg"] = 10.0
    after = parse_contract(raw, record_id="g").kinematic_claims.tolerances.digest()
    assert before != after


def test_parsing_is_stable_across_repeated_loads(raw):
    a = parse_contract(copy.deepcopy(raw), record_id="g")
    b = parse_contract(copy.deepcopy(raw), record_id="g")
    assert a == b


# --------------------------------------------------------------------------------------
# A10's joint-type split, measured rather than assumed
# --------------------------------------------------------------------------------------


def test_anchor_applies_to_the_joints_whose_kinematics_it_moves():
    # Displacing pos by 0.25 m moves the child geometry by: slide 0.000000,
    # hinge 0.171449, ball 0.191345, free 0.000000. An anchor claim is checkable for
    # exactly the two that move.
    from evo_p0p3.p0.schema import JointType

    assert [t for t in JointType if t.has_anchor] == [JointType.HINGE, JointType.BALL]


def test_a_ball_joint_may_declare_an_anchor(raw):
    # A ball joint rotates about its anchor point; putting that point in the wrong place
    # swings the child through a different arc, so the claim can genuinely fail.
    j = joint(raw, "door_hinge")
    j["type"] = "ball"
    j["anchor"] = {"through_center_of": "door"}
    assert "A10" not in rules_fired(raw)


def test_a_ball_joint_without_an_anchor_is_rejected(raw):
    j = joint(raw, "door_hinge")
    j["type"] = "ball"
    j["anchor"] = None
    assert "A10" in rules_fired(raw)
