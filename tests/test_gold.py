"""The gold assets earn their keep, or they are not gold.

Two things have to hold for a counterexample to be worth anything, and neither is
obvious enough to trust:

1. It must actually differ from the control *in the compiled model*. A substitution that
   matched nothing, or that cancels itself out, leaves an asset identical to the correct
   one -- and a predicate tested against it looks sound while checking nothing.
2. Its difference must be *minimal*. If the "wrong parent" asset also moved a geom and
   changed a range, then a predicate firing on it has not been shown to detect a wrong
   parent; it has been shown to detect that something is different.

The first test catches a fake counterexample. The second pins down what each one changed,
so that a later edit widening a defect shows up here rather than quietly weakening every
predicate it is used to test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evo_p0p3.p3 import gold, mjcf

MODEL_FIELDS = (
    "body_parentid", "body_pos", "body_quat", "body_jntnum", "body_geomnum",
    "jnt_type", "jnt_bodyid", "jnt_axis", "jnt_pos", "jnt_range", "jnt_limited",
    "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_bodyid", "eq_data",
)


def fingerprint(asset: mjcf.LoadedAsset) -> dict[str, bytes]:
    out = {}
    for name in MODEL_FIELDS:
        value = np.asarray(getattr(asset.model, name))
        if np.issubdtype(value.dtype, np.floating):
            value = np.round(value, 6)
        out[name] = value.tobytes()
    out["neq"] = bytes([asset.model.neq])
    return out


@pytest.fixture(scope="module")
def materialised(tmp_path_factory) -> dict[str, Path]:
    return gold.write_all(tmp_path_factory.mktemp("gold"))


@pytest.fixture(scope="module")
def control(materialised) -> mjcf.LoadedAsset:
    return mjcf.load(materialised["cabinet_correct"], record_id="cabinet_correct")


def control_for(defect, materialised) -> mjcf.LoadedAsset:
    """The control a defect was cut from. Comparing a gearbox against a cabinet would
    report every field as changed and prove nothing about the edit."""
    stem = Path(defect.family).stem
    return mjcf.load(materialised[stem], record_id=stem)


def test_the_control_compiles_with_its_geometry_intact(control):
    assert control.model.nbody == 8
    assert control.model.njnt == 3
    assert control.model.ngeom == 13  # six carcass panels + front lip + drawers, handles, door, knob
    assert control.inertia_synthesized  # visual-only, like the real assets


def test_the_control_has_no_self_intersection(control):
    # The control has to be clean, or a swept-interference predicate would fail on it and
    # the failure would be the asset's rather than the predicate's.
    pairs = [("drawer_1", "drawer_2"), ("door", "drawer_1"), ("door", "drawer_2")]
    for a, b in pairs:
        distance, _ = mjcf.body_pair_distance(
            control, control.body_id(a), control.body_id(b), distmax=1.0
        )
        assert distance > 0, f"{a} and {b} interpenetrate at the reference pose"


def test_the_drawers_start_inside_the_carcass(control):
    # Stated in the file's header as a measured property; checked here so it stays one.
    for drawer in ("drawer_1", "drawer_2"):
        lo, hi = mjcf.subtree_aabb(control, (control.body_id(drawer),))
        assert lo[1] > -0.21 and hi[1] < 0.19


@pytest.mark.parametrize("defect", gold.defects(), ids=lambda d: d.name)
def test_every_defect_actually_changes_the_compiled_model(defect, materialised):
    asset = mjcf.load(materialised[defect.name], record_id=defect.name)
    base = fingerprint(control_for(defect, materialised))
    changed = [k for k, v in fingerprint(asset).items() if v != base[k]]
    assert changed, (
        f"{defect.name} compiles to a model identical to the control, so it is not a "
        f"counterexample and anything tested against it proves nothing"
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        ("wrong_parent", {"body_parentid"}),
        ("wrong_joint_type", {"jnt_type", "jnt_range"}),
        ("hinge_through_middle", {"body_pos", "geom_pos"}),
        ("axis_rotated_90", {"jnt_axis"}),
        ("range_too_small", {"jnt_range"}),
        ("detached_follower", {"body_parentid", "body_pos"}),
    ],
)
def test_each_defect_changes_only_what_it_claims_to(name, expected, materialised, control):
    asset = mjcf.load(materialised[name], record_id=name)
    changed = {k for k, v in fingerprint(asset).items() if v != fingerprint(control)[k]}
    assert changed == expected


def test_the_hinge_defect_moves_the_axis_and_not_the_door(materialised, control):
    # The sharpest of the set, and the one whose minimality is easiest to get wrong: the
    # door must end up in the same place, with only the line it turns about displaced from
    # its left edge to its middle. Otherwise an anchor predicate could be passing by
    # noticing the door moved.
    import mujoco

    defective = mjcf.load(materialised["hinge_through_middle"], record_id="hinge")
    poses = {}
    for tag, asset in (("control", control), ("defect", defective)):
        mujoco.mj_forward(asset.model, asset.data)
        door = asset.body_id("door")
        poses[tag] = (
            np.array(asset.data.xanchor[asset.joint_id("door_hinge")]),
            np.array(asset.data.geom_xpos[asset.model.body_geomadr[door]]),
        )
    assert np.allclose(poses["control"][1], poses["defect"][1], atol=1e-9)
    assert not np.allclose(poses["control"][0], poses["defect"][0], atol=1e-6)
    assert poses["control"][0][0] == pytest.approx(-0.17)  # left edge
    assert poses["defect"][0][0] == pytest.approx(0.0)  # dead centre


def test_a_substitution_that_matches_nothing_is_an_error():
    with pytest.raises(gold.DefectNotApplied):
        gold._apply(gold.correct_urdf(), [{"find": "no such text", "replace": "x"}], "bogus")


def test_materialising_is_reproducible():
    a = {d.name: d.urdf for d in gold.defects()}
    b = {d.name: d.urdf for d in gold.defects()}
    assert a == b
