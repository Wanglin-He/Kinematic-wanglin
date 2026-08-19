"""The loading recipe, tested on URDFs shaped like the real ones.

Inline rather than against the corpus on disk: the tests have to run for anyone who
clones this, and a test that silently skips when a path is missing is a test that stops
protecting anything.

The two facts being protected are the ones that decide P3's whole shape. Drop
``discardvisual="false"`` and an Articraft asset compiles to zero geoms, because
``<visual>`` is the only geometry it has -- every geometric predicate would then have
nothing to measure and would quietly pass. And no asset declares ``<collision>``, so
contacts are always empty and distance has to come from ``mj_geomDistance``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from evo_p0p3.p3 import mjcf

# A cabinet in miniature: static carcass, one prismatic drawer, one handle rigidly
# attached to the drawer (a link with no joint of its own). Visual geometry only and no
# inertial element, exactly like the real materialised assets.
CABINET = textwrap.dedent(
    """
    <robot name="cabinet">
      <link name="cabinet_body">
        <visual><origin xyz="0 0 0.25"/><geometry><box size="0.30 0.40 0.50"/></geometry></visual>
      </link>
      <link name="drawer">
        <visual><origin xyz="0 0 0"/><geometry><box size="0.26 0.34 0.12"/></geometry></visual>
      </link>
      <link name="handle">
        <visual><origin xyz="0 -0.19 0"/><geometry><cylinder radius="0.01" length="0.12"/></geometry></visual>
      </link>
      <joint name="drawer_slide" type="prismatic">
        <parent link="cabinet_body"/><child link="drawer"/>
        <origin xyz="0 0 0.30"/><axis xyz="0 -1 0"/>
        <limit lower="0.0" upper="0.24" effort="50" velocity="0.5"/>
      </joint>
      <joint name="handle_fixed" type="fixed">
        <parent link="drawer"/><child link="handle"/>
        <origin xyz="0 0 0"/>
      </joint>
    </robot>
    """
).strip()


@pytest.fixture
def cabinet(tmp_path: Path) -> mjcf.LoadedAsset:
    path = tmp_path / "model.urdf"
    path.write_text(CABINET, encoding="utf-8")
    return mjcf.load(path, record_id="cabinet")


def test_an_articraft_shaped_urdf_compiles(cabinet):
    assert cabinet.model.nbody >= 3
    assert cabinet.model.njnt == 1


def test_visual_geometry_survives_the_import(cabinet):
    # Without discardvisual="false" this is zero, and every geometric predicate would
    # then have nothing to measure while still reporting a pass.
    assert cabinet.model.ngeom == 3


def test_inertia_is_flagged_as_synthesized(cabinet):
    assert cabinet.inertia_synthesized
    assert cabinet.provenance["inertia_synthesized"] is True
    assert any("Never used in scoring" in n for n in cabinet.notes)


def test_no_geom_is_collidable_so_contacts_are_useless(cabinet):
    assert cabinet.collidable_geoms == 0
    assert cabinet.data.ncon == 0
    assert cabinet.provenance["distance_backend"] == "mj_geomDistance"


def test_distance_still_works_between_parent_and_child(cabinet):
    # The pair MuJoCo's parent filter would hide is exactly the one a swept-interference
    # check needs, so this must return a real number rather than nothing.
    body = cabinet.body_id("cabinet_body")
    drawer = cabinet.body_id("drawer")
    distance, which = mjcf.body_pair_distance(cabinet, body, drawer, distmax=2.0)
    assert which is not None
    assert distance < 2.0


def test_a_declared_inertial_is_left_alone(tmp_path: Path):
    with_inertial = CABINET.replace(
        '<link name="drawer">',
        '<link name="drawer">\n    <inertial><mass value="2.0"/>'
        '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>',
    )
    path = tmp_path / "model.urdf"
    path.write_text(with_inertial, encoding="utf-8")
    asset = mjcf.load(path)
    assert not asset.inertia_synthesized


def test_rigid_subtree_holds_a_jointless_follower(cabinet):
    drawer = cabinet.body_id("drawer")
    names = {cabinet.body_name(b) for b in cabinet.rigid_subtree(drawer)}
    assert names == {"drawer", "handle"}


def test_rigid_subtree_stops_at_a_joint(cabinet):
    body = cabinet.body_id("cabinet_body")
    names = {cabinet.body_name(b) for b in cabinet.rigid_subtree(body)}
    assert "drawer" not in names


def test_a_handle_bolted_to_the_carcass_is_not_in_the_drawers_subtree(tmp_path: Path):
    # The defect this distinguishes: a handle that looks right at the reference pose but
    # stays behind when the drawer opens.
    detached = CABINET.replace(
        '<parent link="drawer"/><child link="handle"/>',
        '<parent link="cabinet_body"/><child link="handle"/>',
    )
    path = tmp_path / "model.urdf"
    path.write_text(detached, encoding="utf-8")
    asset = mjcf.load(path)
    drawer = asset.body_id("drawer")
    assert "handle" not in {asset.body_name(b) for b in asset.rigid_subtree(drawer)}


def test_geoms_are_addressed_by_body_not_by_name(cabinet):
    # Real assets duplicate geom names across bodies and MuJoCo unnames the duplicates,
    # so the body tree is the only reliable handle.
    drawer = cabinet.body_id("drawer")
    assert len(cabinet.geoms_of(drawer)) == 1


def test_a_urdf_with_no_geometry_is_refused_rather_than_scored(tmp_path: Path):
    empty = textwrap.dedent(
        """
        <robot name="empty">
          <link name="a"/>
          <link name="b"/>
          <joint name="j" type="prismatic">
            <parent link="a"/><child link="b"/>
            <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
            <limit lower="0" upper="1" effort="1" velocity="1"/>
          </joint>
        </robot>
        """
    ).strip()
    path = tmp_path / "model.urdf"
    path.write_text(empty, encoding="utf-8")
    with pytest.raises(mjcf.AssetLoadError):
        mjcf.load(path)


def test_a_missing_file_is_a_load_error(tmp_path: Path):
    with pytest.raises(mjcf.AssetLoadError):
        mjcf.load(tmp_path / "nope.urdf")


def test_aabb_uses_the_parts_own_scale(cabinet):
    drawer = cabinet.body_id("drawer")
    lo, hi = mjcf.subtree_aabb(cabinet, (drawer,))
    assert 0.0 < mjcf.aabb_diagonal(lo, hi) < 1.0
