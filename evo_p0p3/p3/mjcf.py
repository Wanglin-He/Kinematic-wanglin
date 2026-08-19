"""Loading an Articraft asset into MuJoCo, and being honest about what we changed.

Articraft ships ``model.urdf`` plus OBJ meshes under
``cache/record_materialization/<record_id>/``. Two things about those files decide the
whole shape of P3, and both were measured across all 546 materialised assets rather than
assumed:

**No asset has a ``<collision>`` element.** Every geom therefore compiles with
``contype = conaffinity = 0``, MuJoCo's broad phase skips all of them, and ``mjData.ncon``
is always zero. Contact-based collision detection is not merely awkward here, it returns
nothing at all. Every distance in P3 goes through :func:`mujoco.mj_geomDistance`, which
ignores ``contype``/``conaffinity`` and the parent-child filter alike. That last part is
not a workaround but the correct behaviour: MuJoCo disables collision between a parent and
child joined by a joint, which is exactly the pair -- door against its own frame, drawer
against its own carcass -- that a swept-interference check most needs to see.

**Only 205 of 546 declare ``<inertial>``.** MuJoCo refuses to compile a moving body with
no mass, so the rest need one synthesised. That value is ours, not the asset's, and
:attr:`LoadedAsset.inertia_synthesized` records it so no score can ever be built on it.
P3's scope excludes mass, friction, damping and actuation anyway; the synthetic inertia
exists only to get past the compiler. Whether an asset's *real* inertial properties are
valid is Gate G3's question, and 341 of these assets will genuinely fail it.

With that recipe, 545 of the 546 load. The remaining one has a link whose declared inertia
is itself degenerate.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

COMPILER_DIRECTIVE = (
    '<mujoco><compiler discardvisual="false" inertiafromgeom="true" '
    'balanceinertia="true" strippath="false" fusestatic="false"/></mujoco>'
)
"""What MuJoCo needs in order to keep an Articraft URDF's geometry.

``discardvisual="false"`` is the load-bearing one. MuJoCo's URDF importer treats
``<visual>`` as decoration and drops it by default, which on these assets means dropping
*everything* -- they contain no other geometry. Without this flag the model compiles to
zero geoms and every geometric predicate silently has nothing to measure.

``fusestatic="false"`` keeps a static root link as its own body rather than merging it
into the world, so a part id declared ``fixed`` still has a body to bind to.
"""

SYNTHETIC_INERTIAL = (
    '<inertial><mass value="1.0"/>'
    '<inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>'
)

_ROBOT = re.compile(r"(<robot[^>]*>)")
_LINK = re.compile(r'(<link name="[^"]*">)')


class AssetLoadError(RuntimeError):
    """The asset could not be compiled. This is a Gate outcome, never a KF score."""


@dataclass(frozen=True, slots=True)
class LoadedAsset:
    """A compiled asset, plus every liberty taken to compile it."""

    record_id: str
    model: mujoco.MjModel
    data: mujoco.MjData
    source: Path
    inertia_synthesized: bool
    collidable_geoms: int
    """Geoms MuJoCo would consider for contact. Expected to be zero on Articraft assets;
    reported so that a future asset with real collision geometry is noticed rather than
    silently changing what the distance queries mean."""

    notes: tuple[str, ...] = ()

    @property
    def provenance(self) -> dict[str, object]:
        """What a report must carry so nobody mistakes our patches for the asset's."""
        return {
            "record_id": self.record_id,
            "source": str(self.source),
            "inertia_synthesized": self.inertia_synthesized,
            "collidable_geoms": self.collidable_geoms,
            "distance_backend": "mj_geomDistance",
            "notes": list(self.notes),
        }

    # -- naming ------------------------------------------------------------------------

    def body_id(self, name: str) -> int | None:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return None if i < 0 else i

    def joint_id(self, name: str) -> int | None:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return None if i < 0 else i

    def body_name(self, body_id: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"<body {body_id}>"

    def joint_name(self, joint_id: int) -> str:
        return (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            or f"<joint {joint_id}>"
        )

    def geoms_of(self, body_id: int) -> tuple[int, ...]:
        """Geom indices owned by one body.

        By index rather than by name: MuJoCo renames duplicates to unnamed geoms, and the
        real assets duplicate names freely (``knob_cap`` appears on several bodies of the
        same model). Names are not a reliable handle here; the body tree is.
        """
        start = int(self.model.body_geomadr[body_id])
        count = int(self.model.body_geomnum[body_id])
        return tuple(range(start, start + count)) if count > 0 else ()

    def subtree_bodies(self, body_id: int) -> tuple[int, ...]:
        """``body_id`` and every descendant, in index order."""
        out = [body_id]
        for b in range(body_id + 1, self.model.nbody):
            parent = int(self.model.body_parentid[b])
            if parent in out:
                out.append(b)
        return tuple(out)

    def rigid_subtree(self, body_id: int) -> tuple[int, ...]:
        """Bodies that move rigidly with ``body_id``: reachable without crossing a joint.

        This is what "rigidly attached" has to mean operationally. A handle that rides its
        drawer has no joint of its own, so it sits inside the drawer's rigid subtree; a
        handle bolted to the carcass instead does not, however similar the two look at the
        reference configuration.
        """
        out = [body_id]
        for b in range(body_id + 1, self.model.nbody):
            parent = int(self.model.body_parentid[b])
            if parent in out and int(self.model.body_jntnum[b]) == 0:
                out.append(b)
        return tuple(out)


def patch_urdf(text: str) -> tuple[str, bool]:
    """Return the URDF MuJoCo will accept, and whether inertia had to be invented."""
    patched = _ROBOT.sub(lambda m: m.group(1) + "\n" + COMPILER_DIRECTIVE, text, count=1)
    needs_inertia = "<inertial" not in text
    if needs_inertia:
        patched = _LINK.sub(lambda m: m.group(1) + "\n    " + SYNTHETIC_INERTIAL, patched)
    return patched, needs_inertia


def load(urdf_path: str | Path, *, record_id: str | None = None) -> LoadedAsset:
    """Compile one Articraft URDF.

    The patched file is written beside the original because MuJoCo resolves ``<mesh
    filename=...>`` relative to the model file, and the OBJ meshes sit in ``assets/meshes``
    next to it. It is removed again afterwards.
    """
    urdf_path = Path(urdf_path).resolve()
    if not urdf_path.exists():
        raise AssetLoadError(f"{urdf_path} does not exist")

    text = urdf_path.read_text(encoding="utf-8")
    patched, synthesized = patch_urdf(text)

    notes: list[str] = []
    if synthesized:
        notes.append(
            "no <inertial> in source; mass and inertia synthesised to satisfy the "
            "compiler. Never used in scoring."
        )
    if "<collision" not in text:
        notes.append(
            "no <collision> in source; all geoms are visual-only, so mjData.contact is "
            "always empty and every distance goes through mj_geomDistance."
        )

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".urdf", dir=urdf_path.parent, delete=False, encoding="utf-8"
        ) as fh:
            fh.write(patched)
            tmp = Path(fh.name)
        try:
            model = mujoco.MjModel.from_xml_path(str(tmp))
        except ValueError as exc:
            raise AssetLoadError(f"{urdf_path.parent.name}: {exc}") from exc
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)

    if model.ngeom == 0:
        raise AssetLoadError(
            f"{urdf_path.parent.name}: compiled to zero geoms, so no geometric predicate "
            f"can be evaluated"
        )

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    collidable = int(((model.geom_contype | model.geom_conaffinity) != 0).sum())
    return LoadedAsset(
        record_id=record_id or urdf_path.parent.name,
        model=model,
        data=data,
        source=urdf_path,
        inertia_synthesized=synthesized,
        collidable_geoms=collidable,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------------------
# geometry helpers shared by the KF predicates
# --------------------------------------------------------------------------------------


def geom_distance(asset: LoadedAsset, geom_a: int, geom_b: int, distmax: float) -> float:
    """Signed distance between two geoms: negative means interpenetration.

    ``distmax`` censors the result -- MuJoCo returns ``distmax`` for anything further
    apart -- so callers that report a minimum clearance must pass a bound large enough to
    contain the answer, and treat a returned value at the bound as "at least this far".
    """
    fromto = np.zeros(6)
    return float(mujoco.mj_geomDistance(asset.model, asset.data, geom_a, geom_b, distmax, fromto))


def body_pair_distance(
    asset: LoadedAsset, body_a: int, body_b: int, distmax: float
) -> tuple[float, tuple[int, int] | None]:
    """Closest approach between two bodies' geometry, and which geoms achieved it.

    Bodies rather than geoms because a part binds to a body: Articraft splits one part
    across many geoms (a drawer is a floor plus four walls), and the part-level question
    is whether *any* of them touches.
    """
    best = distmax
    which: tuple[int, int] | None = None
    for ga in asset.geoms_of(body_a):
        for gb in asset.geoms_of(body_b):
            d = geom_distance(asset, ga, gb, distmax)
            if d < best:
                best, which = d, (ga, gb)
    return best, which


def subtree_aabb(asset: LoadedAsset, bodies: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    """World-frame axis-aligned bounds of a set of bodies' geometry at the current pose.

    Uses ``geom_rbound``, MuJoCo's bounding-sphere radius, rather than the exact mesh:
    it is conservative, cheap, and available for every geom type including meshes. A
    conservative bound is the right side to err on here, since these AABBs gate whether
    a pair of degrees of freedom is swept jointly at all.
    """
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for b in bodies:
        for g in asset.geoms_of(b):
            centre = asset.data.geom_xpos[g]
            r = float(asset.model.geom_rbound[g])
            lo = np.minimum(lo, centre - r)
            hi = np.maximum(hi, centre + r)
    if not np.isfinite(lo).all():
        return np.zeros(3), np.zeros(3)
    return lo, hi


def aabb_diagonal(lo: np.ndarray, hi: np.ndarray) -> float:
    """Scale reference for normalising a distance.

    Always a *part's own* diagonal, never the whole asset's. Of 607 annotated records only
    19 could produce a whole-asset bounding-box diagonal, so any threshold expressed
    against one was uncomputable on 97% of the corpus.
    """
    return float(np.linalg.norm(np.asarray(hi) - np.asarray(lo)))
