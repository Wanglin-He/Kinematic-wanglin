"""Pictures of the configuration a claim was decided at, for human review.

Nothing here feeds a score. Every number in a report comes from ``mj_geomDistance`` and the
arithmetic in kf1/kf2/kf3; this module only re-poses the asset at the configuration those
predicates already recorded and takes a picture of it, so a person can look at the same
thing the arithmetic looked at. P3's scope forbids rendering *as evidence for a verdict* --
that prohibition is what ``travel_scale`` exists to work around -- and this respects it:
the renderer runs after scoring, reads the frozen result, and cannot change it.

Two pictures per failing claim, from one camera: the reference pose and the configuration
that failed. Same viewpoint for both, because the thing worth seeing is the difference.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import mujoco
import numpy as np

GREY = (0.74, 0.74, 0.72, 1.0)
"""Assets carry their own materials, and a black laptop on a black background is not
evidence of anything. Everything is repainted a neutral grey and only the parts a claim
names get a colour.

The grey is nearly transparent because most interpenetrations are internal -- a shaft
inside a sleeve, a pusher inside a basket -- and an opaque exterior render shows a clean
appliance no matter how deep the overlap. Making the unnamed parts translucent keeps the
context that says which appliance this is, while letting the named pair be seen through it.
"""

HIGHLIGHT = ((0.86, 0.24, 0.16, 1.0), (0.13, 0.42, 0.78, 1.0), (0.92, 0.68, 0.11, 1.0))

WIDTH, HEIGHT = 460, 340


@dataclass(frozen=True, slots=True)
class Shot:
    reference_png: str
    """base64 PNG at the reference pose."""
    failing_png: str
    """base64 PNG at the configuration the claim was decided at."""
    caption: str
    highlighted: tuple[str, ...]


def _paint(model, binding, parts: tuple[str, ...]) -> None:
    model.geom_matid[:] = -1
    model.geom_rgba[:] = GREY
    # Translucency needs back-to-front sorting to read correctly; MuJoCo does that per
    # geom, so the only thing required here is that the named parts stay fully opaque.
    for colour, part in zip(HIGHLIGHT, parts, strict=False):
        for body in binding.bodies(part):
            for g in range(model.ngeom):
                if int(model.geom_bodyid[g]) == body:
                    model.geom_rgba[g] = colour


def _frame(model, data, binding, parts: tuple[str, ...]):
    """Point the camera at the parts the claim names, not at the whole asset.

    A 3 mm interpenetration inside a 0.4 m appliance is invisible when the camera frames
    the appliance. Framing the named parts is what makes the picture worth including.
    """
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for part in parts:
        for body in binding.bodies(part):
            for g in range(model.ngeom):
                if int(model.geom_bodyid[g]) != body:
                    continue
                centre = np.asarray(data.geom_xpos[g], dtype=float)
                radius = float(model.geom_rbound[g]) or 0.01
                lo = np.minimum(lo, centre - radius)
                hi = np.maximum(hi, centre + radius)
    if not np.isfinite(lo).all():
        return np.asarray(model.stat.center, dtype=float), 1.3 * float(model.stat.extent)
    span = float(np.linalg.norm(hi - lo))
    return (lo + hi) / 2.0, max(span * 1.7, 0.05)


def _png(renderer, model, data, camera, option) -> str:
    from PIL import Image

    renderer.update_scene(data, camera, option)
    image = renderer.render()
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def surround(asset, binding, qpos, *, angles=(0, 90, 180, 270)) -> tuple[str, ...]:
    """Four views of the whole asset at one pose, evenly around it.

    The same four angles for every asset, so two records can be compared by eye without
    wondering whether the camera moved. Nothing here is measured; the measurements are
    already in the report and appear next to these as text.
    """
    from PIL import Image

    model, data = asset.model, asset.data
    model.geom_matid[:] = -1
    model.geom_rgba[:] = GREY
    model.vis.headlight.ambient[:] = [0.48, 0.48, 0.48]
    model.vis.headlight.diffuse[:] = [0.78, 0.78, 0.78]
    model.vis.headlight.specular[:] = [0.10, 0.10, 0.10]

    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray(model.stat.center, dtype=float)
    camera.distance = 1.25 * float(model.stat.extent)
    camera.elevation = -20.0
    option = mujoco.MjvOption()

    out = []
    for angle in angles:
        camera.azimuth = float(angle)
        renderer.update_scene(data, camera, option)
        buffer = io.BytesIO()
        Image.fromarray(renderer.render()).save(buffer, format="PNG", optimize=True)
        out.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    renderer.close()
    return tuple(out)


def shoot(
    asset, binding, reference_qpos, failing_qpos, parts: tuple[str, ...],
    *, caption: str, azimuth: float = 135.0, show_joints: bool = False,
) -> Shot:
    """Both poses through one camera, framed on ``parts``."""
    model, data = asset.model, asset.data
    _paint(model, binding, parts)
    model.vis.headlight.ambient[:] = [0.45, 0.45, 0.45]
    model.vis.headlight.diffuse[:] = [0.80, 0.80, 0.80]
    model.vis.headlight.specular[:] = [0.10, 0.10, 0.10]
    option = mujoco.MjvOption()
    if show_joints:
        # An anchor claim is entirely about where the axis sits, so draw the axis. This is
        # MuJoCo's own joint visualisation, not an overlay computed here.
        option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        model.vis.scale.jointlength = 1.1
        model.vis.scale.jointwidth = 0.045
        model.vis.rgba.joint[:] = [0.10, 0.85, 0.35, 0.95]

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = azimuth
    camera.elevation = -18.0

    # The camera is framed on the failing pose and reused for the reference one, so the two
    # pictures differ only by the joint that moved.
    data.qpos[:] = failing_qpos
    mujoco.mj_forward(model, data)
    lookat, distance = _frame(model, data, binding, parts)
    camera.lookat[:] = lookat
    camera.distance = distance
    failing = _png(renderer, model, data, camera, option)

    data.qpos[:] = reference_qpos
    mujoco.mj_forward(model, data)
    reference = _png(renderer, model, data, camera, option)

    renderer.close()
    return Shot(reference_png=reference, failing_png=failing,
                caption=caption, highlighted=parts)
