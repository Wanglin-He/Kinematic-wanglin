"""Pictures of what the verifier is looking at.

Not decoration. A KF3 failure reports a first failing configuration, and "``door_hinge``
at q = 1.21 rad interpenetrates ``drawer_1`` by 4 mm" is far easier to act on -- and far
easier to *disbelieve correctly* -- next to a frame showing it. The previous project's
central finding was that annotators judge how the mesh moves rather than what a field
declares; if the evaluator is going to argue with a human verdict it should show its work
in the same currency.

Rendering is offscreen, so no window and no GUI are involved.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from evo_p0p3.p3.mjcf import LoadedAsset, aabb_diagonal, subtree_aabb


def _frame_camera(asset: LoadedAsset, azimuth: float, elevation: float) -> mujoco.MjvCamera:
    """Point a free camera at the asset and back off far enough to contain it."""
    lo, hi = subtree_aabb(asset, tuple(range(asset.model.nbody)))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (lo + hi) / 2.0
    cam.distance = max(aabb_diagonal(lo, hi) * 1.35, 1e-3)
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def render_pose(
    asset: LoadedAsset,
    qpos: np.ndarray | None = None,
    *,
    width: int = 480,
    height: int = 360,
    azimuth: float = 135.0,
    elevation: float = -20.0,
) -> np.ndarray:
    """One RGB frame at a given configuration."""
    if qpos is not None:
        asset.data.qpos[:] = qpos
    mujoco.mj_forward(asset.model, asset.data)
    cam = _frame_camera(asset, azimuth, elevation)
    with mujoco.Renderer(asset.model, height=height, width=width) as renderer:
        renderer.update_scene(asset.data, camera=cam)
        return renderer.render()


def contact_sheet(
    asset: LoadedAsset,
    poses: list[tuple[str, np.ndarray]],
    out_path: str | Path,
    *,
    columns: int = 4,
    width: int = 400,
    height: int = 300,
) -> Path:
    """Render several configurations into one labelled grid.

    Labels are drawn only if Pillow can find a font; the frames are the point, and a
    missing font should not cost us the picture.
    """
    from PIL import Image, ImageDraw

    frames = [(label, render_pose(asset, q, width=width, height=height)) for label, q in poses]
    rows = (len(frames) + columns - 1) // columns
    band = 22
    sheet = Image.new("RGB", (columns * width, rows * (height + band)), "white")
    draw = ImageDraw.Draw(sheet)

    for i, (label, frame) in enumerate(frames):
        x = (i % columns) * width
        y = (i // columns) * (height + band)
        sheet.paste(Image.fromarray(frame), (x, y + band))
        draw.text((x + 6, y + 5), label, fill="black")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def sweep_poses(asset: LoadedAsset, joint_name: str, samples: int = 4) -> list[tuple[str, np.ndarray]]:
    """Configurations stepping one joint across its range, others at their reference.

    A joint with no limits (URDF ``continuous``, which compiles to a hinge with
    ``jnt_limited = 0``) has no range to step through, so it gets one turn of the wheel --
    enough to see whether anything moves.
    """
    model = asset.model
    jid = asset.joint_id(joint_name)
    if jid is None:
        raise KeyError(f"no joint named {joint_name!r}")

    adr = int(model.jnt_qposadr[jid])
    if model.jnt_limited[jid]:
        lo, hi = (float(v) for v in model.jnt_range[jid])
    else:
        lo, hi = 0.0, 2.0 * np.pi

    out = []
    for value in np.linspace(lo, hi, samples):
        q = np.array(model.qpos0, copy=True)
        q[adr] = value
        out.append((f"{joint_name} = {value:+.3f}", q))
    return out
