"""Where a threshold belongs, read off a curve instead of guessed.

Every tolerance in the contract started as a protocol value somebody wrote down. That is
fine as a starting point and useless as an answer, and the pilot showed why: KF1.anchor
fired on five of ten real assets and on ten of the twelve hinges it could evaluate, which
is not a credible defect rate.

The calibration here is deterministic fault injection. Take a hinge known to be correct --
the gold cabinet's door, hinged exactly on its edge -- and walk it toward the middle of the
panel by a known fraction of the panel's width. At each step, measure. The result is a
curve from "how wrong is it" to "what does the predicate read", and a threshold is a point
on that curve rather than an opinion about one.

The alternative, tuning until the pilot numbers look reasonable, fits the instrument to the
sample. It would also destroy the only property the pilot has: the code was frozen before
it saw those assets.
"""

from __future__ import annotations

import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from evo_p0p3.p3 import binding as binding_mod
from evo_p0p3.p3 import gold, mjcf

DOOR_WIDTH = 0.34
"""The gold cabinet door's span from its hinge, in metres. Displacements below are
fractions of this, so the curve is scale-free."""


@dataclass(frozen=True, slots=True)
class Reading:
    """What both candidate measures say at one known displacement."""

    offset_fraction: float
    """How far the axis was moved toward the middle, as a fraction of the part's width.
    0.0 is a hinge exactly on the edge; 0.5 is a hinge through the centre."""

    side_fraction: float
    """The measure KF1.anchor uses today: the larger share of vertices on one side."""

    inset_fraction: float
    """The proposed measure: how far inside the part the axis sits, as a fraction of the
    part's extent perpendicular to it. Directly comparable to offset_fraction."""

    vertices: int


def _displaced_door(offset: float) -> str:
    """The gold cabinet with its hinge moved ``offset`` metres toward the panel's middle.

    The panel is held in place and only the axis moves, so the reading changes for exactly
    one reason. Moving both would measure nothing.
    """
    text = gold.correct_urdf("cabinet_correct.urdf")
    hinge_x = -0.17 + offset
    panel_x = 0.17 - offset
    text = text.replace(
        '<origin xyz="-0.17 -0.19 0.21"/>', f'<origin xyz="{hinge_x:.6f} -0.19 0.21"/>'
    )
    text = re.sub(
        r'(<visual name="panel">\s*<origin xyz=")0\.17( 0 0"/>)',
        lambda m: m.group(1) + f"{panel_x:.6f}" + m.group(2),
        text,
    )
    return text


def _measure(asset: mjcf.LoadedAsset, bound: binding_mod.Binding) -> tuple[float, float, int]:
    """Both measures for the cabinet's door hinge, at the current pose."""
    from evo_p0p3.p3.kf1 import _points_of

    mujoco.mj_forward(asset.model, asset.data)
    joint = asset.joint_id("door_hinge")
    body = bound.root_body("door")
    points = _points_of(bound, (body,))

    p0 = np.asarray(asset.data.xanchor[joint], dtype=float)
    axis = np.asarray(asset.data.xaxis[joint], dtype=float)
    axis = axis / (np.linalg.norm(axis) or 1.0)
    radial = np.array([v - np.dot(v, axis) * axis for v in (points - p0)])

    # -- today's measure: the larger share of vertices on one side ---------------------
    centre = radial.mean(axis=0)
    norm = float(np.linalg.norm(centre))
    if norm > 1e-9:
        direction = centre / norm
    else:
        _, _, vh = np.linalg.svd(radial - radial.mean(axis=0), full_matrices=False)
        direction = vh[0] - np.dot(vh[0], axis) * axis
        direction = direction / (np.linalg.norm(direction) or 1.0)
    s = radial @ direction
    positive = int((s > 1e-6).sum())
    negative = int((s < -1e-6).sum())
    on_axis = len(s) - positive - negative
    side_fraction = max(positive + on_axis, negative + on_axis) / len(s)

    # -- proposed measure: how far inside the part the axis sits ----------------------
    # The part spans [min(s), max(s)] along the split direction. An axis on the edge sits
    # at one end; an axis through the middle sits halfway. Reported as the smaller share,
    # so 0.0 is an edge hinge and 0.5 is a centre hinge -- directly comparable to the
    # displacement that produced it, which a vertex count is not.
    lo, hi = float(s.min()), float(s.max())
    span = hi - lo
    inset = min(abs(0.0 - lo), abs(hi - 0.0)) / span if span > 1e-9 else 0.0
    return side_fraction, inset, len(s)


def sweep_hinge(fractions: tuple[float, ...] | None = None) -> tuple[Reading, ...]:
    """Walk the gold door's hinge from its edge to its centre, measuring at each step."""
    if fractions is None:
        fractions = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)

    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in fractions:
            path = Path(tmp) / f"door_{f:.3f}.urdf"
            path.write_text(_displaced_door(f * DOOR_WIDTH), encoding="utf-8")
            asset = mjcf.load(path, record_id=f"offset_{f:.3f}")
            bound = binding_mod.identity(
                asset,
                ("cabinet_body", "drawer_1", "drawer_2", "door",
                 "handle_1", "handle_2", "door_knob"),
            )
            side, inset, n = _measure(asset, bound)
            out.append(Reading(offset_fraction=f, side_fraction=side,
                               inset_fraction=inset, vertices=n))
    return tuple(out)


def knee(readings: tuple[Reading, ...], measure: str) -> str:
    """A short description of how the measure behaves, for the report."""
    values = [getattr(r, measure) for r in readings]
    steps = [abs(values[i + 1] - values[i]) for i in range(len(values) - 1)]
    biggest = max(range(len(steps)), key=lambda i: steps[i])
    return (
        f"largest single step {steps[biggest]:.3f} between offset "
        f"{readings[biggest].offset_fraction:.0%} and "
        f"{readings[biggest + 1].offset_fraction:.0%}"
    )
