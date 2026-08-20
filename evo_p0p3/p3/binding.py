"""Which bodies a declared part is.

This is the Gate's output, and P3 only reads it. That division is the reason this project
exists in its current shape: when the evaluator resolves names itself, "part not found"
means either that the asset is missing a part or that the contract used a name nothing
binds to, and the previous run showed those cannot be separated afterwards -- the flag
produced 42 of 63 false alarms while carrying 4 of 7 true positives, and a paired bootstrap
put the achievable improvement's confidence interval across zero.

Two ways in, and a report always says which was used:

* an explicit table, one line per part, written by the Gate or by hand for a pilot;
* identity, where a part id is a link name, which is true of the gold assets by
  construction and must never be assumed of a generated one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml

from evo_p0p3.p3.mjcf import LoadedAsset


class BindingSource(StrEnum):
    TABLE = "table"
    """An explicit part-to-body table. The only source a real evaluation may use."""

    IDENTITY = "identity"
    """Part ids taken as link names. Valid for hand-authored assets, where we chose the
    names; never valid for a generated asset, where matching a name is the very inference
    the architecture removes."""


class BindingError(RuntimeError):
    """A declared part has no body. Never a KF score -- the asset failed Gate G1."""


@dataclass(frozen=True, slots=True)
class Binding:
    """Part id to body ids, plus where the mapping came from."""

    parts: dict[str, tuple[int, ...]]
    source: BindingSource
    asset: LoadedAsset

    def bodies(self, part: str) -> tuple[int, ...]:
        if part not in self.parts:
            raise BindingError(
                f"part {part!r} has no binding. A part that reached P3 without one means "
                f"the Gate admitted an asset it should have failed at G1."
            )
        return self.parts[part]

    def root_body(self, part: str) -> int:
        """The body of a part that is an ancestor of its other bodies.

        A part is often several bodies -- a drawer is a floor and four walls -- and the
        claims about it (its parent, what it rides, where its joint sits) are claims about
        the one the others hang from.
        """
        bodies = self.bodies(part)
        if len(bodies) == 1:
            return bodies[0]
        depth = {b: self._depth(b) for b in bodies}
        return min(bodies, key=lambda b: (depth[b], b))

    def _depth(self, body: int) -> int:
        model, d = self.asset.model, 0
        while body != 0:
            body = int(model.body_parentid[body])
            d += 1
        return d

    def part_of(self, body: int) -> str | None:
        for part, bodies in self.parts.items():
            if body in bodies:
                return part
        return None

    def nearest_declared_ancestor(self, body: int) -> tuple[str | None, int]:
        """The first declared part strictly above ``body``, and how many links up it is.

        This is what a parent claim is actually about. A drawer whose body hangs off
        another drawer's body has ``cabinet_body`` somewhere up its chain too, so merely
        asking "is the declared parent an ancestor" passes an asset where the upper drawer
        drags the lower one along.
        """
        model = self.asset.model
        steps = 0
        current = int(model.body_parentid[body])
        while True:
            steps += 1
            part = self.part_of(current)
            if part is not None:
                return part, steps
            if current == 0:
                return None, steps
            current = int(model.body_parentid[current])


def identity(asset: LoadedAsset, part_ids: tuple[str, ...]) -> Binding:
    """Bind each part id to the link of the same name, plus its jointless descendants.

    The descendants matter: a part authored as several links joined by ``fixed`` joints is
    one rigid object, and its geometry is spread across all of them.
    """
    parts: dict[str, tuple[int, ...]] = {}
    missing = []
    for pid in part_ids:
        root = asset.body_id(pid)
        if root is None:
            missing.append(pid)
            continue
        parts[pid] = (root,)
    if missing:
        raise BindingError(
            f"no link named {missing} in {asset.record_id}. Under identity binding a part "
            f"id must be a link name exactly; there is no fuzzy fallback by design."
        )
    return Binding(parts=parts, source=BindingSource.IDENTITY, asset=asset)


def from_table(asset: LoadedAsset, path: str | Path) -> Binding:
    """Read a Gate binding table: ``{part_id: [link name, ...]}``."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    parts: dict[str, tuple[int, ...]] = {}
    problems = []
    for pid, names in raw.items():
        names = [names] if isinstance(names, str) else list(names)
        ids = []
        for name in names:
            bid = asset.body_id(str(name))
            if bid is None:
                problems.append(f"{pid} -> {name!r} (no such link)")
            else:
                ids.append(bid)
        if ids:
            parts[str(pid)] = tuple(ids)
    if problems:
        raise BindingError(
            f"binding table names links the model does not have: {problems}. This is a "
            f"Gate failure, not an asset defect."
        )
    return Binding(parts=parts, source=BindingSource.TABLE, asset=asset)
