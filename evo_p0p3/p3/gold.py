"""Materialising the gold-standard assets from one correct file plus a list of edits.

A predicate cannot be argued into correctness. It can only be shown to fire on an input
whose answer is known, and to stay quiet on one that is right -- and the only inputs whose
answers are known are the ones we broke ourselves. Real assets do not come with their true
anchor quality recorded anywhere, and the specific disease this project keeps catching --
a predicate no model can fail -- is invisible against them, because a tautology passing
everything looks exactly like a corpus of good assets.

So each defect is stored as the smallest textual edit that breaks the correct file, and
the substitution is required to actually change the compiled model. An edit that silently
matches nothing would leave a "counterexample" identical to the control, and the predicate
tested against it would look sound while checking nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

GOLD_DIR = Path(__file__).resolve().parents[2] / "assets" / "gold"


class DefectNotApplied(RuntimeError):
    """A substitution matched nothing, so the 'defective' asset is the correct one."""


@dataclass(frozen=True, slots=True)
class Defect:
    name: str
    targets: str
    defect: str
    urdf: str
    expect: dict[str, str]

    @property
    def expected_failures(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.expect.items() if v == "fail")

    @property
    def expected_passes(self) -> tuple[str, ...]:
        return tuple(k for k, v in self.expect.items() if v == "pass")


def _manifest() -> dict:
    return yaml.safe_load((GOLD_DIR / "manifest.yaml").read_text(encoding="utf-8"))


def correct_urdf() -> str:
    return (GOLD_DIR / _manifest()["base"]).read_text(encoding="utf-8")


def _apply(text: str, substitutions: list[dict[str, str]], name: str) -> str:
    for i, sub in enumerate(substitutions):
        find = sub["find"].strip("\n")
        replace = sub["replace"].strip("\n")
        # Compare on collapsed whitespace so the manifest can be indented for reading
        # while still matching a file indented differently.
        needle = _normalise(find)
        haystack = _normalise(text)
        if needle not in haystack:
            raise DefectNotApplied(
                f"{name}: substitution {i} matched nothing. The defect would be identical "
                f"to the control, and any predicate tested against it would look sound "
                f"while checking nothing.\n  looking for: {find[:120]!r}"
            )
        text = _replace_normalised(text, find, replace)
    return text


def _normalise(text: str) -> str:
    return " ".join(text.split())


def _replace_normalised(text: str, find: str, replace: str) -> str:
    """Replace ``find`` in ``text`` ignoring how either is indented.

    Walks the source once, comparing whitespace-collapsed windows, so a manifest written
    with readable indentation still matches a URDF indented some other way.
    """
    target = _normalise(find)
    tokens = target.split()
    if not tokens:
        return text

    words = []  # (start, end) of every whitespace-delimited token in text
    i = 0
    while i < len(text):
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < len(text) and not text[j].isspace():
            j += 1
        words.append((i, j))
        i = j

    n = len(tokens)
    for k in range(len(words) - n + 1):
        window = " ".join(text[a:b] for a, b in words[k : k + n])
        if window == target:
            start = words[k][0]
            end = words[k + n - 1][1]
            return text[:start] + replace.strip() + text[end:]
    raise DefectNotApplied(f"substitution matched nothing: {find[:120]!r}")


def defects() -> tuple[Defect, ...]:
    """Every defective asset, with its URDF text already materialised."""
    manifest = _manifest()
    base = correct_urdf()
    out = []
    for entry in manifest["defects"]:
        out.append(
            Defect(
                name=entry["name"],
                targets=entry["targets"],
                defect=entry["defect"].strip(),
                urdf=_apply(base, entry["substitute"], entry["name"]),
                expect=dict(entry.get("expect") or {}),
            )
        )
    return tuple(out)


def defect(name: str) -> Defect:
    for d in defects():
        if d.name == name:
            return d
    raise KeyError(f"no gold defect named {name!r}; have {[d.name for d in defects()]}")


def write_all(out_dir: str | Path) -> dict[str, Path]:
    """Write the control and every defect to disk, for inspection or for a viewer."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {"cabinet_correct": out_dir / "cabinet_correct.urdf"}
    written["cabinet_correct"].write_text(correct_urdf(), encoding="utf-8")
    for d in defects():
        path = out_dir / f"cabinet_{d.name}.urdf"
        path.write_text(d.urdf, encoding="utf-8")
        written[d.name] = path
    return written
