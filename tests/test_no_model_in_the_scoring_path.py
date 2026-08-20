"""No model decides anything here.

Every verdict this package produces must come from arithmetic, set operations and
geometric queries on a compiled mjModel. That is not a preference; it is what makes a
score a function of the asset rather than a function of the asset and some model's state
on the day. The previous attempt at these metrics had an LLM extracting requirements from
prose prompts, and the same asset with reworded instructions could score differently.

The distinction that matters, and the reason these tests exist rather than a promise:

    An LLM may *write* this code, the way an engineer writes code. What it may never do
    is *run inside* it. The test of that is mechanical -- same input, same bytes out, no
    network, no inference -- so it is checked mechanically.

These tests fail if anyone ever imports an inference library, opens a socket, or makes the
evaluation non-reproducible.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "evo_p0p3"

FORBIDDEN_ROOTS = frozenset(
    {
        # inference clients
        "openai", "anthropic", "google", "genai", "cohere", "mistralai", "ollama",
        "replicate", "together", "groq", "litellm", "langchain", "llama_cpp",
        # model runtimes
        "torch", "tensorflow", "jax", "flax", "transformers", "sentence_transformers",
        "onnxruntime", "clip", "open_clip", "timm",
        # anything that could reach one
        "requests", "httpx", "aiohttp", "urllib", "urllib3", "http", "socket",
        "websockets", "ftplib", "telnetlib", "smtplib", "xmlrpc",
        # nondeterminism
        "random", "secrets", "uuid",
    }
)
"""Roots that must not appear in an import anywhere in the package.

``random`` is on the list alongside the inference clients on purpose. A score that moves
between runs is unfalsifiable for the same reason a score that depends on a model's mood
is: nobody can tell a real regression from noise. Deterministic sampling -- the Sobol fill
in the sweep, for instance -- must be constructed from a frozen seed, not drawn.
"""

ALLOWED_THIRD_PARTY = frozenset({"mujoco", "numpy", "yaml", "PIL"})

RENDERING_MODULES = frozenset({"render.py", "review.py"})
"""The only modules allowed to produce pixels.

Pictures are for people. They are made after scoring, from the configuration a frozen
result already recorded, and nothing downstream of them feeds a verdict. Keeping that
true needs a rule rather than an intention, which is the test below: no module that
decides anything may import an imaging library, so a picture can never quietly become
the reason a claim passed or failed. The prohibition matters because P3's own scope
forbids rendering as evidence -- it is why travel_scale measures a part's size instead
of watching it fail to move.
"""
"""``PIL`` is rendering only: pictures are evidence for a human reader, never an input to
a score."""


def python_files() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py"))


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_the_package_has_files_to_check():
    # Guards against the rest of this file passing vacuously.
    assert len(python_files()) >= 6


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_no_module_imports_an_inference_client_or_a_socket(path: Path):
    offending = imported_roots(path) & FORBIDDEN_ROOTS
    assert not offending, f"{path.name} imports {sorted(offending)}"


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_third_party_imports_stay_on_the_declared_list(path: Path):
    import sys

    roots = imported_roots(path)
    unexpected = {
        r
        for r in roots
        if r not in ALLOWED_THIRD_PARTY
        and r != "evo_p0p3"
        and r not in sys.stdlib_module_names
    }
    assert not unexpected, f"{path.name} imports undeclared third party: {sorted(unexpected)}"


@pytest.mark.parametrize("path", python_files(), ids=lambda p: p.name)
def test_only_the_rendering_modules_touch_an_imaging_library(path: Path):
    if path.name in RENDERING_MODULES:
        return
    assert "PIL" not in imported_roots(path), (
        f"{path.name} decides things, so it must not import an imaging library. "
        f"If it needs a picture, the picture belongs in one of {sorted(RENDERING_MODULES)}, "
        f"which run after scoring and cannot change a verdict."
    )


def test_the_rendering_modules_are_not_reachable_from_a_verdict(path=None):
    """Nothing in the scoring path may import the renderers either."""
    offenders = []
    for p in python_files():
        if p.name in RENDERING_MODULES:
            continue
        text = p.read_text(encoding="utf-8")
        for module in RENDERING_MODULES:
            stem = module[:-3]
            if f"import {stem}" in text or f"from evo_p0p3.p3.{stem}" in text:
                offenders.append(f"{p.name} -> {module}")
    assert not offenders, (
        f"a scoring module reaches a renderer: {offenders}. Pictures are produced from "
        f"frozen results, never consulted while producing them."
    )


def test_no_source_line_calls_out_to_a_network():
    # Catches the string-built import and the stray URL that an import scan would miss.
    needles = ("http://", "https://", "api_key", "API_KEY", "bearer ", "Bearer ")
    offenders = []
    for path in python_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(n in line for n in needles):
                offenders.append(f"{path.name}:{lineno}")
    assert not offenders, offenders


def test_the_declared_dependencies_cannot_run_a_model():
    text = (PACKAGE.parent / "pyproject.toml").read_text(encoding="utf-8")
    body = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    declared = {
        line.strip().strip('",').split(">=")[0].split("==")[0].split("[")[0]
        for line in body.splitlines()
        if line.strip().startswith('"')
    }
    assert declared <= {"mujoco", "numpy", "pyyaml"}, declared


# --------------------------------------------------------------------------------------
# reproducibility: the same input must give the same bytes
# --------------------------------------------------------------------------------------


def test_admission_findings_are_identical_across_runs():
    import yaml

    from evo_p0p3.p0 import admission
    from evo_p0p3.p0.loader import parse_contract

    raw = yaml.safe_load(
        (PACKAGE.parent / "contracts" / "gold_cabinet.yaml").read_text(encoding="utf-8")
    )
    runs = []
    for _ in range(3):
        report = admission.check(parse_contract(raw, record_id="gold"))
        runs.append(json.dumps([str(f) for f in report.findings], sort_keys=True))
    assert len(set(runs)) == 1


def test_the_tolerance_digest_is_stable_across_runs():
    import yaml

    from evo_p0p3.p0.loader import parse_contract

    raw = yaml.safe_load(
        (PACKAGE.parent / "contracts" / "gold_cabinet.yaml").read_text(encoding="utf-8")
    )
    digests = {
        parse_contract(raw, record_id="g").kinematic_claims.tolerances.digest()
        for _ in range(3)
    }
    assert len(digests) == 1


def test_geometry_measurements_repeat_exactly(tmp_path: Path):
    # The compiled model, the forward kinematics and the distance queries must all be
    # bit-reproducible, or no reported failure can be argued with.
    from tests.test_p3_mjcf import CABINET
    from evo_p0p3.p3 import mjcf

    path = tmp_path / "model.urdf"
    path.write_text(CABINET, encoding="utf-8")

    readings = []
    for _ in range(3):
        asset = mjcf.load(path, record_id="cabinet")
        body = asset.body_id("cabinet_body")
        drawer = asset.body_id("drawer")
        distance, which = mjcf.body_pair_distance(asset, body, drawer, distmax=2.0)
        lo, hi = mjcf.subtree_aabb(asset, (drawer,))
        readings.append(
            json.dumps(
                {
                    "distance": repr(distance),
                    "geoms": which,
                    "aabb": [repr(float(v)) for v in (*lo, *hi)],
                    "provenance": {
                        k: v for k, v in asset.provenance.items() if k != "source"
                    },
                },
                sort_keys=True,
            )
        )
    assert len(set(readings)) == 1
