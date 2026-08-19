"""The ledger against the original specification stays honest.

The final deliverable is a document explaining how each item of task-2_08-16-p0.html and
task-2_08-16-p3.html was solved. That document is only worth reading if the answers were
recorded as the work happened rather than reconstructed afterwards, so the ledger lives in
the repository and these tests keep it complete.

What they enforce is narrow and deliberate: every departure from the spec must carry a
reason, and every claim to have implemented something must point at code that exists. They
cannot check that a reason is a *good* one -- that is a reading, not a test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "docs" / "traceability.yaml"

DISPOSITIONS = {
    "literal", "clarified", "replaced", "changed",
    "added", "decided", "deferred", "external",
}
NEEDS_REASON = {"replaced", "changed", "added", "decided"}
NEEDS_REPLACEMENT = {"replaced", "changed", "decided"}
BUILT = {"literal", "clarified", "replaced", "changed", "added"}
"""Dispositions that assert code exists. `decided` deliberately does not: it records a
settled argument whose implementation is still owed, which is the honest state for a
spec bullet ruled out before its replacement is written."""


@pytest.fixture(scope="module")
def ledger() -> dict:
    return yaml.safe_load(LEDGER.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def items(ledger) -> list[dict]:
    return ledger["items"]


def test_every_item_has_an_id_a_spec_quote_and_a_disposition(items):
    for item in items:
        assert item.get("id"), item
        assert item.get("spec"), item["id"]
        assert item.get("disposition") in DISPOSITIONS, item["id"]


def test_ids_are_unique(items):
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), [x for x in ids if ids.count(x) > 1]


def test_a_departure_from_the_spec_must_say_why(items):
    # This is the rule that keeps the ledger from becoming a list of assertions. Deciding
    # a spec bullet cannot be implemented is a strong claim and has to be argued in place.
    missing = [
        i["id"]
        for i in items
        if i["disposition"] in NEEDS_REASON and not str(i.get("reason", "")).strip()
    ]
    assert not missing, f"no reason given for: {missing}"


def test_a_departure_must_say_what_stands_in_its_place(items):
    missing = [
        i["id"]
        for i in items
        if i["disposition"] in NEEDS_REPLACEMENT and not str(i.get("replaced_by", "")).strip()
    ]
    assert not missing, f"no replacement named for: {missing}"


def test_reasons_are_arguments_rather_than_labels(items):
    # A one-line reason is almost always a restatement of the disposition. The bar is low
    # on purpose; it only catches placeholders.
    thin = [
        i["id"]
        for i in items
        if i["disposition"] in NEEDS_REASON and len(str(i.get("reason", ""))) < 80
    ]
    assert not thin, f"reason too thin to be an argument: {thin}"


def test_every_referenced_path_exists(items):
    missing = []
    for item in items:
        where = item.get("where")
        if not where:
            continue
        path = ROOT / where.split("::")[0]
        if not path.exists():
            missing.append((item["id"], where))
    assert not missing, missing


def test_implemented_items_point_at_code(items):
    # An item claiming to be done without naming where is not checkable by anyone.
    done_without_location = [
        i["id"]
        for i in items
        if i["disposition"] in BUILT
        and not i.get("where")
        and not i.get("note")
    ]
    assert not done_without_location, done_without_location


def test_both_specs_are_covered(items):
    families = {i["id"].split(".")[0] for i in items}
    assert {"P0", "P3"} <= families


def test_the_ledger_covers_every_kf_bullet(items):
    # The three metrics are the deliverable; none of their bullets may be missing from
    # the ledger, whatever their disposition.
    required = {
        "P3.KF1.bullet1", "P3.KF1.bullet2", "P3.KF1.bullet3", "P3.KF1.bullet4",
        "P3.KF1.formula",
        "P3.KF2.bullet1", "P3.KF2.bullet2", "P3.KF2.bullet3", "P3.KF2.bullet4",
        "P3.KF2.formula", "P3.KF2.na",
        "P3.KF3.motion1", "P3.KF3.motion2", "P3.KF3.motion3",
        "P3.KF3.collision1", "P3.KF3.collision2", "P3.KF3.collision3",
        "P3.KF3.formula",
    }
    assert required <= {i["id"] for i in items}, required - {i["id"] for i in items}


def test_the_scope_boundary_is_recorded_as_literal(items):
    # If P3's scope ever drifts -- if mass, friction or a rollout creeps in -- it must be
    # a visible decision rather than an accident.
    scope = next(i for i in items if i["id"] == "P3.scope")
    assert scope["disposition"] == "literal"


def test_progress_is_reported_honestly(items, capsys):
    from collections import Counter

    counts = Counter(i["disposition"] for i in items)
    with capsys.disabled():
        print("\n  ledger:", dict(sorted(counts.items())), f"total={len(items)}")
    assert counts["deferred"] >= 0  # the count is the point, not a threshold
