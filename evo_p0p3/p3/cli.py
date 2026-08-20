"""``p3 run`` -- evaluate assets against a frozen contract.

    p3 run <contract.yaml> <asset.urdf> [<asset.urdf> ...] [--json out/]
    p3 gold                                     evaluate the gold-standard set
    p3 sweep <contract.yaml> <asset.urdf>       print the schedule without scoring

The contract is validated before anything is measured. A contract that fails admission
cannot produce a meaningful score, and running anyway would attribute a contract-authoring
mistake to the asset -- the confusion this whole architecture exists to prevent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evo_p0p3.p0 import admission as p0_admission
from evo_p0p3.p0.loader import ContractSyntaxError, load_contract
from evo_p0p3.p3 import gate, gold, kf1, kf2, kf3, report
from evo_p0p3.p3.sweep import Sweeper


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="p3", description="Kinematic fidelity evaluation")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="evaluate assets against a contract")
    run.add_argument("contract", type=Path)
    run.add_argument("assets", nargs="+", type=Path)
    run.add_argument("--json", type=Path, help="write one report per asset here")
    run.add_argument("-v", "--verbose", action="store_true", help="also list abstentions")

    g = sub.add_parser("gold", help="evaluate the gold-standard set")
    g.add_argument("--json", type=Path)
    g.add_argument("-v", "--verbose", action="store_true")

    sweep_cmd = sub.add_parser("sweep", help="print the sweep schedule without scoring")
    sweep_cmd.add_argument("contract", type=Path)
    sweep_cmd.add_argument("asset", type=Path)

    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args.contract, args.assets, args.json, args.verbose)
    if args.command == "gold":
        return _gold(args.json, args.verbose)
    if args.command == "sweep":
        return _sweep(args.contract, args.asset)
    return 2  # pragma: no cover


def _load_contract(path: Path):
    try:
        contract = load_contract(path)
    except ContractSyntaxError as exc:
        print(f"{path}: SYNTAX ERROR\n  {exc}", file=sys.stderr)
        return None
    report_ = p0_admission.check(contract)
    if not report_.admitted:
        print(f"{path}: {report_.summary()}", file=sys.stderr)
        for finding in report_.errors:
            print(f"  ✗ {finding}", file=sys.stderr)
        print(
            "\nA contract that cannot be checked cannot produce a score. Fix it before "
            "evaluating, or the asset gets blamed for the contract.",
            file=sys.stderr,
        )
        return None
    return contract


def evaluate(contract, asset_path: Path) -> report.AssetReport:
    """One asset, end to end."""
    admitted = gate.admit(asset_path, contract)
    if not admitted.admitted or admitted.binding is None:
        return report.build(contract, admitted)

    binding = admitted.binding
    results = kf1.evaluate(contract, binding) + kf2.evaluate(contract, binding)
    session = kf3.Session(contract, binding)
    results = results + session.evaluate()
    return report.build(contract, admitted, results, session.schedule)


def _emit(reports, out_dir: Path | None, verbose: bool) -> int:
    for r in reports:
        print(r.render(verbose=verbose))
        print()
        if out_dir:
            r.write(out_dir / f"{r.record_id}.json")

    run = report.RunReport(tuple(reports))
    print("=" * 78)
    print(run.render())
    if out_dir:
        (out_dir / "run.json").write_text(
            __import__("json").dumps(run.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nwrote {len(reports) + 1} file(s) to {out_dir}")
    return 0 if all(not r.failures for r in reports) else 1


def _run(contract_path: Path, assets: list[Path], out_dir: Path | None, verbose: bool) -> int:
    contract = _load_contract(contract_path)
    if contract is None:
        return 2
    return _emit([evaluate(contract, a) for a in assets], out_dir, verbose)


def _gold(out_dir: Path | None, verbose: bool) -> int:
    """Run the whole gold set, each family against its own contract."""
    import tempfile

    root = Path(__file__).resolve().parents[2]
    families = {
        "cabinet_correct.urdf": root / "contracts" / "gold_cabinet.yaml",
        "gearbox_correct.urdf": root / "contracts" / "gold_gearbox.yaml",
    }
    written = gold.write_all(Path(tempfile.mkdtemp(prefix="gold-")))

    reports = []
    for base, contract_path in families.items():
        contract = _load_contract(contract_path)
        if contract is None:
            return 2
        stem = Path(base).stem
        reports.append(evaluate(contract, written[stem]))
        for defect in gold.defects(base):
            reports.append(evaluate(contract, written[defect.name]))
    return _emit(reports, out_dir, verbose)


def _sweep(contract_path: Path, asset_path: Path) -> int:
    contract = _load_contract(contract_path)
    if contract is None:
        return 2
    admitted = gate.admit(asset_path, contract)
    if admitted.binding is None:
        print(admitted.summary(), file=sys.stderr)
        return 1
    schedule = Sweeper(contract, admitted.binding).schedule()
    print(f"{admitted.record_id}: {schedule.size} configurations")
    for layer, count in schedule.by_layer().items():
        print(f"  {layer:10s} {count}")
    print(f"  driven    {list(schedule.driven)}")
    print(f"  dependent {list(schedule.dependent)}")
    print(f"  pairs swept  {[list(p) for p in schedule.pairs_swept]}")
    print(f"  pairs pruned {[list(p) for p in schedule.pairs_skipped]}")
    for note in schedule.notes:
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
