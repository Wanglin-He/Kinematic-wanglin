"""``p0 validate`` -- run the admission checks over one or more contracts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evo_p0p3.p0 import admission
from evo_p0p3.p0.loader import ContractSyntaxError, load_contract


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="p0", description="Evo-Articraft prompt contract tools")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="run admission checks (A1..A16)")
    validate.add_argument("paths", nargs="+", type=Path, help="contract .yaml files or directories")
    validate.add_argument("-q", "--quiet", action="store_true", help="only print failures")

    args = parser.parse_args(argv)
    if args.command == "validate":
        return _validate(args.paths, quiet=args.quiet)
    return 2  # pragma: no cover


def _expand(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        out.extend(sorted(p.glob("*.yaml")) if p.is_dir() else [p])
    return out


def _validate(paths: list[Path], *, quiet: bool) -> int:
    files = _expand(paths)
    if not files:
        print("no contract files found", file=sys.stderr)
        return 2

    rejected = 0
    for path in files:
        try:
            contract = load_contract(path)
        except ContractSyntaxError as exc:
            rejected += 1
            print(f"{path}: SYNTAX ERROR")
            print(f"  {exc}")
            continue

        report = admission.check(contract)
        if not report.admitted:
            rejected += 1
        if quiet and report.admitted and not report.warnings:
            continue

        print(f"{path}: {report.summary()}")
        for finding in report.findings:
            marker = "✗" if finding.severity is admission.Severity.ERROR else "!"
            print(f"  {marker} {finding}")

    print(f"\n{len(files) - rejected}/{len(files)} admitted")
    return 1 if rejected else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
