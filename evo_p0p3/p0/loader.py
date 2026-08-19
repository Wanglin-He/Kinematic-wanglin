"""YAML on disk to a typed :class:`~evo_p0p3.p0.schema.Contract`.

This layer answers one question only: is the file shaped like a contract? A key of the
wrong type, an unknown enum value, a joint with no range -- those are syntax, and they
raise here with the exact path that is wrong.

Whether a *well-formed* contract is *admissible* is a different question with a different
answer shape. "Every id you reference must be declared" is not a parse error; it is a
finding a human author has to act on, and several can be true at once. That lives in
:mod:`evo_p0p3.p0.admission` and returns a report instead of raising.

Keeping the two apart matters because they fail at different times: a syntax error means
the contract cannot be read at all, while an admission failure means the contract reads
fine and says something that cannot be checked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from evo_p0p3.p0.schema import (
    Anchor,
    AssetState,
    Axis,
    AxisSemantic,
    CanonicalFrame,
    ContactClaim,
    ContactPolicy,
    ContactRelation,
    Contract,
    Coupling,
    CouplingRelation,
    Joint,
    JointType,
    KinematicClaims,
    Part,
    Range,
    RelationalAxis,
    RelationalKind,
    ResidualNorm,
    RigidAttachment,
    Role,
    StateScope,
    Tolerances,
)


class ContractSyntaxError(ValueError):
    """The file is not shaped like a contract. Always carries the offending path."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message


def load_contract(path: str | Path) -> Contract:
    """Read and type a contract file."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - message passthrough
        raise ContractSyntaxError(str(path), f"not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ContractSyntaxError(str(path), "top level must be a mapping")
    return parse_contract(raw, source_path=str(path), record_id=path.stem)


def parse_contract(
    raw: dict[str, Any], *, source_path: str | None = None, record_id: str = ""
) -> Contract:
    parts = _parts(raw.get("required_parts"), "required_parts")
    groups: dict[str, list[str]] = {}
    for p in parts:
        if p.group:
            groups.setdefault(p.group, []).append(p.id)

    return Contract(
        record_id=str(raw.get("record_id") or record_id),
        overall_description=str(raw.get("overall_description") or ""),
        required_parts=parts,
        kinematic_claims=_kinematics(raw.get("kinematic_claims") or {}, "kinematic_claims", groups),
        part_geometry=_part_geometry(raw.get("part_geometry")),
        part_relations=tuple(_mappings(raw.get("part_relations"), "part_relations")),
        proportion_claims=tuple(_mappings(raw.get("proportion_claims"), "proportion_claims")),
        global_form=dict(raw.get("global_form") or {}),
        envelope_m=dict(raw.get("envelope_m") or {}),
        source_path=source_path,
    )


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _require(value: Any, path: str, what: str = "is required") -> Any:
    if value is None:
        raise ContractSyntaxError(path, what)
    return value


def _enum(cls, value: Any, path: str):
    if value is None:
        return None
    try:
        return cls(str(value))
    except ValueError:
        allowed = ", ".join(m.value for m in cls)
        raise ContractSyntaxError(path, f"{value!r} is not one of: {allowed}") from None


def _float(value: Any, path: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ContractSyntaxError(path, f"expected a number, got {value!r}") from None


def _seq(value: Any, path: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ContractSyntaxError(path, f"expected a list, got {type(value).__name__}")
    return value


def _mappings(value: Any, path: str) -> list[dict[str, Any]]:
    out = []
    for i, item in enumerate(_seq(value, path)):
        if not isinstance(item, dict):
            raise ContractSyntaxError(f"{path}[{i}]", "expected a mapping")
        out.append(item)
    return out


def _part_geometry(value: Any) -> dict[str, str]:
    """Accepts the list-of-records form the schema example uses."""
    out: dict[str, str] = {}
    for i, item in enumerate(_mappings(value, "part_geometry")):
        pid = item.get("id")
        if not pid:
            raise ContractSyntaxError(f"part_geometry[{i}]", "missing 'id'")
        out[str(pid)] = str(item.get("geometry") or "")
    return out


def _parts(value: Any, path: str) -> tuple[Part, ...]:
    out = []
    for i, item in enumerate(_mappings(value, path)):
        p = f"{path}[{i}]"
        if "count" in item:
            raise ContractSyntaxError(
                p,
                "'count' is not supported; enumerate instances as separate entries and "
                "use 'group' to address them collectively",
            )
        out.append(
            Part(
                id=str(_require(item.get("id"), f"{p}.id")),
                role=_enum(Role, _require(item.get("role"), f"{p}.role"), f"{p}.role"),
                group=str(item["group"]) if item.get("group") else None,
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------------------
# kinematic claims
# --------------------------------------------------------------------------------------


def _kinematics(raw: Any, path: str, groups: dict[str, list[str]]) -> KinematicClaims:
    if not isinstance(raw, dict):
        raise ContractSyntaxError(path, "expected a mapping")
    return KinematicClaims(
        canonical_frame=_frame(raw.get("canonical_frame") or {}, f"{path}.canonical_frame"),
        joints=_joints(raw.get("joints"), f"{path}.joints"),
        rigid_attachments=_attachments(
            raw.get("rigid_attachments"), f"{path}.rigid_attachments"
        ),
        couplings=_couplings(raw.get("couplings"), f"{path}.couplings"),
        independent_dofs=tuple(
            str(x) for x in _seq(raw.get("independent_dofs"), f"{path}.independent_dofs")
        ),
        asset_states=_asset_states(raw.get("asset_states"), f"{path}.asset_states"),
        contact_policy=_contact_policy(
            raw.get("contact_policy") or {}, f"{path}.contact_policy", groups
        ),
        tolerances=_tolerances(raw.get("tolerances") or {}, f"{path}.tolerances"),
    )


def _frame(raw: Any, path: str) -> CanonicalFrame:
    if not isinstance(raw, dict):
        raise ContractSyntaxError(path, "expected a mapping")
    up = str(raw.get("up") or "gravity")
    if up != "gravity":
        raise ContractSyntaxError(
            f"{path}.up",
            f"{up!r} is not supported; 'up' is always 'gravity' (MuJoCo's default is "
            "(0, 0, -9.81), which is the only direction the file actually determines)",
        )
    front = raw.get("front")
    if front is None:
        return CanonicalFrame(up=up, front=None)
    if isinstance(front, dict):
        anchor = front.get("outward_normal_of")
        if not anchor:
            raise ContractSyntaxError(
                f"{path}.front", "expected {outward_normal_of: <part id>} or null"
            )
        return CanonicalFrame(up=up, front=str(anchor))
    raise ContractSyntaxError(f"{path}.front", "expected a mapping or null")


def _axis(raw: Any, path: str) -> Axis:
    if raw is None:
        raise ContractSyntaxError(path, "is required")
    if not isinstance(raw, dict):
        raise ContractSyntaxError(path, "expected a mapping")

    numeric = None
    if raw.get("numeric") is not None:
        vec = _seq(raw["numeric"], f"{path}.numeric")
        if len(vec) != 3:
            raise ContractSyntaxError(f"{path}.numeric", "expected three components")
        numeric = tuple(_float(v, f"{path}.numeric[{i}]") for i, v in enumerate(vec))

    relational = None
    if raw.get("relational") is not None:
        r = raw["relational"]
        if not isinstance(r, dict):
            raise ContractSyntaxError(f"{path}.relational", "expected a mapping")
        kind = _enum(
            RelationalKind,
            _require(r.get("type"), f"{path}.relational.type"),
            f"{path}.relational.type",
        )
        between = r.get("between")
        if between is not None:
            pair = _seq(between, f"{path}.relational.between")
            if len(pair) != 2:
                raise ContractSyntaxError(
                    f"{path}.relational.between", "expected exactly two part ids"
                )
            between = (str(pair[0]), str(pair[1]))
        relational = RelationalAxis(
            kind=kind,
            from_part=str(r["from"]) if r.get("from") else None,
            to_part=str(r["to"]) if r.get("to") else None,
            between=between,
            part=str(r["part"]) if r.get("part") else None,
        )
        _check_relational_shape(relational, f"{path}.relational")

    return Axis(
        semantic=_enum(AxisSemantic, raw.get("semantic"), f"{path}.semantic"),
        relational=relational,
        numeric=numeric,
    )


def _check_relational_shape(rel: RelationalAxis, path: str) -> None:
    """Each relational kind needs its own operands; a missing one is unresolvable."""
    needs = {
        RelationalKind.INTERFACE_NORMAL: ("from", "to"),
        RelationalKind.INTERFACE_LINE: ("between",),
        RelationalKind.TOWARD: ("from", "to"),
        RelationalKind.SYMMETRY_AXIS_OF: ("part",),
    }[rel.kind]
    have = {
        "from": rel.from_part,
        "to": rel.to_part,
        "between": rel.between,
        "part": rel.part,
    }
    missing = [k for k in needs if not have[k]]
    if missing:
        raise ContractSyntaxError(
            path, f"type {rel.kind.value!r} requires {', '.join(missing)}"
        )


def _anchor(raw: Any, path: str) -> Anchor | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ContractSyntaxError(path, "expected a mapping or null")
    on_edge = raw.get("on_edge_of")
    through = raw.get("through_center_of")
    if bool(on_edge) == bool(through):
        raise ContractSyntaxError(
            path, "give exactly one of 'on_edge_of' or 'through_center_of'"
        )
    return Anchor(
        on_edge_of=str(on_edge) if on_edge else None,
        through_center_of=str(through) if through else None,
    )


def _range(raw: Any, path: str) -> Range:
    if not isinstance(raw, dict):
        raise ContractSyntaxError(path, "expected a mapping")
    unit = str(_require(raw.get("unit"), f"{path}.unit"))
    if unit not in ("m", "rad"):
        raise ContractSyntaxError(f"{path}.unit", f"expected 'm' or 'rad', got {unit!r}")
    lo = _float(_require(raw.get("min"), f"{path}.min"), f"{path}.min")
    hi = _float(_require(raw.get("max"), f"{path}.max"), f"{path}.max")
    if not hi > lo:
        raise ContractSyntaxError(path, f"max ({hi}) must exceed min ({lo})")
    return Range(unit=unit, min=lo, max=hi)


def _joints(value: Any, path: str) -> tuple[Joint, ...]:
    out = []
    for i, item in enumerate(_mappings(value, path)):
        p = f"{path}[{i}]"
        jtype = _enum(JointType, _require(item.get("type"), f"{p}.type"), f"{p}.type")
        states = item.get("states") or {}
        if not isinstance(states, dict):
            raise ContractSyntaxError(f"{p}.states", "expected a mapping")
        out.append(
            Joint(
                id=str(_require(item.get("id"), f"{p}.id")),
                part=str(_require(item.get("part"), f"{p}.part")),
                parent=str(_require(item.get("parent"), f"{p}.parent")),
                type=jtype,
                axis=_axis(item.get("axis"), f"{p}.axis"),
                anchor=_anchor(item.get("anchor"), f"{p}.anchor"),
                range=_range(_require(item.get("range"), f"{p}.range"), f"{p}.range"),
                states={
                    str(k): _float(v, f"{p}.states.{k}") for k, v in states.items()
                },
            )
        )
    return tuple(out)


def _attachments(value: Any, path: str) -> tuple[RigidAttachment, ...]:
    out = []
    for i, item in enumerate(_mappings(value, path)):
        p = f"{path}[{i}]"
        out.append(
            RigidAttachment(
                follower=str(_require(item.get("follower"), f"{p}.follower")),
                leader=str(_require(item.get("leader"), f"{p}.leader")),
            )
        )
    return tuple(out)


def _couplings(value: Any, path: str) -> tuple[Coupling, ...]:
    out = []
    for i, item in enumerate(_mappings(value, path)):
        p = f"{path}[{i}]"
        rel = _require(item.get("relation"), f"{p}.relation")
        if not isinstance(rel, dict):
            raise ContractSyntaxError(f"{p}.relation", "expected a mapping")
        norm = item.get("residual_norm") or {}
        if isinstance(norm, dict):
            norm_by = norm.get("by", "absolute")
        else:
            norm_by = norm
        out.append(
            Coupling(
                id=str(_require(item.get("id"), f"{p}.id")),
                relation=CouplingRelation(
                    dependent=str(_require(rel.get("dependent"), f"{p}.relation.dependent")),
                    independent=str(
                        _require(rel.get("independent"), f"{p}.relation.independent")
                    ),
                    coefficient=_float(
                        _require(rel.get("coefficient"), f"{p}.relation.coefficient"),
                        f"{p}.relation.coefficient",
                    ),
                    offset=_float(rel.get("offset", 0.0), f"{p}.relation.offset"),
                ),
                expected_dof=int(_require(item.get("expected_dof"), f"{p}.expected_dof")),
                residual_norm=_enum(ResidualNorm, norm_by, f"{p}.residual_norm.by"),
                mechanism=str(item.get("mechanism") or "any"),
                epsilon=(
                    _float(item["epsilon"], f"{p}.epsilon")
                    if item.get("epsilon") is not None
                    else None
                ),
            )
        )
    return tuple(out)


def _asset_states(value: Any, path: str) -> dict[str, AssetState]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ContractSyntaxError(path, "expected a mapping of state name to definition")
    out = {}
    for name, spec in value.items():
        p = f"{path}.{name}"
        if not isinstance(spec, dict):
            raise ContractSyntaxError(p, "expected a mapping")
        tol = spec.get("tolerance") or {}
        out[str(name)] = AssetState(
            scope=_enum(StateScope, spec.get("scope", "named_parts"), f"{p}.scope"),
            tolerance_relative=_float(tol.get("relative", 0.02), f"{p}.tolerance.relative"),
        )
    return out


def _contact_parts(
    value: Any, path: str, groups: dict[str, list[str]]
) -> list[tuple[tuple[str, str], str | None]]:
    """Expand one ``parts`` entry into concrete pairs.

    A group on either side becomes the cross product. Self-pairs are dropped during
    expansion -- ``[{group: drawer}, {group: drawer}]`` means the drawers against each
    other, not each drawer against itself -- but a self-pair written out literally is an
    error rather than a silent no-op, because dropping it would leave the author with a
    claim they can see in the file and the evaluator never scores.
    """
    items = _seq(value, path)
    if len(items) != 2:
        raise ContractSyntaxError(path, "a contact pair needs exactly two entries")

    sides: list[list[str]] = []
    labels: list[str] = []
    for i, item in enumerate(items):
        if isinstance(item, dict):
            g = item.get("group")
            if not g:
                raise ContractSyntaxError(f"{path}[{i}]", "expected {group: <name>} or a part id")
            if str(g) not in groups:
                raise ContractSyntaxError(f"{path}[{i}]", f"unknown group {g!r}")
            sides.append(list(groups[str(g)]))
            labels.append(f"group:{g}")
        else:
            sides.append([str(item)])
            labels.append(str(item))

    from_group = any(lbl.startswith("group:") for lbl in labels)
    if not from_group and sides[0][0] == sides[1][0]:
        raise ContractSyntaxError(
            path, f"a contact pair needs two distinct parts, both are {sides[0][0]!r}"
        )

    expanded_from = f"[{labels[0]}, {labels[1]}]" if from_group else None
    pairs = []
    seen = set()
    for a in sides[0]:
        for b in sides[1]:
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append(((a, b), expanded_from))
    return pairs


def _contact_bucket(
    value: Any, path: str, groups: dict[str, list[str]], *, default_relation: ContactRelation | None
) -> tuple[ContactClaim, ...]:
    out = []
    for i, item in enumerate(_mappings(value, path)):
        p = f"{path}[{i}]"
        relation = _enum(ContactRelation, item.get("relation"), f"{p}.relation")
        if relation is None:
            relation = default_relation
        tol = item.get("tolerance")
        for pair, expanded in _contact_parts(item.get("parts"), f"{p}.parts", groups):
            out.append(
                ContactClaim(
                    parts=pair,
                    state=str(item.get("state") or "all"),
                    relation=relation,
                    tolerance_override=_float(tol, f"{p}.tolerance") if tol is not None else None,
                    expanded_from=expanded,
                )
            )
    return tuple(out)


def _contact_policy(raw: Any, path: str, groups: dict[str, list[str]]) -> ContactPolicy:
    if not isinstance(raw, dict):
        raise ContractSyntaxError(path, "expected a mapping")
    if "allowed" in raw:
        raise ContractSyntaxError(
            f"{path}.allowed",
            "'allowed' is ambiguous between permission and obligation; use 'required' "
            "(the contact must occur) or 'permitted' (the contact is not a violation)",
        )
    return ContactPolicy(
        required=_contact_bucket(
            raw.get("required"), f"{path}.required", groups,
            default_relation=ContactRelation.STOP_CONTACT,
        ),
        permitted=_contact_bucket(
            raw.get("permitted"), f"{path}.permitted", groups,
            default_relation=ContactRelation.REST_CONTACT,
        ),
        forbidden=_contact_bucket(
            raw.get("forbidden"), f"{path}.forbidden", groups, default_relation=None
        ),
        precedence=str(raw.get("precedence") or "required_and_permitted_over_forbidden"),
    )


def _tolerances(raw: Any, path: str) -> Tolerances:
    if not isinstance(raw, dict):
        raise ContractSyntaxError(path, "expected a mapping")
    known = set(Tolerances.__dataclass_fields__)
    unknown = sorted(set(map(str, raw)) - known)
    if unknown:
        raise ContractSyntaxError(
            path,
            f"unknown tolerance key(s): {', '.join(unknown)}. Known keys: "
            f"{', '.join(sorted(known))}",
        )
    return Tolerances(**{k: _float(v, f"{path}.{k}") for k, v in raw.items()})
