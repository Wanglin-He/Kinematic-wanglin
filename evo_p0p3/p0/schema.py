"""The frozen prompt contract.

P0 is authored before the asset exists and is never edited afterwards. Everything here
is immutable on purpose: a contract that can be revised after seeing a candidate asset
is not a contract, it is a rationalisation.

Two design rules run through the whole schema and explain most of its shape.

**Every name must resolve to geometry.** The previous attempt at these metrics matched
prose part names against code identifiers and drowned: of 70 flagged failures 63 were
false alarms, 42 of those from "part not found" -- but 4 of the 7 true positives came
from that same flag. Signal and noise were the same event, and a paired bootstrap showed
no better matcher could separate them. So here, every identifier used anywhere must be
declared in ``required_parts`` or ``joints`` (rule A1), and instances are enumerated
rather than expanded from a count. A "part not found" then has exactly one meaning.

**Prefer relations over coordinates.** A URDF has no semantic orientation -- only numbers.
Two identical cabinets authored ninety degrees apart are the same object with different
files. Requiring generators to author in a canonical frame would turn a convention
violation into a kinematics failure, which across several generators measures who read
the frame documentation rather than whose joints are right. So an axis may be stated
semantically (resolved against gravity, which is free), relationally (resolved from the
geometry of two named parts, which follows the asset wherever it is placed), or numerically
(only when ``canonical_frame.front`` is anchored, and reported N/A otherwise).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum


class Role(StrEnum):
    """What a part is expected to do, which fixes what the Gate demands of it."""

    FIXED = "fixed"
    """No joint of its own; part of the static base."""

    MOVABLE = "movable"
    """Has its own joint. Gate G2 requires exactly one resolvable motion chain."""

    ATTACHED = "attached"
    """Moves, but rigidly with another part and with no joint of its own.

    This role exists because the previous schema wrote ``role: follows_drawer``, which
    both broke the declared two-value vocabulary and could not say *which* handle followed
    *which* drawer. Worse, calling a handle ``movable`` would make Gate G2 demand a joint
    the correct handle must not have.
    """


class JointType(StrEnum):
    """MuJoCo's joint vocabulary, which is what the model is ultimately checked against.

    URDF's names map on load: ``prismatic`` becomes slide, and both ``revolute`` and
    ``continuous`` become hinge, separated only by ``jnt_limited``.
    """

    SLIDE = "slide"
    HINGE = "hinge"
    BALL = "ball"
    FREE = "free"

    @property
    def has_anchor(self) -> bool:
        """Whether a pivot point affects this joint's kinematics at all.

        Measured rather than assumed, by displacing ``pos`` by 0.25 m and reading how far
        the child's geometry moved:

        ===========  ==================  ==================================================
        joint        child displacement  meaning
        ===========  ==================  ==================================================
        ``slide``    0.000000            a translation has no centre; ``pos`` is inert
        ``hinge``    0.171449            rotates about the line through the anchor
        ``ball``     0.191345            rotates about the anchor *point*
        ``free``     0.000000            MuJoCo ignores ``pos`` on a free joint outright
        ===========  ==================  ==================================================

        So an anchor claim is checkable for hinge and ball, and uncheckable for slide and
        free. An earlier version excluded ball on the assumption that only a hinge has a
        line in space to be wrong about; that rejected a claim a model can genuinely fail.
        A ball joint's anchor is the centre of rotation, and putting it in the wrong place
        swings the child through a different arc.
        """
        return self in (JointType.HINGE, JointType.BALL)

    @property
    def unit(self) -> str:
        return "m" if self is JointType.SLIDE else "rad"


class AxisSemantic(StrEnum):
    """Directions that gravity alone pins down, and which therefore cost nothing.

    MuJoCo's default gravity is (0, 0, -9.81) and +z up is near universal in MJCF;
    URDF/ROS REP-103 agrees. So "vertical" and "horizontal" are free, need no canonical
    frame, and survive any rotation of the asset about the vertical.
    """

    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


class RelationalKind(StrEnum):
    """How to recover a reference direction from the asset's own geometry."""

    INTERFACE_NORMAL = "interface_normal"
    """Normal of the mating surface between two parts. Drawer pull-out, piston travel."""

    INTERFACE_LINE = "interface_line"
    """Longest principal direction of the contact region between two parts. Hinge lines."""

    TOWARD = "toward"
    """Centroid of one part toward the centroid of another. Operating direction."""

    SYMMETRY_AXIS_OF = "symmetry_axis_of"
    """A part's own axis of rotational symmetry. Wheels, gears, knobs, turntables."""


class ContactRelation(StrEnum):
    """What a declared contact asserts. The vocabulary is closed, and frozen here.

    The previous schema had a single ``allowed`` bucket carrying an undefined token
    ``stop_contact``, never saying whether it meant "contact here is not a violation"
    (in which case it can never fail and contributes nothing) or "these parts must touch"
    (in which case it needs a tolerance and a state predicate, neither of which existed).
    Splitting the buckets forces that decision at authoring time.
    """

    STOP_CONTACT = "stop_contact"
    """Obligation: the pair must touch in the named state. A travel limit with no
    geometric stop is the difference between a mechanism and a declaration."""

    MESH_CONTACT = "mesh_contact"
    """Obligation: the pair must touch throughout. Gear teeth, rack and pinion."""

    REST_CONTACT = "rest_contact"
    """Permission: contact here is not a violation, and its absence is not a failure."""

    @property
    def is_obligation(self) -> bool:
        return self is not ContactRelation.REST_CONTACT


class StateScope(StrEnum):
    """Which joints a named whole-asset state constrains.

    ``joints[].states`` labels are per joint, but ``contact_policy`` and
    ``dynamics_claims.initial_state`` consume them as properties of the whole asset. With
    three independent drawers, a sweep sample at (0.0, 0.40, 0.35) has to be classifiable
    or the state-conditioned claims are never evaluated -- and never evaluated reads as
    passing.
    """

    NAMED_PARTS = "named_parts"
    """Only the joints of the parts named by the consuming claim. The default: "drawer_1
    is shut" should not require the other two drawers to be shut as well."""

    ALL_DECLARING = "all_declaring"
    """Every joint that declares this label. For whole-asset initial poses."""


class ResidualNorm(StrEnum):
    """What to divide a coupling residual by before comparing it against epsilon.

    This must be chosen per coupling, not defaulted. A gear whose range is plus or minus
    three turns spans 37.7 rad, so a relative epsilon of 0.02 would tolerate 0.75 rad --
    about 43 degrees of phase error, which is meaningless for a gear.
    """

    DEPENDENT_RANGE_SPAN = "dependent_range_span"
    INDEPENDENT_RANGE_SPAN = "independent_range_span"
    ABSOLUTE = "absolute"


# --------------------------------------------------------------------------------------
# Parts
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Part:
    """One required part instance.

    Instances are enumerated rather than declared as ``{id: drawer, count: 3}``, because
    nothing ever specified how a count expanded into ``drawer_1..3`` nor what order the
    Gate would assign indices in. ``group`` restores the convenience of addressing them
    collectively without reintroducing that ambiguity: membership is explicit, so the
    expansion is checkable.
    """

    id: str
    role: Role
    group: str | None = None


# --------------------------------------------------------------------------------------
# Kinematic claims
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CanonicalFrame:
    """How to turn direction words into vectors.

    ``up`` is always gravity and needs no check. ``front`` is the genuinely undetermined
    one -- no fact in the file or in physics says which way a cabinet faces -- so it is
    None unless the task anchors it to a declared part's outward normal.
    """

    up: str = "gravity"
    front: str | None = None
    """When set, the id of the part whose outward normal defines forward."""

    @property
    def front_anchored(self) -> bool:
        return self.front is not None


@dataclass(frozen=True, slots=True)
class RelationalAxis:
    """A direction recovered from the asset's own geometry, so it needs no frame."""

    kind: RelationalKind
    from_part: str | None = None
    to_part: str | None = None
    between: tuple[str, str] | None = None
    part: str | None = None

    def referenced_parts(self) -> tuple[str, ...]:
        names = [self.from_part, self.to_part, self.part]
        if self.between:
            names.extend(self.between)
        return tuple(n for n in names if n)


@dataclass(frozen=True, slots=True)
class Axis:
    """The joint's direction, in up to three independently scored forms.

    All three may be given. They are scored separately and reported separately, because
    they do not survive the same things: rotate a correct asset ninety degrees about the
    vertical and ``numeric`` fails while ``semantic`` and ``relational`` do not. That is
    a controlled experiment, and the point of keeping all three for now.
    """

    semantic: AxisSemantic | None = None
    relational: RelationalAxis | None = None
    numeric: tuple[float, float, float] | None = None

    @property
    def forms(self) -> tuple[str, ...]:
        present = []
        if self.semantic is not None:
            present.append("semantic")
        if self.relational is not None:
            present.append("relational")
        if self.numeric is not None:
            present.append("numeric")
        return tuple(present)


@dataclass(frozen=True, slots=True)
class Anchor:
    """Where a hinge's line of rotation must sit, stated as a geometric relation.

    Not as coordinates: P0 is written before the asset exists, so it cannot know how big
    the door will be or where it will sit. ``envelope_m`` gives a size range, not a
    geometry, and a hard-coded ``[0.4, 0, 0]`` is only checkable if the door happens to
    land where the author guessed.

    The two forms are opposites and both are needed. A door's hinge demands the geometry
    lie entirely to one side of the axis plane; a gear's shaft demands the geometry be
    symmetric about it. Scoring a gear with ``on_edge_of`` would fail it by construction,
    since every plane through a disc's symmetry axis cuts it in half.

    Applies to hinge and ball joints. See :attr:`JointType.has_anchor` for why those two
    and not the others.
    """

    on_edge_of: str | None = None
    through_center_of: str | None = None

    def referenced_parts(self) -> tuple[str, ...]:
        return tuple(n for n in (self.on_edge_of, self.through_center_of) if n)


@dataclass(frozen=True, slots=True)
class Range:
    """Declared travel. ``unit`` is explicit because slide and hinge do not share one."""

    unit: str
    min: float
    max: float

    @property
    def span(self) -> float:
        return self.max - self.min


@dataclass(frozen=True, slots=True)
class Joint:
    """One articulation claim.

    ``states`` values live in ``range.unit``. Note that ``states['closed']``,
    the reference configuration and ``range.min`` are typically the same number; KF1
    scores them as a single combined claim rather than three, since counting one declared
    number three times is exactly the double counting P0 exists to prevent.
    """

    id: str
    part: str
    parent: str
    type: JointType
    axis: Axis
    range: Range
    states: Mapping[str, float] = field(default_factory=dict)
    anchor: Anchor | None = None


@dataclass(frozen=True, slots=True)
class RigidAttachment:
    """A part that moves rigidly with another and has no joint of its own.

    Declared as an explicit instance pair. Without it, ``handle_3`` riding ``drawer_2``
    passes every check, because nothing ever said which one it was supposed to follow.
    """

    follower: str
    leader: str


@dataclass(frozen=True, slots=True)
class CouplingRelation:
    """``q_dependent = coefficient * q_independent + offset``.

    Deliberately isomorphic to MuJoCo's ``equality type="joint"`` polycoef semantics, so
    the comparison is field against field with nothing to interpret. The mechanism is not
    specified: a fixed tendon realising the same relation is equally correct.
    """

    dependent: str
    independent: str
    coefficient: float
    offset: float = 0.0


@dataclass(frozen=True, slots=True)
class Coupling:
    """A declared mechanical linkage.

    ``expected_dof`` is what makes "two joints that merely happen to move together"
    distinguishable from "two joints actually constrained to each other": it is checked
    as a rank, not as a declaration.
    """

    id: str
    relation: CouplingRelation
    expected_dof: int
    residual_norm: ResidualNorm = ResidualNorm.ABSOLUTE
    mechanism: str = "any"
    epsilon: float | None = None
    """Overrides ``tolerances.coupling_residual`` when set."""


@dataclass(frozen=True, slots=True)
class AssetState:
    """A named whole-asset configuration, with the tolerance that makes it non-empty.

    Without a tolerance, "closed" would mean exact equality, and a continuous sweep can
    contain zero samples satisfying it -- so the claims conditioned on that state would
    never be checked, and never checked reads as passing.
    """

    scope: StateScope = StateScope.NAMED_PARTS
    tolerance_relative: float = 0.02


@dataclass(frozen=True, slots=True)
class ContactClaim:
    """One contact pair, in one of the three buckets.

    ``parts`` always holds two concrete part ids. A claim written against a group is
    expanded to the cross product at load time, with ``expanded_from`` recording the
    original text, so the pair count is never in doubt and the expansion is auditable.
    """

    parts: tuple[str, str]
    state: str = "all"
    relation: ContactRelation | None = None
    tolerance_override: float | None = None
    expanded_from: str | None = None


@dataclass(frozen=True, slots=True)
class ContactPolicy:
    """Contact expectations over the swept configuration space.

    ``required`` and ``permitted`` win over ``forbidden`` in the states where they apply.
    Without that precedence the canonical example is undefined at exactly the
    configurations the sweep begins and ends in: a drawer resting against its stop is
    simultaneously required at the closed state and forbidden at all states.
    """

    required: tuple[ContactClaim, ...] = ()
    permitted: tuple[ContactClaim, ...] = ()
    forbidden: tuple[ContactClaim, ...] = ()
    precedence: str = "required_and_permitted_over_forbidden"

    def all_claims(self):
        yield from self.required
        yield from self.permitted
        yield from self.forbidden


@dataclass(frozen=True, slots=True)
class Tolerances:
    """Every threshold the evaluation uses, frozen with the contract.

    Values here are protocol starting points and must be calibrated before freezing. They
    will not be calibrated on real failures: the annotated corpus has 47 failing assets in
    607, 6-16 positives per item, and a sweep of the decision threshold over [0, 1]
    produced a best kappa of 0.022-0.326. The scores never separated the classes, so the
    threshold was never the problem. Calibration goes through deterministic fault
    injection instead, taking the knee of a detection-rate against severity curve.
    """

    axis_angle_deg: float = 15.0
    anchor_side_fraction_min: float = 0.95
    anchor_center_offset_max: float = 0.05
    travel_scale_min: float = 0.50
    follower_drift_m: float = 0.0005
    state_match_relative: float = 0.02
    required_contact_m: float = 0.002
    forbidden_penetration_m: float = 0.001
    coupling_residual: float = 0.02

    def digest(self) -> str:
        """Stable hash of the effective tolerance set, emitted with every result.

        Frozen thresholds that nobody can point at are not frozen.
        """
        import hashlib

        parts = [f"{k}={getattr(self, k)!r}" for k in sorted(self.__dataclass_fields__)]
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class KinematicClaims:
    """Everything P3 consumes."""

    joints: tuple[Joint, ...] = ()
    rigid_attachments: tuple[RigidAttachment, ...] = ()
    couplings: tuple[Coupling, ...] = ()
    independent_dofs: tuple[str, ...] = ()
    asset_states: Mapping[str, AssetState] = field(default_factory=dict)
    contact_policy: ContactPolicy = field(default_factory=ContactPolicy)
    tolerances: Tolerances = field(default_factory=Tolerances)
    canonical_frame: CanonicalFrame = field(default_factory=CanonicalFrame)

    def joint(self, joint_id: str) -> Joint | None:
        for j in self.joints:
            if j.id == joint_id:
                return j
        return None


# --------------------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Contract:
    """A complete frozen prompt contract.

    Geometry claims are carried but not modelled in detail: they belong to P2, and P3
    must not rescore them. They are kept because rule A1 checks referential integrity
    across every field, including the ones this module does not otherwise interpret.
    """

    record_id: str
    overall_description: str
    required_parts: tuple[Part, ...]
    kinematic_claims: KinematicClaims
    part_geometry: Mapping[str, str] = field(default_factory=dict)
    part_relations: tuple[Mapping[str, str], ...] = ()
    proportion_claims: tuple[Mapping[str, object], ...] = ()
    global_form: Mapping[str, str] = field(default_factory=dict)
    envelope_m: Mapping[str, Sequence[float]] = field(default_factory=dict)
    source_path: str | None = None

    def part(self, part_id: str) -> Part | None:
        for p in self.required_parts:
            if p.id == part_id:
                return p
        return None

    @property
    def part_ids(self) -> frozenset[str]:
        return frozenset(p.id for p in self.required_parts)

    @property
    def joint_ids(self) -> frozenset[str]:
        return frozenset(j.id for j in self.kinematic_claims.joints)

    def group_members(self, group: str) -> tuple[str, ...]:
        """Members of a group, in declaration order.

        Order is declaration order rather than sorted order so that expansion is stable
        and readable, and so two contracts differing only in declaration order are
        visibly different rather than silently identical.
        """
        return tuple(p.id for p in self.required_parts if p.group == group)

    @property
    def groups(self) -> frozenset[str]:
        return frozenset(p.group for p in self.required_parts if p.group)
