"""Segment-boundary definitions for the 15-segment bi-atrial model, taken
from the "Definition of segmental boundaries" subsections of Althoff et al.,
Europace 2025;27:euaf134.

Three kinds of boundary are distinguished:

- Interior geodesics (`LA_EDGES`): a plain shortest-path line between two
  landmarks across the open atrial surface.
- Boundary arcs (`LA_BOUNDARY_ARC_EDGES`, `ArcEdge`): the two landmarks both
  sit on the same open mesh boundary loop (e.g. the mitral annulus rim), and
  the segment border is simply that rim between them - not a geodesic, since
  the rim itself already is the anatomical border. Normally the shorter of
  the two possible arcs; `ArcEdge(whole_loop=True)` instead tags the entire
  loop (both arcs combined) for a two-point hole shared by two segments
  (see the IVC entry in RA_BOUNDARY_ARC_EDGES) - neither arc alone can be
  reliably attributed to one segment or the other (tried picking by length
  and by nearest landmark; both turned out not to be anatomically
  meaningful and flipped between sessions), but that's fine: each segment
  already has other unique tags that identify it, so "touches this hole at
  all" is all the IVC tag needs to contribute.
- Loop composite paths (`LA_LOOP_PATHS`): a border that runs from a point,
  to the nearest point on a closed appendage-neck loop, along the loop, to
  the nearest-to-the-other-point loop vertex, then on to the other point.

RA's tables are derived from connectivity supplied directly by the user
(who cross-checked it against the reference implementation's boundary
files) rather than from the paper text - see the RA_EDGES/RA_SEGMENT_BORDERS
comments below for the exact per-region edge lists.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArcEdge:
    """A segment boundary that walks a mesh open-boundary rim between two
    landmarks on the same loop, rather than a Dijkstra geodesic.

    `whole_loop=True` tags every vertex of the entire loop (both possible
    arcs between a and b) instead of picking just one - for a two-point
    hole (e.g. the IVC) shared by two segments, where there's no reliable
    way to attribute one specific arc to one specific segment. Each segment
    still gets identified correctly via its other, unique tags; the shared
    tag just confirms "touches this hole."

    `name` overrides the default `f"{a}_{b}"` curve name.
    """

    a: str
    b: str
    whole_loop: bool = False
    name: str | None = None

    @property
    def curve_name(self) -> str:
        return self.name or edge_name(self.a, self.b)


LA_EDGES: list[tuple[str, str]] = [
    ("A", "B"),  # segment 1 proximal (dome-side) border / segment 3 left edge
    ("C", "D"),  # segment 2 proximal (dome-side) border / segment 3 right edge
    ("A", "C"),  # segment 3 superior border ("roof line") / segment 4 cranial border
    ("B", "D"),  # segment 3 inferior border / segment 7 cranial border
    ("E", "C"),  # segment 4 septal border / segment 8 antero-superior border
    ("D", "I"),  # segment 7 septal border / segment 8 infero-posterior border
    ("H", "B"),  # segment 6 infero-posterior border / segment 7 lateral border
]

# Both landmarks sit on the same detected mesh boundary loop (the mitral
# annulus rim); the border is that rim arc, not a Dijkstra shortest path.
LA_BOUNDARY_ARC_EDGES: list[ArcEdge] = [
    ArcEdge("E", "F"),  # segment 4 apical border (mitral annulus 9-1 o'clock)
    ArcEdge("F", "H"),  # segment 6 apical border (mitral annulus 1-4 o'clock)
    ArcEdge("H", "I"),  # segment 7 apical border (mitral annulus 4-7 o'clock)
    ArcEdge("I", "E"),  # segment 8 apical border (mitral annulus 7-9 o'clock)
]

# Composite paths routed via a loop landmark: (endpoint_a, loop_name, endpoint_b)
# means geodesic(a, nearest loop point to a) + loop arc + geodesic(nearest
# loop point to b, b).
LA_LOOP_PATHS: list[tuple[str, str, str]] = [
    ("A", "LAA_neck", "F"),  # segment 4 lateral border / segment 6 antero-superior border
]

# RA connectivity supplied directly by the user, cross-checked against the
# reference implementation's boundary files and internally verified: every
# region's edge set forms a properly closed loop (each of its landmarks has
# degree 2 within that set). Region 9: M-L-K-J-M. Region 10: L-K-N-O-L.
# Region 11: N-O-S-P-Q-N. Region 12: Q-N-T-U-Q. Region 13: L-S-P-Q-U-M-L.
# Region 14: N-K-J-V-T-N. Region 15: J-M-V-J.
RA_EDGES: list[tuple[str, str]] = [
    ("M", "L"),  # segment 9 / segment 13
    ("L", "K"),  # segment 9 / segment 10
    ("K", "J"),  # segment 9 / segment 14
    ("J", "M"),  # segment 9 / segment 15
    ("N", "O"),  # segment 10 / segment 11
    ("S", "P"),  # segment 11 / segment 13
    ("P", "Q"),  # segment 11 / segment 13
    ("Q", "N"),  # segment 11 / segment 12
    ("N", "T"),  # segment 12 / segment 14
    ("Q", "U"),  # segment 12 / segment 13
    ("V", "J"),  # segment 14 / segment 15
]

# Both landmarks sit on the same detected mesh boundary loop; the border is
# that rim arc, not a Dijkstra shortest path. Tricuspid annulus, clockwise
# from 1 o'clock: M -> V -> T -> U -> M. SVC orifice: S -> O -> L -> S.
#
# IVC only has two defining points (K, N - no third point the way SVC has
# S/O/L), so its two bordering segments (10 and 14) can't each get their own
# named arc the way the other holes do, and no reliable way was found to
# attribute one specific arc to one specific segment (tried an extra
# waypoint-based path, picking by raw arc length, and anchoring to the
# nearest landmark - all either added an unwanted click or turned out not
# to be anatomically meaningful, flipping between sessions). Instead, the
# *entire* IVC rim (both arcs together) is tagged once as "K_N" and shared
# by both segments 10 and 14 - each is still identified unambiguously by
# its other unique tags, so all the shared tag needs to confirm is "this
# piece touches the IVC hole," which is true for both sides equally.
RA_BOUNDARY_ARC_EDGES: list[ArcEdge] = [
    ArcEdge("U", "M"),  # segment 13 apical border (tricuspid annulus 11-1 o'clock)
    ArcEdge("M", "V"),  # segment 15 apical border (tricuspid annulus 1-5 o'clock)
    ArcEdge("V", "T"),  # segment 14 apical border (tricuspid annulus 5-8 o'clock)
    ArcEdge("T", "U"),  # segment 12 apical border (tricuspid annulus 8-11 o'clock)
    ArcEdge("S", "O"),  # segment 11 border (SVC orifice, anterior-lateral)
    ArcEdge("O", "L"),  # segment 10 border (SVC orifice, lateral-septal)
    ArcEdge("L", "S"),  # segment 13 border (SVC orifice, septal-anterior)
    ArcEdge("K", "N", whole_loop=True),  # segment 10 AND segment 14 border (IVC orifice, whole rim)
]

RA_LOOP_PATHS: list[tuple[str, str, str]] = []  # RA has no appendage-neck loop to route through

_EDGES = {"LA": LA_EDGES, "RA": RA_EDGES}
_ARC_EDGES = {"LA": LA_BOUNDARY_ARC_EDGES, "RA": RA_BOUNDARY_ARC_EDGES}
_LOOP_PATHS = {"LA": LA_LOOP_PATHS, "RA": RA_LOOP_PATHS}


def edge_name(a: str, b: str) -> str:
    return f"{a}_{b}"


def loop_path_name(a: str, loop_name: str, b: str) -> str:
    return f"{a}_{loop_name}_{b}"


# Every named curve that bounds a segment, mapped to the segment numbers it
# separates. Used to identify a flood-filled mesh piece by checking which
# curves border it, rather than by guessing a spatial seed point. A "loop"
# landmark (e.g. LAA_neck) bounds only its own enclosed segment.
#
# Each antrum neck is a closed loop with two arcs: the direct A-B/C-D
# geodesic (the "near" arc, facing the dome) borders segment 3; the
# *_antrum_outer path (the "far" arc, facing away from the dome) borders
# segment 6 / segment 8 respectively - confirmed by the reference boundary
# filenames left_venoatrial_junction_A_B_{sup,inf} / lateral_wall_A_B_inf /
# septal_wall_C_D_inf, where "inf" (far arc) is shared with the lateral /
# septal wall, not the dome. A curve can only ever separate two segments,
# so A_B and C_D must NOT also appear in segment 6's / segment 8's sets.
#
# The LAA_neck loop has two arcs relative to its composite path's two
# loop-attachment points (nearest-to-A and nearest-to-F): the short arc
# between them (already folded into the composite's own tag) and the long
# arc (the rest of the loop's circumference). The long arc genuinely
# borders whichever of segment 4 / segment 6 the composite doesn't - in
# practice this is consistently segment 6 (A's attachment point sits close
# to where segment 4 and 6 already meet, leaving segment 6 to wrap around
# most of the appendage base), so segment 6 alone also expects "LAA_neck"
# as well as the composite path.
LA_SEGMENT_BORDERS: dict[int, frozenset[str]] = {
    1: frozenset({edge_name("A", "B"), "LPV_antrum_outer"}),
    2: frozenset({edge_name("C", "D"), "RPV_antrum_outer"}),
    3: frozenset({edge_name("A", "C"), edge_name("B", "D"), edge_name("A", "B"), edge_name("C", "D")}),
    4: frozenset({edge_name("A", "C"), edge_name("E", "C"), edge_name("E", "F"), loop_path_name("A", "LAA_neck", "F")}),
    5: frozenset({"LAA_neck"}),
    6: frozenset({edge_name("F", "H"), edge_name("H", "B"), "LPV_antrum_outer",
                  loop_path_name("A", "LAA_neck", "F"), "LAA_neck"}),
    7: frozenset({edge_name("B", "D"), edge_name("H", "I"), edge_name("H", "B"), edge_name("D", "I")}),
    8: frozenset({edge_name("I", "E"), edge_name("E", "C"), edge_name("D", "I"), "RPV_antrum_outer"}),
}

# Segments 10 and 14 both expect "K_N" (see RA_BOUNDARY_ARC_EDGES above,
# whole_loop=True) - the IVC's whole rim, shared by both, since neither of
# its two arcs can be reliably attributed to one segment or the other. Each
# is still identified unambiguously via its own other, unique tags.
RA_SEGMENT_BORDERS: dict[int, frozenset[str]] = {
    9: frozenset({edge_name("M", "L"), edge_name("L", "K"), edge_name("K", "J"), edge_name("J", "M")}),
    10: frozenset({edge_name("L", "K"), edge_name("K", "N"), edge_name("N", "O"), edge_name("O", "L")}),
    11: frozenset({edge_name("N", "O"), edge_name("S", "O"), edge_name("S", "P"),
                    edge_name("P", "Q"), edge_name("Q", "N")}),
    12: frozenset({edge_name("Q", "N"), edge_name("N", "T"), edge_name("T", "U"), edge_name("Q", "U")}),
    13: frozenset({edge_name("L", "S"), edge_name("S", "P"), edge_name("P", "Q"),
                    edge_name("Q", "U"), edge_name("U", "M"), edge_name("M", "L")}),
    14: frozenset({edge_name("K", "N"), edge_name("K", "J"), edge_name("V", "J"),
                    edge_name("V", "T"), edge_name("N", "T")}),
    15: frozenset({edge_name("J", "M"), edge_name("M", "V"), edge_name("V", "J")}),
}

_SEGMENT_BORDERS = {"LA": LA_SEGMENT_BORDERS, "RA": RA_SEGMENT_BORDERS}


def get_segment_borders(chamber: str) -> dict[int, frozenset[str]]:
    chamber = chamber.upper()
    if chamber == "BOTH":
        return {**LA_SEGMENT_BORDERS, **RA_SEGMENT_BORDERS}
    return dict(_SEGMENT_BORDERS.get(chamber, {}))


def get_edges(chamber: str) -> list[tuple[str, str]]:
    chamber = chamber.upper()
    if chamber == "BOTH":
        return [*LA_EDGES, *RA_EDGES]
    return list(_EDGES.get(chamber, []))


def get_boundary_arc_edges(chamber: str) -> list[ArcEdge]:
    chamber = chamber.upper()
    if chamber == "BOTH":
        return [*LA_BOUNDARY_ARC_EDGES, *RA_BOUNDARY_ARC_EDGES]
    return list(_ARC_EDGES.get(chamber, []))


def get_loop_paths(chamber: str) -> list[tuple[str, str, str]]:
    chamber = chamber.upper()
    if chamber == "BOTH":
        return [*LA_LOOP_PATHS, *RA_LOOP_PATHS]
    return list(_LOOP_PATHS.get(chamber, []))
