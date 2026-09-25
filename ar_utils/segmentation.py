"""Cell-based region labeling.

Every drawn segment-boundary curve is expanded to its dense, mesh-exact
vertex sequence. Each *edge* of that sequence is then independently
classified by whether it actually has two adjacent cells (a real interior
edge - cut the cell-adjacency (dual) graph there, tagging both resulting
sides) or only one (an open mesh boundary edge - a hole has nothing on its
far side to cut, so its vertices are tagged directly, catching whichever
cells touch them on whichever side has real tissue). A curve whose
landmarks are properly snapped to a valve/vein rim ends up entirely the
second kind, walking that rim exactly like before; a curve whose landmarks
aren't on a detected boundary (the mesh may not be clipped open there)
falls back to the first kind, a plain geodesic - see geometry.smart_chain.

The resulting connected pieces are matched to segment numbers purely by
which curves border them - no spatial seed guessing required, since each
curve already carries the identity of the two segments it separates. A
component whose curve set doesn't match a "real" segment cleanly can still
match one of the fallback segments (16-19) that only exist to catch
leftover tissue enclosed when a boundary loop wasn't fully on the rim -
see the LA_SEGMENT_BORDERS/RA_SEGMENT_BORDERS comments in geodesics.py.

A cell gets label 0 if it falls in a piece that didn't match any segment
cleanly (topology incomplete, or chamber not fully landmarked) - never a
guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .geodesics import edge_name, get_segment_borders, get_subset_match_segments, loop_path_name
from .geometry import arc_between, nearest_on_point_list, smart_chain


def _iterate_cells(mesh) -> list[list[int]]:
    """Return each cell's vertex ids, in original cell order."""
    faces = mesh.faces
    cells: list[list[int]] = []
    i = 0
    while i < len(faces):
        npts = faces[i]
        cells.append([int(v) for v in faces[i + 1 : i + 1 + npts]])
        i += 1 + npts
    return cells


def _mesh_edge_to_cells(cells: list[list[int]]) -> dict[frozenset, list[int]]:
    edge_cells: dict[frozenset, list[int]] = {}
    for cell_id, verts in enumerate(cells):
        n = len(verts)
        for k in range(n):
            edge = frozenset((verts[k], verts[(k + 1) % n]))
            edge_cells.setdefault(edge, []).append(cell_id)
    return edge_cells


def collect_curves(app) -> dict[str, list[int]]:
    """Collect every drawn boundary curve as {name: dense ordered vertex
    ids}. Each segment of every multi-point curve (boundary_arc_edges,
    loops, paths) independently resolves to a rim arc if its two endpoints
    are on the same detected mesh boundary loop, or a plain geodesic
    otherwise (see geometry.smart_chain) - `label_regions` decides, per
    resulting edge (not per curve), whether that means a real graph cut or
    a rim-touch tag, since an open boundary edge only ever touches one
    cell and so can't be "cut" in the usual two-sided sense."""
    mesh = app.mesh
    boundary = app.boundary
    curves: dict[str, list[int]] = {}

    for a, b in app.edges:
        pa, pb = app.session.points[a], app.session.points[b]
        seg = mesh.geodesic(pa["vertex_id"], pb["vertex_id"])
        curves[edge_name(a, b)] = seg.point_data["vtkOriginalPointIds"].tolist()

    for arc_edge in app.boundary_arc_edges:
        a, b = arc_edge.a, arc_edge.b
        pa, pb = app.session.points[a], app.session.points[b]
        name = arc_edge.curve_name
        loop_a = boundary.loop_containing(pa["vertex_id"])
        loop_b = boundary.loop_containing(pb["vertex_id"])
        if loop_a is not None and loop_a == loop_b:
            curves[name] = arc_between(loop_a, pa["vertex_id"], pb["vertex_id"])
        else:
            curves[name] = mesh.geodesic(
                pa["vertex_id"], pb["vertex_id"]
            ).point_data["vtkOriginalPointIds"].tolist()
            print(f"{a} and {b} are not on the same mesh boundary rim; "
                  f"'{name}' will be a geodesic instead of a rim arc.")

    for a, loop_name, b in app.loop_paths:
        pa, pb = app.session.points[a], app.session.points[b]
        loop = app.session.loops[loop_name]
        full_ids = app._get_full_loop(loop_name, loop["vertex_ids"])
        full_xyz = mesh.points[full_ids]
        x_a = nearest_on_point_list(full_ids, full_xyz, pa["xyz"])
        x_b = nearest_on_point_list(full_ids, full_xyz, pb["xyz"])
        spoke_a = mesh.geodesic(pa["vertex_id"], x_a).point_data["vtkOriginalPointIds"].tolist()
        arc_mid = arc_between(full_ids, x_a, x_b)
        spoke_b = mesh.geodesic(x_b, pb["vertex_id"]).point_data["vtkOriginalPointIds"].tolist()
        curves[loop_path_name(a, loop_name, b)] = spoke_a[:-1] + arc_mid[:-1] + spoke_b

    # Only process session loops/paths that are still part of the current
    # landmark plan - a session file can carry stale entries left over from
    # an earlier version of the plan (e.g. a landmark that was since
    # replaced by a different mechanism), and those must not be treated as
    # real curves: they'd tag vertices that a *new* curve covering the same
    # territory also touches, incorrectly turning it into a junction.
    for name, data in app.session.loops.items():
        spec = app.plan_by_name.get(name)
        if spec is None or spec.kind != "loop":
            continue
        vertex_ids = data["vertex_ids"]
        if spec.seed_points:
            # e.g. KN_loop: prepend K and N's own already-placed vertices
            # before the operator's own clicked waypoints, closing back to
            # K (the first seed point) rather than to the first waypoint.
            seed_vertex_ids = [app.session.points[n]["vertex_id"] for n in spec.seed_points]
            vertex_ids = [*seed_vertex_ids, *vertex_ids]
        curves[name] = smart_chain(mesh, boundary, vertex_ids, close=True)

    for name in app.session.paths:
        spec = app.plan_by_name.get(name)
        if spec is None or spec.kind != "path":
            continue
        # data["vertex_ids"] holds only the operator's own waypoints -
        # path_start/path_end are resolved fresh from session.points, same
        # as a loop's seed_points (see landmarks.py), so an edited
        # start/end landmark is always reflected here.
        curves[name] = smart_chain(mesh, boundary, app._path_vertex_ids(name), close=False)

    return curves


@dataclass
class ComponentMatch:
    component_id: int
    n_cells: int
    borders: set[str]
    segment: int  # 0 if no clean match
    exact: bool
    best_score: int
    candidates: dict[int, int] = field(default_factory=dict)  # segment -> score, for diagnostics


def label_regions(app) -> tuple[np.ndarray, list[ComponentMatch], dict]:
    """Return (cell_labels, match_report, halo_stats). cell_labels has one
    entry per mesh cell, valued 1-15 (matched segment) or 0 (unmatched).
    halo_stats reports how many junction-vertex halo cells (see
    _propagate_into_halo) were resolved, left as a genuine conflict, or
    left isolated with no evidence at all."""
    mesh = app.mesh
    cells = _iterate_cells(mesh)
    n_cells = len(cells)
    edge_cells = _mesh_edge_to_cells(cells)

    curves = collect_curves(app)
    segment_borders = get_segment_borders(app.chamber)
    subset_match_segments = get_subset_match_segments(app.chamber)

    # A loop-composite path (e.g. A_LAA_neck_F) physically reuses some of the
    # same mesh edges as the loop it routes through (e.g. LAA_neck), since
    # its middle arc walks along that loop. Both curves would legitimately
    # tag those shared edges, which breaks exact-match border comparison for
    # segments on the composite's side. The composite name is the more
    # specific identity for that stretch, so it wins: tag composite paths
    # first (always as interior cuts - a composite never touches a real
    # mesh boundary edge), then skip any edge they've already claimed when
    # tagging everything else.
    composite_names = {loop_path_name(a, loop, b) for a, loop, b in app.loop_paths}
    cut_edge_tags: dict[frozenset, set[str]] = {}
    rim_vertex_tags: dict[int, set[str]] = {}
    claimed_edges: set[frozenset] = set()
    for name in composite_names:
        if name not in curves:
            continue
        for u, v in zip(curves[name], curves[name][1:]):
            edge = frozenset((u, v))
            cut_edge_tags.setdefault(edge, set()).add(name)
            claimed_edges.add(edge)

    # Every other curve is classified per *edge*, not per curve: an edge
    # with two adjacent cells is a real interior cut (tag both sides, once
    # they're known to differ - see component_cut_borders below); an edge
    # with only one (an open mesh boundary edge, since that's a hole with
    # nothing on its far side) can't be "cut" in that two-sided sense at
    # all, so its vertices are tagged directly instead, catching whichever
    # cells actually touch them (component_rim_borders below). This is what
    # lets one curve (e.g. KN_loop) be a rim arc along some stretches and a
    # real cut along others, depending on where the mesh actually has an
    # open boundary - see geometry.smart_chain and the module docstring.
    for name, ids in curves.items():
        if name in composite_names:
            continue
        for u, v in zip(ids, ids[1:]):
            edge = frozenset((u, v))
            if edge in claimed_edges:
                continue
            adj_cells = edge_cells.get(edge)
            if adj_cells and len(adj_cells) == 2:
                cut_edge_tags.setdefault(edge, set()).add(name)
            else:
                rim_vertex_tags.setdefault(u, set()).add(name)
                rim_vertex_tags.setdefault(v, set()).add(name)

    # Vertices touched by 2+ distinct curves (every named landmark, plus any
    # point where a composite path meets a loop) are junctions: only the
    # handful of mesh edges that happen to be a curve's own first step get
    # cut there, leaving the rest of that vertex's triangle fan uncut - a
    # leak straight between regions that should be separate. Cells touching
    # a junction vertex are excluded from the adjacency graph entirely
    # (never bridge two components), leaving a thin unlabeled halo there
    # instead of a false merge.
    #
    # A single curve can also revisit the same vertex twice without a
    # second curve ever being involved: a loop or multi-waypoint path is
    # built by chaining Dijkstra geodesics (or rim arcs) between consecutive
    # waypoints (see geometry.smart_chain), and if two non-adjacent segments
    # of that chain happen to cross - more likely the sparser the waypoints, since
    # each geodesic then has more freedom to bow away from the intended
    # route - only the mesh edges belonging to *those two segments* get cut
    # at the crossing vertex, not necessarily its whole edge ring. Counting
    # *distinct curve names* per vertex misses this entirely, since it's
    # still only one name; what actually matters is how many times the
    # vertex was visited in total, by any curve.
    vertex_curve_count: dict[int, set[str]] = {}
    self_crossing_vertices: set[int] = set()
    for name, ids in curves.items():
        seen_in_this_curve: set[int] = set()
        for v in ids:
            vertex_curve_count.setdefault(v, set()).add(name)
            if v in seen_in_this_curve:
                self_crossing_vertices.add(v)
            seen_in_this_curve.add(v)
    junction_vertices = {v for v, names in vertex_curve_count.items() if len(names) >= 2}
    junction_vertices |= self_crossing_vertices
    excluded_cells = {
        cell_id for cell_id, verts in enumerate(cells)
        if any(v in junction_vertices for v in verts)
    }

    rows, cols = [], []
    for edge, adj_cells in edge_cells.items():
        if len(adj_cells) != 2:
            continue  # open mesh boundary edge - nothing on the other side
        if edge in cut_edge_tags:
            continue  # explicit segment-boundary cut
        c0, c1 = adj_cells
        if c0 in excluded_cells or c1 in excluded_cells:
            continue  # never let a junction-adjacent cell bridge two pieces
        rows.append(c0)
        cols.append(c1)

    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_cells, n_cells))
    n_components, comp_labels = connected_components(graph, directed=False)

    component_cut_borders: list[set[str]] = [set() for _ in range(n_components)]
    for edge, tags in cut_edge_tags.items():
        adj_cells = edge_cells.get(edge)
        if not adj_cells or len(adj_cells) != 2:
            continue
        c0, c1 = adj_cells
        comp0, comp1 = comp_labels[c0], comp_labels[c1]
        if comp0 != comp1:
            component_cut_borders[comp0].update(tags)
            component_cut_borders[comp1].update(tags)

    component_rim_borders: list[set[str]] = [set() for _ in range(n_components)]
    for cell_id, verts in enumerate(cells):
        comp = comp_labels[cell_id]
        for v in verts:
            tags = rim_vertex_tags.get(v)
            if tags:
                component_rim_borders[comp].update(tags)

    # A component made up entirely of junction-halo cells was, by
    # definition, never actually connected to anything in the real matching
    # graph (that's exactly why those cells were excluded) - it's a
    # degenerate artifact of connected_components() still having to assign
    # *some* id to isolated nodes, not a real region. Its border-tag set can
    # coincidentally equal a genuine segment's signature (e.g. a single
    # stray cell whose only tagged edge is "LAA_neck" trivially "matches"
    # segment 5's entire expected set) despite representing nothing real.
    # Such components must never be matched directly; they're left at 0 and
    # handled only by neighbour-vote propagation into the halo below.
    comp_all_halo = np.ones(n_components, dtype=bool)
    for cell_id in range(n_cells):
        if cell_id not in excluded_cells:
            comp_all_halo[comp_labels[cell_id]] = False

    cell_labels = np.zeros(n_cells, dtype=int)
    report: list[ComponentMatch] = []
    for comp_id in range(n_components):
        n_comp_cells = int((comp_labels == comp_id).sum())
        if comp_all_halo[comp_id]:
            report.append(ComponentMatch(
                component_id=comp_id, n_cells=n_comp_cells, borders=set(),
                segment=0, exact=False, best_score=0, candidates={},
            ))
            continue
        borders = component_cut_borders[comp_id] | component_rim_borders[comp_id]
        candidates: dict[int, int] = {}
        for seg, expected in segment_borders.items():
            score = len(borders & expected) - len(expected - borders) - len(borders - expected)
            candidates[seg] = score
        best_seg = max(candidates, key=candidates.get) if candidates else 0
        best_score = candidates.get(best_seg, -1)
        expected = segment_borders.get(best_seg, frozenset())
        if best_seg in subset_match_segments:
            # See geodesics.get_subset_match_segments: these accept any
            # non-empty subset of their expected tag set, not just an
            # exact match - real segments (not in this set) stay strict
            # exact-match only, since partial evidence there could just as
            # easily mean a genuinely broken/incomplete cut.
            exact = len(borders) > 0 and borders <= expected
        else:
            exact = best_score == len(expected) and borders == expected
        if exact:
            cell_labels[comp_labels == comp_id] = best_seg
        report.append(ComponentMatch(
            component_id=comp_id, n_cells=n_comp_cells, borders=borders,
            segment=best_seg if exact else 0, exact=exact, best_score=best_score,
            candidates=candidates,
        ))

    n_halo, n_resolved, n_conflicted, n_isolated = _propagate_into_halo(
        cell_labels, cells, edge_cells, cut_edge_tags, excluded_cells
    )
    halo_stats = {
        "halo_cells": n_halo, "resolved": n_resolved,
        "conflicted": n_conflicted, "isolated": n_isolated,
    }

    return cell_labels, report, halo_stats


def _propagate_into_halo(
    cell_labels: np.ndarray,
    cells: list[list[int]],
    edge_cells: dict[frozenset, list[int]],
    cut_edge_tags: dict[frozenset, set[str]],
    excluded_cells: set[int],
) -> tuple[int, int, int, int]:
    """Fill in halo cells (excluded from the matching graph to avoid false
    bridges) whose *actual* neighbours - across real, non-cut mesh edges,
    now that neighbouring pieces have been identified - unanimously agree
    on one label. A halo cell touching two disagreeing labelled neighbours
    is a genuine leak point (not resolvable from local evidence) and is
    left at 0 permanently, never guessed. A halo cell that never touches any
    confirmed neighbour at all (e.g. fully surrounded by other unresolved
    halo cells) is also left at 0, counted separately as "isolated".

    This mutates `cell_labels` in place and returns
    (n_halo_cells, n_resolved, n_conflicted, n_isolated).
    """
    # Full non-cut adjacency, unlike the matching graph this does NOT
    # exclude halo cells - we need to see what a halo cell actually touches.
    full_neighbors: dict[int, list[int]] = {}
    for edge, adj_cells in edge_cells.items():
        if len(adj_cells) != 2 or edge in cut_edge_tags:
            continue
        c0, c1 = adj_cells
        full_neighbors.setdefault(c0, []).append(c1)
        full_neighbors.setdefault(c1, []).append(c0)

    # Multi-source BFS, processed strictly in distance-order rounds: each
    # round only looks at cells adjacent to the *previous* round's frontier,
    # and collects every vote reaching a cell in that round before deciding.
    # This is what makes it order-independent - a cell is examined exactly
    # once, at its true minimum distance from a confirmed region, with every
    # same-distance neighbour's vote counted simultaneously. (A naive greedy
    # "resolve as soon as one neighbour is known" pass is order-dependent: a
    # cell can lock in a label from whichever neighbour happens to resolve
    # first, then never get reconsidered when a conflicting neighbour
    # resolves in a later pass - which is exactly the bug this replaced.)
    resolved = {i: int(cell_labels[i]) for i in range(len(cells)) if cell_labels[i] != 0}
    frozen: set[int] = set()
    unresolved = {c for c in excluded_cells if c not in resolved}

    frontier = set(resolved)
    while frontier and unresolved:
        votes: dict[int, set[int]] = {}
        for src in frontier:
            src_label = resolved[src]
            for nb in full_neighbors.get(src, ()):
                if nb in unresolved:
                    votes.setdefault(nb, set()).add(src_label)
        next_frontier = set()
        for cell, cell_votes in votes.items():
            if len(cell_votes) == 1:
                resolved[cell] = next(iter(cell_votes))
                next_frontier.add(cell)
            else:
                frozen.add(cell)
            unresolved.discard(cell)
        frontier = next_frontier

    for cell, label in resolved.items():
        cell_labels[cell] = label

    # Anything still in `pending` never touched a confirmed neighbour at all
    # (e.g. a cell fully surrounded by other still-unresolved halo cells) -
    # neither resolved nor a detected conflict, just no evidence reaching it.
    n_halo = len(excluded_cells)
    n_resolved = sum(1 for c in excluded_cells if cell_labels[c] != 0)
    n_conflicted = len(frozen)
    n_isolated = n_halo - n_resolved - n_conflicted
    return n_halo, n_resolved, n_conflicted, n_isolated
