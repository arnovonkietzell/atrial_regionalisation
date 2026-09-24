"""Mesh-topology helpers: finding the open boundary (hole) loops of a
clipped atrial surface (valve and vein orifices), and walking arcs of an
ordered cyclic point sequence (a detected boundary loop, or a user-clicked
loop such as the appendage neck)."""

from __future__ import annotations

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def order_boundary_loops(mesh: pv.PolyData, tol: float = 1e-4) -> list[list[int]]:
    """Return the open boundary edges of `mesh` as a list of closed loops,
    each a list of original mesh vertex ids in cyclic connectivity order."""
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    if edges.n_points == 0:
        return []

    tree = cKDTree(mesh.points)
    _, orig_ids = tree.query(edges.points, k=1)
    orig_ids = orig_ids.astype(int)

    adjacency: dict[int, list[int]] = {}
    lines = edges.lines
    i = 0
    while i < len(lines):
        npts = lines[i]
        pts = lines[i + 1 : i + 1 + npts]
        i += 1 + npts
        for a, b in zip(pts[:-1], pts[1:]):
            oa, ob = int(orig_ids[a]), int(orig_ids[b])
            adjacency.setdefault(oa, []).append(ob)
            adjacency.setdefault(ob, []).append(oa)

    visited: set[int] = set()
    loops: list[list[int]] = []
    for start in adjacency:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        prev, cur = None, start
        while True:
            candidates = [n for n in adjacency[cur] if n != prev]
            nxt = next((c for c in candidates if c not in visited), None)
            if nxt is None:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        loops.append(loop)
    return loops


def both_arcs_between(loop: list[int], id1: int, id2: int) -> tuple[list[int], list[int]]:
    """Return (shorter_arc, longer_arc): the two possible arcs of a cyclic
    point sequence `loop` connecting id1 to id2 (inclusive of both ends).
    Together they cover the loop's full circumference exactly once each
    way - what a two-point hole (e.g. the IVC orifice) needs: one arc for
    each of the two segments bordering it, with no need for an extra
    waypoint to disambiguate."""
    i1, i2 = loop.index(id1), loop.index(id2)
    forward = loop[i1 : i2 + 1] if i1 <= i2 else loop[i1:] + loop[: i2 + 1]
    other_fwd = loop[i2 : i1 + 1] if i2 <= i1 else loop[i2:] + loop[: i1 + 1]
    backward = list(reversed(other_fwd))
    return (forward, backward) if len(forward) <= len(backward) else (backward, forward)


def arc_between(loop: list[int], id1: int, id2: int, *, long: bool = False) -> list[int]:
    """Return the shorter arc between id1 and id2 by default, or the
    longer/complementary one if `long=True`. See `both_arcs_between`."""
    shorter, longer = both_arcs_between(loop, id1, id2)
    return longer if long else shorter




def nearest_on_point_list(vertex_ids: list[int], xyz_list, target_xyz) -> int:
    """Return the vertex id (from `vertex_ids`) whose coordinate is closest
    to `target_xyz`."""
    pts = np.asarray(xyz_list)
    d = np.linalg.norm(pts - np.asarray(target_xyz), axis=1)
    return int(vertex_ids[int(np.argmin(d))])


def dense_chain(mesh: pv.PolyData, clicked_ids: list[int], close: bool) -> list[int]:
    """Expand a sparse chain of clicked vertex ids into the full, densely
    ordered list of mesh vertex ids along the *actual drawn* curve - i.e.
    the chained geodesics between consecutive clicked points, not just the
    clicked points themselves. `close=True` treats it as a closed loop
    (e.g. the appendage neck); `close=False` as an open path (e.g. an
    antrum-outer path), keeping both endpoints."""
    n = len(clicked_ids)
    if n < 2:
        return list(clicked_ids)
    pairs = (
        [(clicked_ids[i], clicked_ids[(i + 1) % n]) for i in range(n)]
        if close
        else list(zip(clicked_ids[:-1], clicked_ids[1:]))
    )
    full: list[int] = []
    for a, b in pairs:
        if a == b:
            continue
        seg_ids = mesh.geodesic(a, b).point_data["vtkOriginalPointIds"].tolist()
        full.extend(seg_ids[:-1])  # drop last point: shared with the next segment's start
    if not close:
        full.append(clicked_ids[-1])
    return full


class BoundarySnapper:
    """Snaps an arbitrary 3D pick position to the nearest vertex on any open
    mesh boundary loop, and reports which loop a given vertex belongs to."""

    def __init__(self, mesh: pv.PolyData):
        self.mesh = mesh
        self.loops: list[list[int]] = order_boundary_loops(mesh)
        self.point_ids = np.array(
            sorted({v for loop in self.loops for v in loop}), dtype=int
        )
        self._tree = cKDTree(mesh.points[self.point_ids]) if len(self.point_ids) else None
        self._vertex_to_loop = {v: i for i, loop in enumerate(self.loops) for v in loop}

    @property
    def available(self) -> bool:
        return self._tree is not None

    def snap(self, xyz) -> int:
        """Return the mesh vertex id of the boundary point nearest to xyz."""
        if self._tree is None:
            raise RuntimeError("Mesh has no open boundary edges to snap to")
        _, local_idx = self._tree.query(xyz)
        return int(self.point_ids[local_idx])

    def loop_containing(self, vertex_id: int) -> list[int] | None:
        idx = self._vertex_to_loop.get(int(vertex_id))
        return self.loops[idx] if idx is not None else None
