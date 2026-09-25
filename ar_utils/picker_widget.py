"""Interactive Qt window for placing the named landmarks of the EHRA/EACVI
15-segment bi-atrial model on an atrial surface mesh, and computing the
resulting cell-based region labelling.

This is a `QMainWindow` (using `pyvistaqt.QtInteractor` for the 3D view) so
it can be embedded directly in an EP Workbench WIP, which runs inside a Qt
application - see `wips/atrial_regionalisation/main.py`.

Usage
-----
Right click        Place the current landmark at the nearest visible mesh
                    vertex. (Left click is left free for rotating the
                    camera.)
Shift+right click  Same, but snap to the nearest point on an open mesh
                    boundary (a valve/vein orifice rim) instead of the
                    raw click location. Available for every landmark and
                    waypoint, not just ones that are normally on a rim -
                    segmentation checks which points actually ended up on
                    a boundary for itself, rather than assuming from the
                    landmark, since that depends on how the mesh was
                    clipped (see segmentation.py).
Finish loop/path    Finish the current loop / path landmark. [N]
Undo                Undo the last action - a placed point, a finished
                    loop/path, or an edit-mode move, whichever happened
                    most recently. [U]
Reset all           Clear every placed landmark and start over. [X]
Edit mode           Once every landmark is placed, right click switches to
                    editing: right click a landmark point or a loop/path
                    waypoint to select it (it turns gold), then right
                    click a new location on the mesh to move it there
                    (shift+right click to snap, same as normal placement).
                    Right-clicking a different point instead of a
                    destination changes the selection; clicking the same
                    one again deselects it. Works both before and after
                    "Compute regions" - editing after a computed result
                    discards it, same as Undo, and it must be recomputed.
Compute regions     Once all landmarks are placed: compute the segment
                    regions (cell-based), colour the mesh by region, and
                    label each region with its number for a quick visual
                    check. [C]
Confirm & Close     Finalize the computed labels (`self.cell_labels`) and
                    close the window. Closing the window any other way
                    (the title bar's close button) leaves `self.cell_labels`
                    as `None`, signalling to the caller that nothing should
                    be output. [Y]

A reference image tracks the current landmark automatically in the
right-hand panel; once every landmark is placed it switches to the
chamber's full 15-segment reference figure, for comparing against the
computed regions.
"""

from __future__ import annotations

import textwrap
from typing import Callable

import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from .geodesics import get_boundary_arc_edges, get_edges, get_loop_paths
from .geometry import BoundarySnapper, arc_between, nearest_on_point_list, smart_chain
from .hints import final_image_path_for, image_path_for
from .landmarks import LandmarkSpec, get_landmark_plan
from .session import LandmarkState

POINT_COLOR = "red"
LOOP_COLOR = "orange"
DONE_COLOR = "seagreen"
GEODESIC_COLOR = "dodgerblue"
EDIT_SELECT_COLOR = "gold"

# One fixed colour per label, index = label value, covering the full 1-19
# range so LA (1-8, plus 16 - see below), RA (9-15, plus 17-19), and a
# future "both" chamber all get a complete, consistent legend regardless
# of which subset is present in a given mesh. Index 0 (unmatched/
# unlabelled) is a flat light grey, kept visually distinct from every real
# segment colour. 1-15 are Sasha Trubetskoy's "20 distinct colours" set (a
# hand-tuned categorical palette, not an evenly-spaced hue wheel) - chosen
# for maximum contrast between neighbouring labels rather than smooth/
# pretty transitions, which is what actually matters when comparing
# adjacent regions on the mesh. 16-19 use 4 more colours from that same
# 20-colour set (lavender/beige/mint/apricot - skipping its remaining
# grey, too close to 0's own light grey) for LA_SEGMENT_BORDERS[16] and
# RA_SEGMENT_BORDERS[17-19]: the leftover-tissue segments that only ever
# appear if a boundary landmark wasn't actually snapped to its rim (see
# geodesics.py) - normally these never appear at all.
REGION_COLORS = [
    "#cfcfcf",  # 0: unlabelled
    "#e6194b",  # 1 red
    "#3cb44b",  # 2 green
    "#ffe119",  # 3 yellow
    "#4363d8",  # 4 blue
    "#f58231",  # 5 orange
    "#911eb4",  # 6 purple
    "#42d4f4",  # 7 cyan
    "#f032e6",  # 8 magenta
    "#bfef45",  # 9 lime
    "#fabed4",  # 10 pink
    "#469990",  # 11 teal
    "#9a6324",  # 12 brown
    "#800000",  # 13 maroon
    "#808000",  # 14 olive
    "#000075",  # 15 navy
    "#dcbeff",  # 16 lavender
    "#fffac8",  # 17 beige
    "#aaffc3",  # 18 mint
    "#ffd8b1",  # 19 apricot
]


class LandmarkPickerWidget(QMainWindow):
    """Main WIP window: 3D landmark picker (left) + status/hint panel (right).

    A WIP's `main.py` constructs this from the input case's mesh, shows it,
    and runs the Qt event loop. Once the operator has placed every
    landmark, pressed "Compute regions", and pressed "Confirm & Close",
    `self.cell_labels` holds the final per-cell segment array (0 =
    unmatched) and `self.mesh` is the same mesh object with a `Regions`
    cell array attached - `main.py` uses these to build the output case.
    `self.cell_labels` stays `None` if the window is closed any other way.
    """

    def __init__(self, mesh: pv.PolyData, chamber: str, parent=None):
        super().__init__(parent)
        self.chamber = chamber.upper()
        self.setWindowTitle(f"Atrial Regionalisation - {self.chamber}")
        self.resize(1400, 900)

        self.mesh = mesh
        self.plan: list[LandmarkSpec] = get_landmark_plan(self.chamber)
        self.plan_by_name = {spec.name: spec for spec in self.plan}
        self.session = LandmarkState(chamber=self.chamber)
        self.boundary = BoundarySnapper(self.mesh)
        self.edges = get_edges(self.chamber)
        self.boundary_arc_edges = get_boundary_arc_edges(self.chamber)
        self.loop_paths = get_loop_paths(self.chamber)

        self._loop_buffer: list[tuple[int, list[float]]] = []
        self._full_loop_cache: dict[str, tuple[tuple[int, ...], list[int]]] = {}
        # Edit mode (active once every landmark is placed - see
        # _edit_mode_active): _edit_selected holds the ref of the landmark
        # point or loop/path waypoint currently picked for moving, if any -
        # ("point", name) | ("loop", name, idx) | ("path", name, idx).
        # _marker_refs maps every such actor's pyvista name back to its ref,
        # both to resolve a pick (via _ref_for_actor) and to know which
        # actors to toggle pickable on/off as edit mode is entered/left
        # (_sync_marker_pickability).
        self._edit_selected: tuple | None = None
        self._marker_refs: dict[str, tuple] = {}
        self.last_region_labels: np.ndarray | None = None
        self.cell_labels: np.ndarray | None = None  # set only by _confirm_and_close
        self.index = 0
        self._advance_to_next_pending()
        # Stack of zero-arg closures, each fully reversing one finalized
        # action (place a point, finish a loop/path) in the order they
        # happened - see _undo(). The in-progress loop/path waypoint buffer
        # (_loop_buffer) is popped separately, ahead of this stack, since
        # it isn't part of the session yet.
        self._undo_stack: list[Callable[[], None]] = []

        self._current_hint_image: str | None = None
        self._current_hint_pixmap: QPixmap | None = None

        self._build_gui()
        self._redraw_existing_landmarks()
        self._refresh_status()
        self._update_geodesics()

        self.plotter.enable_point_picking(
            callback=self._on_pick,
            picker="cell",
            left_clicking=False,
            use_picker=True,
            show_message=False,
            show_point=False,
            tolerance=0.01,
        )

        # Keyboard shortcuts, bound both through the VTK interactor (so they
        # work while the 3D view has focus) and as Qt button shortcuts (so
        # they work while the side panel has focus) - belt and suspenders,
        # since which one has keyboard focus isn't always obvious to the
        # operator. Letters avoid VTK's own trackball-camera defaults (e, f,
        # p, q, r, s, w, 3, ...) - see the module docstring for why that
        # matters (the 'f' collision silently mis-placed landmarks before).
        self.plotter.add_key_event("n", self._finish_multi)
        self.plotter.add_key_event("u", self._undo)
        self.plotter.add_key_event("x", self._reset)
        self.plotter.add_key_event("c", self._compute_and_preview_regions)
        self.plotter.add_key_event("y", self._confirm_and_close)

    # -- GUI scaffolding ---------------------------------------------------

    def _build_gui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor, stretch=3)
        self.plotter.set_background("white")
        self.plotter.add_mesh(
            self.mesh, color="lightgray", show_edges=False, smooth_shading=True, pickable=True,
            name="main_mesh",
        )
        self.plotter.add_axes()

        panel = QWidget()
        panel.setMinimumWidth(380)
        panel_layout = QVBoxLayout(panel)
        layout.addWidget(panel, stretch=2)

        self.headline_label = QLabel("")
        self.headline_label.setWordWrap(True)
        self.headline_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        panel_layout.addWidget(self.headline_label)

        self.details_label = QLabel("")
        self.details_label.setWordWrap(True)
        self.details_label.setStyleSheet("font-size: 13px;")
        panel_layout.addWidget(self.details_label)

        self.checklist_label = QLabel("")
        self.checklist_label.setWordWrap(True)
        self.checklist_label.setStyleSheet("font-size: 12px; color: #444;")
        panel_layout.addWidget(self.checklist_label)

        hint_frame = QFrame()
        hint_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        hint_layout = QVBoxLayout(hint_frame)
        self.hint_image_label = QLabel("")
        self.hint_image_label.setAlignment(Qt.AlignCenter)
        self.hint_image_label.setMinimumHeight(260)
        self.hint_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hint_layout.addWidget(self.hint_image_label)
        panel_layout.addWidget(hint_frame, stretch=1)

        button_row = QHBoxLayout()
        self.finish_button = QPushButton("Finish loop/path (N)")
        self.finish_button.setShortcut(QKeySequence("N"))
        self.finish_button.clicked.connect(self._finish_multi)
        button_row.addWidget(self.finish_button)
        self.undo_button = QPushButton("Undo (U)")
        self.undo_button.setShortcut(QKeySequence("U"))
        self.undo_button.clicked.connect(self._undo)
        button_row.addWidget(self.undo_button)
        panel_layout.addLayout(button_row)

        self.reset_button = QPushButton("Reset all landmarks (X)")
        self.reset_button.setShortcut(QKeySequence("X"))
        self.reset_button.clicked.connect(self._reset)
        panel_layout.addWidget(self.reset_button)

        self.compute_button = QPushButton("Compute regions (C)")
        self.compute_button.setShortcut(QKeySequence("C"))
        self.compute_button.clicked.connect(self._compute_and_preview_regions)
        panel_layout.addWidget(self.compute_button)

        self.confirm_button = QPushButton("Confirm && Close (Y)")
        self.confirm_button.setShortcut(QKeySequence("Y"))
        self.confirm_button.setEnabled(False)
        self.confirm_button.clicked.connect(self._confirm_and_close)
        panel_layout.addWidget(self.confirm_button)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._rescale_hint_pixmap()

    def _rescale_hint_pixmap(self) -> None:
        if self._current_hint_pixmap is None:
            return
        self.hint_image_label.setPixmap(
            self._current_hint_pixmap.scaled(
                self.hint_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    # -- plan bookkeeping ---------------------------------------------------

    def _advance_to_next_pending(self) -> None:
        while self.index < len(self.plan) and self.session.has(self.plan[self.index].name):
            self.index += 1

    @property
    def current(self) -> LandmarkSpec | None:
        if self.index >= len(self.plan):
            return None
        return self.plan[self.index]

    def load_session(self, session: LandmarkState) -> None:
        """Replace the current (empty) session with a previously captured
        one and bring the rest of the window's state in line with it -
        `main.py`'s debug-mode runner uses this to restore a session saved
        as JSON on a previous run, purely as a debugging convenience."""
        self.session = session
        self.index = 0
        self._edit_selected = None
        self._marker_refs.clear()
        self._advance_to_next_pending()
        self._redraw_existing_landmarks()
        self._refresh_status()
        self._update_geodesics()

    # -- picking callbacks --------------------------------------------------

    def _on_pick(self, point, picker) -> None:
        current = self.current
        if current is None:
            self._on_pick_edit(point, picker)
            return

        if self._shift_held():
            if not self.boundary.available:
                print(f"{current.name}: mesh has no open boundary edges to snap to.")
                return
            vertex_id = self.boundary.snap(point)
        else:
            vertex_id = picker.GetPointId()
            if vertex_id is None or vertex_id < 0:
                return
        xyz = self.mesh.points[vertex_id]

        if current.kind == "point":
            prior_index = self.index
            self.session.set_point(current.name, vertex_id, xyz)
            self._draw_point_marker(current.name, xyz, POINT_COLOR)
            self.index += 1
            self._advance_to_next_pending()
            self._push_undo(lambda name=current.name, idx=prior_index: self._undo_place_point(name, idx))
        else:  # loop or path
            self._loop_buffer.append((vertex_id, list(xyz)))
            self._draw_buffer_progress(current.name)

        self._refresh_status()
        self._update_geodesics()

    def _shift_held(self) -> bool:
        return bool(self.plotter.iren.interactor.GetShiftKey())

    # -- edit mode: moving an already-placed point --------------------------

    def _edit_mode_active(self) -> bool:
        return self.current is None

    def _ref_for_actor(self, actor) -> tuple | None:
        """Resolve a vtk actor (as returned by a picker) back to the
        (kind, name[, idx]) ref of the movable point it represents, or None
        if it isn't one (e.g. it's the main mesh, or nothing was hit)."""
        if actor is None:
            return None
        for name, act in self.plotter.renderer.actors.items():
            if act is actor:
                return self._marker_refs.get(name)
        return None

    def _actor_name_for_ref(self, ref: tuple) -> str:
        if ref[0] == "point":
            return f"marker_{ref[1]}"
        return f"wp_{ref[1]}_{ref[2]}"

    def _marker_xyz(self, ref: tuple) -> list[float]:
        kind, name, *rest = ref
        if kind == "point":
            return self.session.points[name]["xyz"]
        if kind == "loop":
            return self.session.loops[name]["xyz"][rest[0]]
        return self.session.paths[name]["xyz"][rest[0]]

    def _marker_vertex_id(self, ref: tuple) -> int:
        kind, name, *rest = ref
        if kind == "point":
            return self.session.points[name]["vertex_id"]
        if kind == "loop":
            return self.session.loops[name]["vertex_ids"][rest[0]]
        return self.session.paths[name]["vertex_ids"][rest[0]]

    def _edit_label(self, ref: tuple) -> str:
        if ref[0] == "point":
            return ref[1]
        kind_word = "loop" if ref[0] == "loop" else "path"
        return f"{ref[1]} {kind_word} waypoint {ref[2] + 1}"

    def _highlight_edit_selection(self, ref: tuple) -> None:
        name, xyz = self._actor_name_for_ref(ref), self._marker_xyz(ref)
        self._add_marker_sphere(name, xyz, EDIT_SELECT_COLOR, ref=ref, pickable=True)

    def _unhighlight_edit_selection(self, ref: tuple) -> None:
        name, xyz = self._actor_name_for_ref(ref), self._marker_xyz(ref)
        self._add_marker_sphere(name, xyz, DONE_COLOR, ref=ref, pickable=True)

    def _apply_edit_move(self, ref: tuple, vertex_id: int, xyz) -> None:
        kind, name, *rest = ref
        if kind == "point":
            self.session.set_point(name, vertex_id, xyz)
        elif kind == "loop":
            data = self.session.loops[name]
            data["vertex_ids"][rest[0]] = int(vertex_id)
            data["xyz"][rest[0]] = [float(v) for v in xyz]
        else:  # path
            data = self.session.paths[name]
            data["vertex_ids"][rest[0]] = int(vertex_id)
            data["xyz"][rest[0]] = [float(v) for v in xyz]

    def _on_pick_edit(self, point, picker) -> None:
        actor = picker.GetActor()
        ref = self._ref_for_actor(actor)

        if self._edit_selected is None:
            if ref is None:
                return  # clicked the bare mesh with nothing selected - no-op
            self._edit_selected = ref
            self._highlight_edit_selection(ref)
            self._refresh_status()
            return

        if ref is not None:
            # Clicked a point instead of a destination - change (or, on a
            # repeat click of the same one, cancel) the selection rather
            # than treating that point's own position as a move target.
            prior = self._edit_selected
            self._unhighlight_edit_selection(prior)
            self._edit_selected = None if ref == prior else ref
            if self._edit_selected is not None:
                self._highlight_edit_selection(self._edit_selected)
            self._refresh_status()
            return

        if self._shift_held():
            if not self.boundary.available:
                print("Mesh has no open boundary edges to snap to.")
                return
            vertex_id = self.boundary.snap(point)
        else:
            vertex_id = picker.GetPointId()
            if vertex_id is None or vertex_id < 0:
                return
        xyz = self.mesh.points[vertex_id]

        ref = self._edit_selected
        prior_vertex_id = self._marker_vertex_id(ref)
        prior_xyz = list(self._marker_xyz(ref))
        self._apply_edit_move(ref, vertex_id, xyz)
        print(f"Moved {self._edit_label(ref)}.")
        self._push_undo(
            lambda r=ref, pvid=prior_vertex_id, pxyz=prior_xyz: self._undo_move(r, pvid, pxyz)
        )
        self._edit_selected = None
        self._invalidate_computed_regions()
        self._redraw_existing_landmarks()
        self._refresh_status()
        self._update_geodesics()

    def _undo_move(self, ref: tuple, prior_vertex_id: int, prior_xyz) -> None:
        self._apply_edit_move(ref, prior_vertex_id, prior_xyz)
        self._redraw_existing_landmarks()

    def _push_undo(self, reverse: Callable[[], None]) -> None:
        self._undo_stack.append(reverse)

    def _undo_place_point(self, name: str, prior_index: int) -> None:
        self.session.remove(name)
        self._clear_landmark_actors(name)
        self.index = prior_index

    def _invalidate_computed_regions(self) -> None:
        """Discard a previous 'Compute regions' result after the landmark
        set changes underneath it (e.g. undoing a placed landmark) - it
        would otherwise silently describe stale geometry."""
        if self.last_region_labels is None:
            return
        self.last_region_labels = None
        self.confirm_button.setEnabled(False)
        self.plotter.remove_actor("region_number_labels")
        if self.plotter.scalar_bars:
            self.plotter.remove_scalar_bar()
        if "Regions" in self.mesh.cell_data:
            self.mesh.cell_data.remove("Regions")
        if "Regions_display" in self.mesh.cell_data:
            self.mesh.cell_data.remove("Regions_display")
        self.plotter.add_mesh(
            self.mesh, color="lightgray", show_edges=False, smooth_shading=True, pickable=True,
            name="main_mesh",
        )
        print("Landmarks changed - regions must be recomputed.")

    def _finish_multi(self) -> None:
        current = self.current
        if current is None or current.kind not in ("loop", "path"):
            return

        prior_index = self.index
        prior_buffer = list(self._loop_buffer)  # snapshot, for undo

        if current.kind == "loop":
            # seed_points (e.g. KN_loop's K, N) count toward the minimum of
            # 3 but aren't stored in the buffer/session.loops themselves -
            # they're resolved fresh from session.points wherever needed
            # (see _loop_vertex_ids), so the loop always reflects their
            # current position rather than a stale baked-in copy.
            n_needed = max(0, 3 - len(current.seed_points))
            if len(self._loop_buffer) < n_needed:
                print(f"Need at least {n_needed} more point(s) to close the "
                      f"{current.name} loop ({len(self._loop_buffer)} of at least "
                      f"{n_needed} placed).")
                return
            vertex_ids = [v for v, _ in self._loop_buffer]
            xyz_list = [xyz for _, xyz in self._loop_buffer]
            self.session.set_loop(current.name, vertex_ids, xyz_list)
            self._draw_loop_final(current.name)
        else:  # path
            if len(self._loop_buffer) < 1:
                print(f"Need at least 1 waypoint for {current.name} "
                      f"({len(self._loop_buffer)} placed).")
                return
            # Only the operator's own waypoints are stored - path_start/
            # path_end are resolved fresh from session.points wherever the
            # full path is needed (_path_vertex_ids), same as a loop's
            # seed_points, so editing that landmark afterwards stays in sync.
            vertex_ids = [v for v, _ in self._loop_buffer]
            xyz_list = [xyz for _, xyz in self._loop_buffer]
            self.session.set_path(current.name, vertex_ids, xyz_list)
            self._draw_path_final(current.name)

        self._loop_buffer = []
        self.index += 1
        self._advance_to_next_pending()
        self._push_undo(
            lambda name=current.name, idx=prior_index, buf=prior_buffer:
            self._undo_finish_multi(name, idx, buf)
        )
        self._refresh_status()
        self._update_geodesics()

    def _undo_finish_multi(self, name: str, prior_index: int, prior_buffer: list) -> None:
        self.session.remove(name)
        self._clear_landmark_actors(name)
        self.index = prior_index
        self._loop_buffer = list(prior_buffer)
        if self._loop_buffer:
            self._draw_buffer_progress(name)

    def _undo(self) -> None:
        if self._loop_buffer:
            self._loop_buffer.pop()
            current = self.current
            self._clear_loop_actors(current.name)
            if self._loop_buffer:
                self._draw_buffer_progress(current.name)
            self._refresh_status()
            self._update_geodesics()
            return

        if not self._undo_stack:
            print("Nothing to undo.")
            return
        reverse = self._undo_stack.pop()
        reverse()
        self._edit_selected = None  # the undone landmark may have been the selected one
        self._invalidate_computed_regions()
        self._refresh_status()
        self._update_geodesics()

    def _reset(self) -> None:
        for spec in self.plan:
            self._clear_landmark_actors(spec.name)
        self.session.points.clear()
        self.session.loops.clear()
        self.session.paths.clear()
        self._loop_buffer = []
        self._full_loop_cache.clear()
        self._undo_stack.clear()
        self._edit_selected = None
        self._marker_refs.clear()
        self.index = 0
        self._invalidate_computed_regions()
        self._refresh_status()
        self._update_geodesics()
        print("Landmarks reset.")

    def _compute_and_preview_regions(self) -> None:
        if self.current is not None:
            QMessageBox.warning(
                self, "Landmarks incomplete",
                f"Place all landmarks first (still need: {self.current.name}).",
            )
            return

        from .segmentation import label_regions

        labels, report, halo_stats = label_regions(self)
        self.last_region_labels = labels
        self.mesh.cell_data["Regions"] = labels

        n_matched = sum(1 for m in report if m.exact)
        n_cells_matched = int((labels != 0).sum())
        print(
            f"Regions computed: {len(report)} connected piece(s), "
            f"{n_matched} matched to a segment, "
            f"{n_cells_matched}/{len(labels)} cells labelled."
        )
        print(
            f"Junction halo: {halo_stats['halo_cells']} cells excluded from matching, "
            f"{halo_stats['resolved']} filled in by unanimous-neighbour propagation, "
            f"{halo_stats['conflicted']} left unlabelled (touch 2+ disagreeing regions - a real gap), "
            f"{halo_stats['isolated']} left unlabelled (no confirmed neighbour reached them at all)."
        )
        # Pieces below this size are almost always junction-vertex halo
        # fragments (see segmentation.py's docstring) rather than a real
        # mislabeled region - report them as one summary line instead of
        # flooding the console.
        noise_threshold = 5
        notable = sorted((m for m in report if m.n_cells >= noise_threshold), key=lambda r: -r.n_cells)
        tiny = [m for m in report if m.n_cells < noise_threshold]
        for m in notable:
            status = f"-> segment {m.segment}" if m.exact else "-> UNMATCHED"
            print(f"  piece {m.component_id}: {m.n_cells} cells, borders={sorted(m.borders)} {status}")
        if tiny:
            tiny_cells = sum(m.n_cells for m in tiny)
            print(f"  + {len(tiny)} tiny piece(s) (<{noise_threshold} cells, {tiny_cells} cells "
                  "total, mostly resolved above)")

        # Colour range covers only the segments actually present in this
        # result: the chamber's normal contiguous range (1-8 for LA, 9-15
        # for RA), plus its leftover-tissue segment(s) (16 for LA; 17-19
        # for RA - see geodesics.py) if and only if any of those actually
        # occur here, which is the unusual case (a boundary landmark
        # wasn't snapped to its rim). Present values are remapped to a
        # contiguous 0..n display range purely for colouring/the scalar
        # bar, so the legend never shows a gap for chamber values that
        # can't occur (e.g. LA never has 9-15) or, in the normal case,
        # doesn't occur this time (e.g. 16 not present) - `self.mesh`'s own
        # "Regions" cell data keeps the true segment numbers throughout.
        lo, hi = {"LA": (1, 8), "RA": (9, 15)}.get(self.chamber, (1, 15))
        extra = {"LA": (16,), "RA": (17, 18, 19)}.get(self.chamber, ())
        valid = set(range(lo, hi + 1)) | set(extra)
        present = sorted(valid & set(int(v) for v in labels))
        if not present:
            print("Nothing matched - nothing to colour.")
            return
        display_index = {v: i for i, v in enumerate(present)}
        display_labels = np.array([display_index.get(int(v), -1) for v in labels])
        self.mesh.cell_data["Regions_display"] = display_labels

        # `n_labels` spaces ticks evenly across the *continuous* clim range,
        # including its fractional endpoints - it doesn't know the integers
        # are what matter. `annotations` instead pins each tick to an exact
        # value, so the scalar bar lands exactly on the present segment
        # numbers with no fractional labels. `n_labels: 0` additionally
        # suppresses vtk's own automatic tick labels (which otherwise still
        # draw alongside the annotations, at positions that don't line up
        # with the colour block centres).
        if self.plotter.scalar_bars:
            self.plotter.remove_scalar_bar()
        self.plotter.add_mesh(
            self.mesh, scalars="Regions_display", cmap=[REGION_COLORS[v] for v in present],
            clim=[-0.5, len(present) - 0.5],
            show_edges=False, smooth_shading=False, pickable=True, name="main_mesh",
            scalar_bar_args={"title": "Region", "fmt": "%.0f", "n_labels": 0},
            annotations={float(i): str(v) for i, v in enumerate(present)},
        )
        self._draw_region_number_labels(labels, present)
        self.confirm_button.setEnabled(True)

    def _draw_region_number_labels(self, labels, present: list[int]) -> None:
        """Float each present segment's number just outside the surface, so
        the computed regions can be visually cross-checked against the
        reference figure (which numbers segments the same way).

        The label is anchored to an *actual cell* of the region - the one
        whose own centroid is closest to the region's mean centroid - not
        to the mean centroid itself. For a concave or elongated region the
        mean centroid can land off the surface entirely, so averaging
        normals over the whole region and offsetting from that mean point
        routinely put the label nowhere near the actual mesh. Using one
        real cell's own centroid + normal guarantees the label starts on
        the surface and is offset in a direction that's locally correct
        there."""
        self.plotter.remove_actor("region_number_labels")
        if not present:
            return
        mesh = self.mesh
        if "Normals" not in mesh.cell_data:
            mesh = mesh.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=True)
        centers = mesh.cell_centers().points
        normals = mesh.cell_data["Normals"]
        offset = 0.015 * mesh.length  # mesh.length = bounding-box diagonal, so this scales with mesh size
        points = []
        for seg in present:
            mask = labels == seg
            seg_centers = centers[mask]
            seg_normals = normals[mask]
            mean_centroid = seg_centers.mean(axis=0)
            nearest_idx = int(np.argmin(np.linalg.norm(seg_centers - mean_centroid, axis=1)))
            anchor = seg_centers[nearest_idx]
            normal = seg_normals[nearest_idx]
            norm_len = np.linalg.norm(normal)
            normal = normal / norm_len if norm_len > 1e-9 else np.array([0.0, 0.0, 1.0])
            points.append(anchor + normal * offset)
        self.plotter.add_point_labels(
            points, [str(seg) for seg in present], name="region_number_labels",
            font_size=28, text_color="black", shape_color="white", fill_shape=True,
            always_visible=False, shadow=True, pickable=False,
        )

    def _confirm_and_close(self) -> None:
        if self.last_region_labels is None:
            QMessageBox.warning(self, "Not computed", "Press 'Compute regions' first.")
            return
        self.cell_labels = self.last_region_labels
        self.close()

    # -- drawing: landmarks --------------------------------------------------

    def _marker_radius(self) -> float:
        return 0.006 * self.mesh.length

    def _add_marker_sphere(self, actor_name: str, xyz, color: str, *, ref: tuple, pickable: bool) -> None:
        """Draw a landmark/waypoint marker as an actual small 3D sphere
        (not a screen-space point sprite) so it has real geometry a cell
        picker can hit - a batched point-sprite actor with pickable=True
        was previously tried for this and turned out not to be reliably
        clickable. `ref` records what this marker represents, for
        _ref_for_actor to resolve a pick back to it in edit mode."""
        sphere = pv.Sphere(radius=self._marker_radius(), center=xyz, theta_resolution=12, phi_resolution=12)
        self.plotter.add_mesh(
            sphere, color=color, name=actor_name, pickable=pickable,
            smooth_shading=True, specular=0.0,
        )
        self._marker_refs[actor_name] = ref

    def _edit_mode_active(self) -> bool:
        return self.current is None

    def _sync_marker_pickability(self) -> None:
        """Every landmark/waypoint marker is only a valid edit-mode pick
        target once all landmarks are placed - keep their actors' pickable
        flag in sync with that, called after every action that could change
        it (see _refresh_status)."""
        pickable = self._edit_mode_active()
        for actor_name in self._marker_refs:
            actor = self.plotter.renderer.actors.get(actor_name)
            if actor is not None:
                actor.SetPickable(pickable)

    def _draw_point_marker(self, name: str, xyz, color: str) -> None:
        self._add_marker_sphere(f"marker_{name}", xyz, color, ref=("point", name), pickable=self._edit_mode_active())
        self.plotter.add_point_labels(
            [xyz], [name], name=f"label_{name}", font_size=20, text_color=color,
            shape=None, always_visible=False, pickable=False,
        )

    def _clear_marker_actors_with_prefix(self, prefix: str) -> None:
        for actor_name in [k for k in self._marker_refs if k.startswith(prefix)]:
            self.plotter.remove_actor(actor_name)
            self._marker_refs.pop(actor_name, None)

    def _clear_landmark_actors(self, name: str) -> None:
        for prefix in ("marker_", "label_", "loopline_"):
            self.plotter.remove_actor(f"{prefix}{name}")
        self._marker_refs.pop(f"marker_{name}", None)
        self._clear_marker_actors_with_prefix(f"wp_{name}_")

    def _smart_chain_polydata(self, vertex_ids: list[int], close: bool) -> pv.PolyData | None:
        """Build a drawable polyline through consecutive vertex ids, using a
        rim arc wherever two consecutive ones share a detected mesh
        boundary loop and a geodesic otherwise - see geometry.smart_chain.
        This is purely visual, but mirrors exactly what segmentation.py
        will actually compute, so what's drawn matches what gets cut."""
        if len(vertex_ids) < 2:
            return None
        dense_ids = smart_chain(self.mesh, self.boundary, vertex_ids, close=close)
        if len(dense_ids) < 2:
            return None
        return pv.lines_from_points(self.mesh.points[dense_ids], close=close)

    def _loop_vertex_ids(self, name: str) -> list[int]:
        """Full vertex id list for a finished loop landmark, including any
        seed_points prepended (e.g. KN_loop's K, N) - session.loops only
        stores the operator's own clicked waypoints, so this always
        reflects those points' current position rather than a stale
        baked-in copy."""
        spec = self.plan_by_name[name]
        ids = list(self.session.loops[name]["vertex_ids"])
        if spec.seed_points:
            seed_ids = [self.session.points[n]["vertex_id"] for n in spec.seed_points]
            ids = [*seed_ids, *ids]
        return ids

    def _path_vertex_ids(self, name: str) -> list[int]:
        """Full vertex id list for a finished path landmark, with
        path_start/path_end resolved fresh from session.points (mirrors
        _loop_vertex_ids) - session.paths only stores the operator's own
        waypoints, so this always reflects those points' current position."""
        spec = self.plan_by_name[name]
        start = self.session.points[spec.path_start]
        end = self.session.points[spec.path_end]
        mid_ids = self.session.paths[name]["vertex_ids"]
        return [start["vertex_id"], *mid_ids, end["vertex_id"]]

    def _draw_buffer_progress(self, name: str) -> None:
        """Preview the in-progress waypoints of a loop or path landmark.
        For a path, the chain is anchored to path_start; for a seeded loop
        (e.g. KN_loop), to its seed_points - so the chain is visibly
        connected from the start as soon as the first new waypoint is
        placed."""
        spec = self.plan_by_name[name]
        ids = [v for v, _ in self._loop_buffer]
        pts = [xyz for _, xyz in self._loop_buffer]
        self.plotter.add_points(
            pv.PolyData(pts), color=LOOP_COLOR, point_size=12, render_points_as_spheres=True,
            name=f"marker_{name}", pickable=False,
        )
        chain_ids = ids
        if spec.kind == "path":
            start = self.session.points.get(spec.path_start)
            if start is not None:
                chain_ids = [start["vertex_id"], *ids]
        elif spec.seed_points and all(n in self.session.points for n in spec.seed_points):
            seed_ids = [self.session.points[n]["vertex_id"] for n in spec.seed_points]
            chain_ids = [*seed_ids, *ids]
        chain = self._smart_chain_polydata(chain_ids, close=False)
        if chain is not None:
            self.plotter.add_mesh(chain, color=LOOP_COLOR, line_width=3, name=f"loopline_{name}", pickable=False)

    def _draw_loop_final(self, name: str) -> None:
        self.plotter.remove_actor(f"marker_{name}")  # drop the in-progress buffer marker, if any
        loop = self.session.loops[name]
        full_ids = self._loop_vertex_ids(name)
        pts = loop["xyz"]  # markers only for the operator's own waypoints - seed points already have their own
        chain = self._smart_chain_polydata(full_ids, close=True)
        if chain is not None:
            self.plotter.add_mesh(chain, color=DONE_COLOR, line_width=4, name=f"loopline_{name}", pickable=False)
        pickable = self._edit_mode_active()
        for idx, xyz in enumerate(pts):
            self._add_marker_sphere(f"wp_{name}_{idx}", xyz, DONE_COLOR, ref=("loop", name, idx), pickable=pickable)
        full_xyz = self.mesh.points[full_ids]
        centroid = full_xyz.mean(axis=0).tolist()
        self.plotter.add_point_labels(
            [centroid], [name], name=f"label_{name}", font_size=20, text_color=DONE_COLOR,
            shape=None, always_visible=False, pickable=False,
        )

    def _draw_path_final(self, name: str) -> None:
        self.plotter.remove_actor(f"marker_{name}")  # drop the in-progress buffer marker, if any
        data = self.session.paths[name]
        ids = self._path_vertex_ids(name)
        pts = data["xyz"]  # operator's own waypoints only - path_start/path_end already have their own markers
        chain = self._smart_chain_polydata(ids, close=False)
        if chain is not None:
            self.plotter.add_mesh(chain, color=DONE_COLOR, line_width=4, name=f"loopline_{name}", pickable=False)
        pickable = self._edit_mode_active()
        for idx, xyz in enumerate(pts):
            self._add_marker_sphere(f"wp_{name}_{idx}", xyz, DONE_COLOR, ref=("path", name, idx), pickable=pickable)

    def _clear_loop_actors(self, name: str) -> None:
        self.plotter.remove_actor(f"marker_{name}")
        self.plotter.remove_actor(f"loopline_{name}")

    def _redraw_existing_landmarks(self) -> None:
        for name, data in self.session.points.items():
            self._draw_point_marker(name, data["xyz"], DONE_COLOR)
        for name in self.session.loops:
            self._draw_loop_final(name)
        for name in self.session.paths:
            self._draw_path_final(name)

    # -- drawing: segment-boundary overlays -----------------------------------

    def _draw_geodesic(self, actor_name: str, id1: int, id2: int) -> None:
        if id1 == id2:
            self.plotter.remove_actor(actor_name)
            return
        path = self.mesh.geodesic(id1, id2)
        self.plotter.add_mesh(path, color=GEODESIC_COLOR, line_width=4, name=actor_name, pickable=False)

    def _draw_point_chain(self, actor_name: str, vertex_ids: list[int], close: bool = False) -> None:
        """Draw a polyline through exact mesh points (no shortest-path search) -
        used for arcs that must follow an existing ordered loop, e.g. a rim."""
        if len(vertex_ids) < 2:
            self.plotter.remove_actor(actor_name)
            return
        line = pv.lines_from_points(self.mesh.points[vertex_ids], close=close)
        self.plotter.add_mesh(line, color=GEODESIC_COLOR, line_width=4, name=actor_name, pickable=False)

    def _update_geodesics(self) -> None:
        for a, b in self.edges:
            actor_name = f"edge_{a}_{b}"
            pa, pb = self.session.points.get(a), self.session.points.get(b)
            if pa and pb:
                self._draw_geodesic(actor_name, pa["vertex_id"], pb["vertex_id"])
            else:
                self.plotter.remove_actor(actor_name)

        # Seeded loops (e.g. KN_loop) should behave exactly like a plain
        # edge (A_B, ...) for their own seed points: connected as soon as
        # both are placed, same as any other pair of already-placed
        # points, not only once the operator starts adding the loop's own
        # waypoints. Skipped once the operator's own waypoint buffer for
        # *this* loop has started (its chain already covers the seed link
        # itself - see _draw_buffer_progress) or the loop is finished
        # (_draw_loop_final's closed chain covers it), to avoid drawing
        # the same segment twice.
        for spec in self.plan:
            if spec.kind != "loop" or len(spec.seed_points) < 2:
                continue
            finished = self.session.has(spec.name)
            mid_buffer = self.current is spec and bool(self._loop_buffer)
            for a, b in zip(spec.seed_points, spec.seed_points[1:]):
                actor_name = f"seedlink_{spec.name}_{a}_{b}"
                pa, pb = self.session.points.get(a), self.session.points.get(b)
                if finished or mid_buffer or not (pa and pb):
                    self.plotter.remove_actor(actor_name)
                    continue
                chain = self._smart_chain_polydata([pa["vertex_id"], pb["vertex_id"]], close=False)
                if chain is not None:
                    self.plotter.add_mesh(chain, color=GEODESIC_COLOR, line_width=4, name=actor_name, pickable=False)
                else:
                    self.plotter.remove_actor(actor_name)

        for arc_edge in self.boundary_arc_edges:
            a, b = arc_edge.a, arc_edge.b
            actor_name = f"arc_{arc_edge.curve_name}"
            pa, pb = self.session.points.get(a), self.session.points.get(b)
            if not (pa and pb):
                self.plotter.remove_actor(actor_name)
                continue
            loop_a = self.boundary.loop_containing(pa["vertex_id"])
            loop_b = self.boundary.loop_containing(pb["vertex_id"])
            if loop_a is not None and loop_a == loop_b:
                arc_ids = arc_between(loop_a, pa["vertex_id"], pb["vertex_id"])
                self._draw_point_chain(actor_name, arc_ids)
            else:
                # Not (both) on a detected boundary rim - falls back to a
                # plain geodesic instead of a rim arc, same as
                # segmentation.py; drawn in the same colour as any other
                # cutting curve so it's visually clear it's not a rim arc.
                self._draw_geodesic(actor_name, pa["vertex_id"], pb["vertex_id"])

        for a, loop_name, b in self.loop_paths:
            name_a = f"looppath_{a}_{loop_name}"
            name_arc = f"looppath_{loop_name}_arc"
            name_b = f"looppath_{loop_name}_{b}"
            pa, pb = self.session.points.get(a), self.session.points.get(b)
            loop = self.session.loops.get(loop_name)
            if pa and pb and loop:
                full_ids = self._get_full_loop(loop_name, loop["vertex_ids"])
                full_xyz = self.mesh.points[full_ids]
                x_a = nearest_on_point_list(full_ids, full_xyz, pa["xyz"])
                x_b = nearest_on_point_list(full_ids, full_xyz, pb["xyz"])
                self._draw_geodesic(name_a, pa["vertex_id"], x_a)
                arc_ids = arc_between(full_ids, x_a, x_b)
                self._draw_point_chain(name_arc, arc_ids)
                self._draw_geodesic(name_b, x_b, pb["vertex_id"])
            else:
                self.plotter.remove_actor(name_a)
                self.plotter.remove_actor(name_arc)
                self.plotter.remove_actor(name_b)

    def _get_full_loop(self, loop_name: str, clicked_ids: list[int]) -> list[int]:
        """Cache the dense boundary-curve vertex ids for a clicked loop,
        recomputing only when its clicked points have changed."""
        key = tuple(clicked_ids)
        cached = self._full_loop_cache.get(loop_name)
        if cached is not None and cached[0] == key:
            return cached[1]
        full_ids = smart_chain(self.mesh, self.boundary, clicked_ids, close=True)
        self._full_loop_cache[loop_name] = (key, full_ids)
        return full_ids

    # -- status panel ---------------------------------------------------------

    def _refresh_status(self) -> None:
        current = self.current
        if current is None:
            headline = "ALL LANDMARKS PLACED"
        elif current.kind == "point":
            headline = f"Right-click: {current.name}"
        elif current.kind == "path":
            headline = f"Click waypoints: {current.name}  ({current.path_start} -> {current.path_end})"
        elif current.seed_points:
            headline = f"Click loop points: {current.name}  (continues from {', '.join(current.seed_points)})"
        else:  # loop
            headline = f"Click loop points: {current.name}"
        self.headline_label.setText(headline)

        details: list[str] = []
        if current is not None:
            details.extend(textwrap.wrap(current.description, width=48))
            if current.view:
                details.append(f"Suggested view: {current.view}")
            details.append("(shift+right-click to snap to the nearest valve/vein rim)")
            if current.kind == "path":
                n = len(self._loop_buffer)
                details.append(f"{n} waypoint(s) placed (need >= 1) - press 'Finish loop/path'")
            elif current.kind == "loop":
                n = len(self._loop_buffer)
                n_needed = max(0, 3 - len(current.seed_points))
                details.append(f"{n} point(s) placed (need >= {n_needed}) - press 'Finish loop/path'")
        else:
            details.append("Press 'Compute regions', then 'Confirm & Close'.")
            if self._edit_selected is None:
                details.append(
                    "Right-click a landmark point or waypoint to select it for moving."
                )
            else:
                details.append(
                    f"Selected {self._edit_label(self._edit_selected)} (gold) - right-click "
                    "a new location to move it there (shift+right-click to snap to a rim)."
                )
        self.details_label.setText("\n".join(details))

        lines = [f"{self.chamber} landmarks:"]
        for spec in self.plan:
            mark = "x" if self.session.has(spec.name) else " "
            arrow = "->" if current is not None and spec.name == current.name else "  "
            lines.append(f"{arrow} [{mark}] {spec.name}")
        self.checklist_label.setText("\n".join(lines))

        self._update_hint_image()
        self._sync_marker_pickability()

    def _update_hint_image(self) -> None:
        """Show a reference image in the right-hand panel, tracking the
        current landmark automatically: a figure crop from the paper while
        landmarks remain to place, or the chamber's full 15-segment
        reference figure once every landmark is placed."""
        current = self.current
        path = image_path_for(current.name) if current is not None else final_image_path_for(self.chamber)
        path_str = str(path) if path is not None else None
        if path_str == self._current_hint_image:
            return
        self._current_hint_image = path_str
        if path_str is None:
            self._current_hint_pixmap = None
            self.hint_image_label.clear()
            return
        self._current_hint_pixmap = QPixmap(path_str)
        self._rescale_hint_pixmap()
