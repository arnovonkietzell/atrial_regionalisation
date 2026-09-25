"""Landmark definitions for the EHRA/EACVI 15-segment bi-atrial model
(Althoff et al., Europace 2025;27:euaf134).

Each chamber has an ordered list of named-letter point landmarks from the
paper. LA additionally needs a closed-loop landmark for the LAA neck (the
appendage junction is described only as a curvature discontinuity, not by
named points, so it has to be manually traced) - the RAA has no equivalent
since segment 11 closes fully from named points alone. Both chambers use
"path" landmarks where a segment boundary needs a second, independent route
between two points already connected by a direct geodesic (LA's antrum
necks). RA's IVC orifice is instead a "loop" landmark seeded from its two
already-placed boundary points (K, N) plus 1+ extra waypoints closing back
to K - see `seed_points` below and geodesics.py.

Landmarks are ordered to group consecutive points that share the same
recommended viewing projection (per the paper's own placement instructions),
so the camera can stay put for a whole group instead of being rotated
between every click:
  LA: A,B,C,D (postero-anterior) -> antrum-outer paths -> E,F,H,I
      (apico-basal / left anterior oblique) -> LAA_neck.
  RA: U,M,V,T (apico-basal / left anterior oblique, tricuspid annulus) ->
      J,K,N,KN_loop,L,O (apico-basal / basal-to-apical, venous orifices) ->
      P,Q,S (right lateral, RAA silhouette).

Each landmark's `description` is a concise, paper-derived explanation of
what/where it is, always shown while placing it. A reference image (a
figure crop from the paper) is shown automatically alongside - see
hints.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class LandmarkSpec:
    name: str
    # "point": anywhere on the open surface. Right-click places it raw;
    #   shift+right-click snaps to the nearest open mesh boundary (a
    #   valve/vein orifice rim), if the mesh has one nearby. Several of
    #   these are *expected* to sit on a rim (e.g. the mitral/tricuspid
    #   annulus points) but aren't required to - whether they actually do
    #   depends on how the mesh was clipped, which segmentation checks for
    #   itself rather than assuming from this "kind" - see geodesics.py
    #   and segmentation.py.
    # "loop": a closed ring of clicked points (appendage neck, or - via
    #   `seed_points` - a venous orifice rim seeded from already-placed
    #   points). Each point can individually be raw or shift-snapped, same
    #   as "point".
    # "path": an open chain of 1+ clicked waypoints between two already-
    #   placed point landmarks (path_start -> waypoints -> path_end),
    #   used where a segment boundary needs a second, independent route
    #   between two points already connected by a direct geodesic.
    kind: Literal["point", "loop", "path"]
    description: str
    path_start: str | None = None
    path_end: str | None = None
    # "loop" only: names of already-placed point landmarks to prepend, in
    # order, before the operator's own clicked loop points - e.g. KN_loop
    # starts from K and N (both already placed earlier) so the operator
    # only has to click the remaining waypoint(s) needed to close the ring.
    seed_points: tuple[str, ...] = ()
    # Recommended viewing projection for placing this landmark, per the
    # paper's own text - None where the paper doesn't specify one. Always
    # shown alongside the description.
    view: str | None = None


LA_POINTS: list[LandmarkSpec] = [
    LandmarkSpec(
        "A", "point",
        "Left pulmonary venous antrum, superior point where the curvature "
        "changes from antrum to atrial dome.",
        view="Postero-anterior",
    ),
    LandmarkSpec(
        "B", "point",
        "Left pulmonary venous antrum, inferior point where the curvature "
        "changes from antrum to atrial dome.",
        view="Postero-anterior",
    ),
    LandmarkSpec(
        "LPV_antrum_outer", "path",
        "Outer (vein-facing) border of the left antrum neck, tracing from B "
        "to A on the side away from the dome.",
        path_start="B", path_end="A", view="Postero-anterior",
    ),
    LandmarkSpec(
        "C", "point",
        "Right pulmonary venous antrum, superior point where the curvature "
        "changes from antrum to atrial dome.",
        view="Postero-anterior",
    ),
    LandmarkSpec(
        "D", "point",
        "Right pulmonary venous antrum, inferior point where the curvature "
        "changes from antrum to atrial dome.",
        view="Postero-anterior",
    ),
    LandmarkSpec(
        "RPV_antrum_outer", "path",
        "Outer (vein-facing) border of the right antrum neck, tracing from D "
        "to C on the side away from the dome.",
        path_start="D", path_end="C", view="Postero-anterior",
    ),
    LandmarkSpec("E", "point", "Mitral annulus at 9 o'clock (septal) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec("F", "point", "Mitral annulus at 1 o'clock (lateral) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec("H", "point", "Mitral annulus at 4 o'clock (infero-lateral) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec("I", "point", "Mitral annulus at 7 o'clock (infero-septal) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec(
        "LAA_neck", "loop",
        "Closed ring around the neck of the left atrial appendage, at the "
        "sharp curvature change where the appendage angulates from the "
        "atrial body.",
    ),
]

RA_POINTS: list[LandmarkSpec] = [
    LandmarkSpec("U", "point", "Tricuspid annulus at 11 o'clock (supero-lateral) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec("M", "point", "Tricuspid annulus at 1 o'clock (septal-superior) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec("V", "point", "Tricuspid annulus at 5 o'clock (infero-septal) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec("T", "point", "Tricuspid annulus at 8 o'clock (infero-lateral) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec("J", "point", "Most inferior point of the coronary sinus orifice."),
    LandmarkSpec("K", "point", "Most septal point of the inferior caval vein (IVC) orifice.",
                 view="Apico-basal / basal-to-apical"),
    LandmarkSpec("N", "point", "Most lateral point of the inferior caval vein (IVC) orifice.",
                 view="Apico-basal / basal-to-apical"),
    LandmarkSpec(
        "KN_loop", "loop",
        "Closed boundary of the IVC orifice: continues on from K and N "
        "(already placed) - click 1 or more further waypoints tracing "
        "around the rest of the orifice, then finish to close the loop "
        "back to K.",
        seed_points=("K", "N"), view="Apico-basal / basal-to-apical",
    ),
    LandmarkSpec("L", "point", "Most septal point of the superior caval vein (SVC) orifice.",
                 view="Apico-basal / basal-to-apical"),
    LandmarkSpec("O", "point", "Most lateral point of the superior caval vein (SVC) orifice.",
                 view="Apico-basal / basal-to-apical"),
    LandmarkSpec(
        "P", "point",
        "Anterior junction of the right atrial appendage with the atrial "
        "body, at the point of maximum concavity.",
    ),
    LandmarkSpec(
        "Q", "point",
        "Anterior junction of the protruding appendage with the lateral "
        "wall, at the point of maximum concavity.",
        view="Right lateral",
    ),
    LandmarkSpec(
        "S", "point",
        "Antero-superior cavoatrial junction - the most anterior point of "
        "the SVC orifice.",
        view="Right lateral",
    ),
]

CHAMBER_LANDMARKS: dict[str, list[LandmarkSpec]] = {
    "LA": LA_POINTS,
    "RA": RA_POINTS,
}


def get_landmark_plan(chamber: str) -> list[LandmarkSpec]:
    """Return the ordered landmark plan for 'LA', 'RA', or 'both'."""
    chamber = chamber.upper()
    if chamber == "BOTH":
        return [*LA_POINTS, *RA_POINTS]
    if chamber not in CHAMBER_LANDMARKS:
        raise ValueError(f"Unknown chamber '{chamber}'; expected 'LA', 'RA', or 'both'")
    return CHAMBER_LANDMARKS[chamber]
