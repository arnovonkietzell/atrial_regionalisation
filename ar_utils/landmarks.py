"""Landmark definitions for the EHRA/EACVI 15-segment bi-atrial model
(Althoff et al., Europace 2025;27:euaf134).

Each chamber has an ordered list of named-letter point landmarks from the
paper. LA additionally needs a closed-loop landmark for the LAA neck (the
appendage junction is described only as a curvature discontinuity, not by
named points, so it has to be manually traced) - the RAA has no equivalent
since segment 11 closes fully from named points alone. Both chambers use
"path" landmarks where a segment boundary needs a second, independent route
between two points already connected by a direct geodesic (LA's antrum
necks; RA's IVC uses the complementary rim arc instead - see geodesics.py).

Landmarks are ordered to group consecutive points that share the same
recommended viewing projection (per the paper's own placement instructions),
so the camera can stay put for a whole group instead of being rotated
between every click:
  LA: A,B,C,D (postero-anterior) -> antrum-outer paths -> E,F,H,I
      (apico-basal / left anterior oblique) -> LAA_neck.
  RA: U,M,V,T (apico-basal / left anterior oblique, tricuspid annulus) ->
      J,K,N,L,O (apico-basal / basal-to-apical, venous orifices) ->
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
    # "point": anywhere on the open surface.
    # "boundary": must sit on an open mesh boundary loop (a valve/vein
    #   orifice rim) - snapped automatically to the nearest such vertex.
    # "loop": a closed ring of several clicked points (appendage neck).
    # "path": an open chain of 1+ clicked waypoints between two already-
    #   placed point landmarks (path_start -> waypoints -> path_end),
    #   used where a segment boundary needs a second, independent route
    #   between two points already connected by a direct geodesic.
    kind: Literal["point", "boundary", "loop", "path"]
    description: str
    path_start: str | None = None
    path_end: str | None = None
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
    LandmarkSpec("E", "boundary", "Mitral annulus at 9 o'clock (septal) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec("F", "boundary", "Mitral annulus at 1 o'clock (lateral) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec("H", "boundary", "Mitral annulus at 4 o'clock (infero-lateral) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec("I", "boundary", "Mitral annulus at 7 o'clock (infero-septal) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec(
        "LAA_neck", "loop",
        "Closed ring around the neck of the left atrial appendage, at the "
        "sharp curvature change where the appendage angulates from the "
        "atrial body.",
    ),
]

RA_POINTS: list[LandmarkSpec] = [
    LandmarkSpec("U", "boundary", "Tricuspid annulus at 11 o'clock (supero-lateral) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec("M", "boundary", "Tricuspid annulus at 1 o'clock (septal-superior) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec("V", "boundary", "Tricuspid annulus at 5 o'clock (infero-septal) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec("T", "boundary", "Tricuspid annulus at 8 o'clock (infero-lateral) on the clock model.",
                 view="Apico-basal (~left anterior oblique)"),
    LandmarkSpec("J", "boundary", "Most inferior point of the coronary sinus orifice."),
    LandmarkSpec("K", "boundary", "Most septal point of the inferior caval vein (IVC) orifice.",
                 view="Apico-basal / basal-to-apical"),
    LandmarkSpec("N", "boundary", "Most lateral point of the inferior caval vein (IVC) orifice.",
                 view="Apico-basal / basal-to-apical"),
    LandmarkSpec("L", "boundary", "Most septal point of the superior caval vein (SVC) orifice.",
                 view="Apico-basal / basal-to-apical"),
    LandmarkSpec("O", "boundary", "Most lateral point of the superior caval vein (SVC) orifice.",
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
        "S", "boundary",
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
