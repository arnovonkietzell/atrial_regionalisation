"""Loads the operator-facing reference images pulled from the paper's own
figures. Each one covers several landmarks at once (see IMAGE_MAP), matching
how the paper itself groups points by the view they're placed in. Once every
landmark for a chamber is placed, FINAL_IMAGE shows that chamber's full
15-segment reference figure instead, for visually comparing the computed
regions against the paper.
"""

from __future__ import annotations

from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"

# landmark name -> image filename under images/. Several landmarks share
# one image, same as the paper's own figures group them by view.
IMAGE_MAP: dict[str, str] = {
    "A": "ABCD.png", "B": "ABCD.png", "C": "ABCD.png", "D": "ABCD.png",
    "LPV_antrum_outer": "ABCD.png", "RPV_antrum_outer": "ABCD.png",
    "E": "EFHI_laa.png", "F": "EFHI_laa.png", "H": "EFHI_laa.png", "I": "EFHI_laa.png",
    "LAA_neck": "EFHI_laa.png",
    "J": "J.png",
    "K": "KNLO.png", "N": "KNLO.png", "L": "KNLO.png", "O": "KNLO.png",
    "U": "UMTVPQS.png", "M": "UMTVPQS.png", "T": "UMTVPQS.png", "V": "UMTVPQS.png",
    "P": "UMTVPQS.png", "Q": "UMTVPQS.png", "S": "UMTVPQS.png",
}

# chamber -> filename of the full 15-segment reference figure, shown once
# every landmark for that chamber has been placed.
FINAL_IMAGE: dict[str, str] = {
    "LA": "final_LA.png",
    "RA": "final_RA.png",
}


def image_path_for(name: str) -> Path | None:
    """Return the hint-image path for a landmark, or None if it has none."""
    fname = IMAGE_MAP.get(name)
    if fname is None:
        return None
    path = IMAGES_DIR / fname
    return path if path.exists() else None


def final_image_path_for(chamber: str) -> Path | None:
    """Return the full-reference image path for a chamber once placement is
    complete, or None if there isn't one."""
    fname = FINAL_IMAGE.get(chamber.upper())
    if fname is None:
        return None
    path = IMAGES_DIR / fname
    return path if path.exists() else None
