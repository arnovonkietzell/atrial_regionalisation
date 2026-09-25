"""WIP: interactively place the 15 EHRA/EACVI bi-atrial model landmarks on
an atrial surface mesh and compute the resulting cell-based region
labelling (Althoff et al., Europace 2025;27:euaf134).

The computed labels are stored as `cell_region`, OpenEP's own field name
for a per-cell integer array - so the output case works directly with any
other WIP or tool that already reads that field.

Debug mode (run directly, outside EP Workbench - see the repository
README) takes the input mesh path and chamber as command-line arguments:

    python -m wips.atrial_regionalisation.main <mesh_path> <LA|RA>

`root_dir` and the case name are inferred from `mesh_path` (its parent
directory and stem, respectively). Debug mode also saves/reloads the
placed landmarks as a JSON file in `root_dir` (named
`<case_name>_<chamber>_landmarks.json`), purely to aid debugging - so a
debug run doesn't require re-clicking every landmark from scratch. This
never happens in the real EP Workbench WIP flow.
"""

import json
import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication

from ar_utils import case_io
from ar_utils.picker_widget import LandmarkPickerWidget
from ar_utils.session import LandmarkState

# Boilerplate to run in debug mode or in EP Workbench WIP environment
try:
    case = cases[case_1]
    debug = False
except NameError:
    debug = True


def _parse_debug_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Debug-mode runner: place landmarks and compute regions "
                    "on a local mesh file, outside EP Workbench."
    )
    parser.add_argument("mesh_path", type=Path, help="Path to the input atrial surface mesh (.vtk)")
    parser.add_argument("chamber", choices=["LA", "RA"], help="Which chamber's landmark plan to use")
    return parser.parse_args()


def _debug_session_path(root_dir: Path, case_name: str, chamber_name: str) -> Path:
    return root_dir / f"{case_name}_{chamber_name}_landmarks.json"


def main():

    app = QApplication(sys.argv)

    # `case_name`/`chamber_name` (not `case_1`/`chamber`) deliberately, so
    # that assigning them here in the debug branch can't shadow the real
    # `case_1`/`chamber` globals the EP Workbench harness injects for the
    # non-debug branch - Python treats a name assigned anywhere in a
    # function as local to the *whole* function, so reusing those exact
    # names here would silently break the non-debug path.
    if debug:
        import pyvista as pv

        args = _parse_debug_args()
        mesh_path = args.mesh_path
        chamber_name = args.chamber
        root_dir = mesh_path.parent
        case_name = mesh_path.stem
        mesh = pv.read(str(mesh_path))
    else:
        chamber_name = chamber
        case_name = case_1
        mesh = case_io.mesh_from_case(case)

    window = LandmarkPickerWidget(mesh, chamber_name)

    if debug:
        session_path = _debug_session_path(root_dir, case_name, chamber_name)
        if session_path.exists():
            saved = LandmarkState.from_dict(json.loads(session_path.read_text()))
            if saved.chamber == chamber_name:
                window.load_session(saved)
                print(f"Debug: reloaded landmark session from {session_path}")
            else:
                print(f"Debug: ignoring {session_path} (saved for chamber "
                      f"'{saved.chamber}', not '{chamber_name}')")

    window.show()
    exit_code = app.exec_()

    if debug:
        session_path = _debug_session_path(root_dir, case_name, chamber_name)
        session_path.write_text(json.dumps(window.session.to_dict(), indent=2))
        print(f"Debug: saved landmark session to {session_path}")

    if window.cell_labels is None:
        print("Window closed without confirming - no output case created.")
        sys.exit(exit_code)

    window.mesh.cell_data["cell_region"] = window.cell_labels
    root_name = case_name.rsplit("__", 1)[0] if "__" in case_name else case_name
    new_name = f"{root_name}__regions"

    if debug:
        out_path = root_dir / f"{new_name}.vtk"
        window.mesh.save(str(out_path))
        print(f"Saved labelled mesh to {out_path}")
    else:
        new_case = case_io.case_from_mesh(window.mesh, name=new_name)
        out_cases[new_name] = new_case

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
