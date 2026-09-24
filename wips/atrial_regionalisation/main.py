"""WIP: interactively place the 15 EHRA/EACVI bi-atrial model landmarks on
an atrial surface mesh and compute the resulting cell-based region
labelling (Althoff et al., Europace 2025;27:euaf134).

The computed labels are stored as `cell_region`, OpenEP's own field name
for a per-cell integer array - so the output case works directly with any
other WIP or tool that already reads that field.
"""

import sys

from PyQt5.QtWidgets import QApplication

from ar_utils import case_io
from ar_utils.picker_widget import LandmarkPickerWidget


# Boilerplate to run in debug mode or in EP Workbench WIP environment
try:
    case = cases[case_1]
    debug = False
except NameError:
    import pyvista as pv

    debug = True
    root_dir = "/Users/s1807328/Desktop/"
    mesh_path = f"{root_dir}/test_LA.vtk"
    case_1 = "test_LA"
    chamber = "LA"


def main():

    app = QApplication(sys.argv)

    if debug:
        mesh = pv.read(mesh_path)
    else:
        mesh = case_io.mesh_from_case(case)

    window = LandmarkPickerWidget(mesh, chamber)
    window.show()
    exit_code = app.exec_()

    if window.cell_labels is None:
        print("Window closed without confirming - no output case created.")
        sys.exit(exit_code)

    window.mesh.cell_data["cell_region"] = window.cell_labels
    root_name = case_1.rsplit("__", 1)[0] if "__" in case_1 else case_1
    new_name = f"{root_name}__regions"

    if debug:
        window.mesh.save(f"{root_dir}/{new_name}.vtk")
        print(f"Saved labelled mesh to {root_dir}/{new_name}.vtk")
    else:
        new_case = case_io.case_from_mesh(window.mesh, name=new_name)
        out_cases[new_name] = new_case

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
