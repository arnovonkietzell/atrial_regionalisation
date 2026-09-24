"""Conversion between OpenEP `Case` objects (EP Workbench's own data
structure, passed into a WIP via `cases`/`case_1` and returned via
`out_cases`) and the PyVista meshes this library's landmark picker and
segmentation logic actually operate on.

Unlike a plain point-data transfer, both directions here also carry
`cell_region` (and any other per-cell field) - OpenEP's own `Fields` class
already reserves `cell_region` for exactly this purpose (a per-cell
integer array), which is what the computed 15-segment labelling is."""

from __future__ import annotations

import numpy as np
import openep


def mesh_from_case(case):
    """Build a PyVista mesh from an OpenEP Case, transferring every non-None
    field in `case.fields` to `point_data` or `cell_data` by array length.

    Args:
        case: An `openep.data_structures.case.Case`.

    Returns:
        pyvista.PolyData: Mesh with case fields transferred to point_data
        (length == n_points) or cell_data (length == n_cells).
    """
    mesh = case.create_mesh()

    for field_name in case.fields:
        value = case.fields[field_name]
        if value is None:
            continue
        value = np.asarray(value)
        if len(value) == mesh.n_points:
            mesh.point_data[field_name] = value
        elif len(value) == mesh.n_cells:
            mesh.cell_data[field_name] = value

    return mesh


def case_from_mesh(mesh, name: str = "processed_case"):
    """Convert a PyVista mesh back into an OpenEP Case, copying both point
    and cell data fields - including `cell_region`, the computed
    segmentation labels.

    Args:
        mesh: PyVista mesh whose `faces` array follows the packed-integer
            format (`[n_verts, v0, v1, ...]`, all triangles).
        name: Name assigned to the resulting Case.

    Returns:
        openep.data_structures.case.Case: Case with geometry and both
        point and cell fields populated from the mesh.
    """
    case = openep.data_structures.case.Case(
        name=name,
        points=mesh.points,
        indices=mesh.faces.reshape((-1, 4))[:, 1:],  # drop leading count column -> (n_faces, 3)
        fields=openep.data_structures.surface.Fields(),
        electric=openep.data_structures.electric.Electric(),
        ablation=openep.data_structures.ablation.Ablation(),
    )

    for field_name in mesh.point_data:
        case.fields[field_name] = np.asarray(mesh.point_data[field_name])
    for field_name in mesh.cell_data:
        case.fields[field_name] = np.asarray(mesh.cell_data[field_name])

    return case
