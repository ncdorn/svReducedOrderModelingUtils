"""
Synthetic 3D results and centerlines used to test svromutils.slicer.

The geometry is a U-shaped tube: two parallel straight legs joined by a half-circle
bend. Planes normal to the centerline in the legs and in most of the bend also cut
the other leg, so the slicer must pick the part of the slice closest to the
centerline point.

The velocity is a Poiseuille profile along the centerline tangent and the pressure
decreases linearly with the centerline arc length, so the expected flow and the
expected area averaged pressure are known.
"""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk as n2v

TUBE_RADIUS = 1.0
BEND_RADIUS = 3.0
LEG_LENGTH = 10.0
LENGTH = 2.0 * LEG_LENGTH + np.pi * BEND_RADIUS

# Time steps written as pressure_<step> and velocity_<step> arrays.
TIME_STEPS = [0, 10, 20]

# Maximum velocity and pressure scale for each time step.
MAX_VELOCITY = {0: 1.0, 10: 2.5, 20: -0.5}
PRESSURE_INLET = {0: 100.0, 10: 120.0, 20: 90.0}
PRESSURE_GRADIENT = {0: 1.0, 10: 2.0, 20: -0.5}


def curve(s):
    """ Get the centerline position and tangent at arc lengths s.
    """
    s = np.asarray(s, dtype=float)
    pos = np.zeros(s.shape + (3,))
    tan = np.zeros(s.shape + (3,))

    leg1 = s <= LEG_LENGTH
    pos[leg1] = np.stack([0 * s[leg1], 0 * s[leg1], s[leg1]], axis=-1)
    tan[leg1] = [0.0, 0.0, 1.0]

    bend = (s > LEG_LENGTH) & (s < LEG_LENGTH + np.pi * BEND_RADIUS)
    theta = (s[bend] - LEG_LENGTH) / BEND_RADIUS
    pos[bend] = np.stack([BEND_RADIUS * (1.0 - np.cos(theta)), 0 * theta,
                          LEG_LENGTH + BEND_RADIUS * np.sin(theta)], axis=-1)
    tan[bend] = np.stack([np.sin(theta), 0 * theta, np.cos(theta)], axis=-1)

    leg2 = s >= LEG_LENGTH + np.pi * BEND_RADIUS
    z = LEG_LENGTH - (s[leg2] - LEG_LENGTH - np.pi * BEND_RADIUS)
    pos[leg2] = np.stack([2.0 * BEND_RADIUS + 0 * z, 0 * z, z], axis=-1)
    tan[leg2] = [0.0, 0.0, -1.0]
    return pos, tan


def expected_flow(step):
    return MAX_VELOCITY[step] * np.pi * TUBE_RADIUS**2 / 2.0


def expected_pressure(step, s):
    return PRESSURE_INLET[step] - PRESSURE_GRADIENT[step] * np.asarray(s)


def make_results(num_cross=10, ds=0.25):
    """ Create the tetrahedral mesh and 3D results of the U-shaped tube.

    Args:
        num_cross (int): Number of cells across the tube.
        ds (float): Approximate cell length along the tube.
    """
    # Map a square grid onto the disk, so there is no singular point at the center.
    uv = np.linspace(-1.0, 1.0, num_cross + 1)
    u, v = np.meshgrid(uv, uv, indexing="ij")
    x = TUBE_RADIUS * u * np.sqrt(1.0 - v**2 / 2.0)
    y = TUBE_RADIUS * v * np.sqrt(1.0 - u**2 / 2.0)

    num_s = int(round(LENGTH / ds))
    s = np.linspace(0.0, LENGTH, num_s + 1)
    pos, tan = curve(s)
    binormal = np.array([0.0, 1.0, 0.0])
    normal = np.cross(binormal, tan)

    # Structured grid point order: x index fastest, then y, then s.
    points = (pos[:, None, None, :] + x.T[None, :, :, None] * normal[:, None, None, :] +
              y.T[None, :, :, None] * binormal[None, None, None, :])
    points = points.reshape(-1, 3)

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(num_cross + 1, num_cross + 1, num_s + 1)
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(n2v(points, deep=True))
    grid.SetPoints(vtk_points)

    rho2 = np.broadcast_to((x.T**2 + y.T**2)[None, :, :] / TUBE_RADIUS**2,
                           (num_s + 1, num_cross + 1, num_cross + 1)).reshape(-1)
    point_s = np.broadcast_to(s[:, None, None], (num_s + 1, num_cross + 1, num_cross + 1)).reshape(-1)
    point_tan = np.repeat(tan, (num_cross + 1)**2, axis=0)

    point_data = grid.GetPointData()
    for step in TIME_STEPS:
        pressure = n2v(expected_pressure(step, point_s), deep=True)
        pressure.SetName("pressure_%05d" % step)
        point_data.AddArray(pressure)
        velocity = MAX_VELOCITY[step] * (1.0 - rho2)[:, None] * point_tan
        velocity = n2v(velocity, deep=True)
        velocity.SetName("velocity_%05d" % step)
        point_data.AddArray(velocity)

    # Arrays that are not mapped to the centerline.
    wss = n2v(np.ones(len(points)), deep=True)
    wss.SetName("WSS_00000")
    point_data.AddArray(wss)

    tetra_filter = vtk.vtkDataSetTriangleFilter()
    tetra_filter.SetInputData(grid)
    tetra_filter.Update()
    results = tetra_filter.GetOutput()

    cell_pressure = n2v(np.zeros(results.GetNumberOfCells()), deep=True)
    cell_pressure.SetName("pressure_cell")
    results.GetCellData().AddArray(cell_pressure)
    return results


def make_centerlines(s):
    """ Create a single centerline with points at arc lengths s.
    """
    s = np.asarray(s, dtype=float)
    pos, tan = curve(s)

    centerlines = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetData(n2v(pos, deep=True))
    centerlines.SetPoints(points)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(len(s))
    for i in range(len(s)):
        lines.InsertCellPoint(i)
    centerlines.SetLines(lines)

    point_arrays = {
        "CenterlineSectionNormal": tan,
        "MaximumInscribedSphereRadius": np.full(len(s), TUBE_RADIUS),
        "CenterlineSectionArea": np.full(len(s), np.pi * TUBE_RADIUS**2),
        "BranchId": np.zeros(len(s), dtype=np.int32),
        "CenterlineId": np.zeros(len(s), dtype=np.int32),
        "GlobalNodeId": np.arange(len(s), dtype=np.int32),
        "Path": s,
        "Unused": np.zeros(len(s)),
    }
    for name, values in point_arrays.items():
        array = n2v(values, deep=True)
        array.SetName(name)
        centerlines.GetPointData().AddArray(array)

    cell_array = n2v(np.zeros(1), deep=True)
    cell_array.SetName("CellUnused")
    centerlines.GetCellData().AddArray(cell_array)
    return centerlines


def centerline_arc_lengths(spacing=0.3):
    """ Get the arc lengths of the test centerline points.

    The points are offset from the mesh layers so slice planes do not pass through
    mesh points, and the first and last points are inside the mesh.
    """
    return np.arange(0.07, LENGTH - 0.05, spacing)


def write(data, file_name):
    if isinstance(data, vtk.vtkUnstructuredGrid):
        writer = vtk.vtkXMLUnstructuredGridWriter()
    else:
        writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(file_name))
    writer.SetInputData(data)
    writer.Write()
