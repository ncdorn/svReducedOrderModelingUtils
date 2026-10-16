"""
Tests for svromutils.slicer, including a regression comparison against the C++ svSlicer.

The live comparison runs only if an svslicer executable is found on the PATH or
given by the SVSLICER environment variable.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

import slicer_case
from make_svslicer_reference import run_svslicer
from svromutils.slicer import slice_results, slice_results_files

DATA_DIR = Path(__file__).parent / "data"
RESULT_ARRAYS = ["%s_%05d" % (name, step) for name in ("pressure", "velocity")
                 for step in slicer_case.TIME_STEPS]


def point_arrays(polydata):
    point_data = polydata.GetPointData()
    return {point_data.GetArrayName(i): v2n(point_data.GetArray(i))
            for i in range(point_data.GetNumberOfArrays())}


def read_vtp(file_name):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(file_name))
    reader.Update()
    return reader.GetOutput()


def assert_same_results(actual, expected, rtol=1e-10):
    for name in RESULT_ARRAYS:
        scale = np.abs(expected[name]).max()
        np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=rtol * scale, err_msg=name)


@pytest.fixture(scope="module")
def results():
    return slicer_case.make_results()


@pytest.fixture(scope="module")
def centerlines():
    return slicer_case.make_centerlines(slicer_case.centerline_arc_lengths())


@pytest.fixture(scope="module")
def sliced(results, centerlines):
    return point_arrays(slice_results(results, centerlines))


def test_matches_svslicer_reference(sliced):
    reference = point_arrays(read_vtp(DATA_DIR / "svslicer_u_tube.vtp"))
    assert sorted(sliced) == sorted(reference)
    assert_same_results(sliced, reference)


def test_matches_analytic_solution(sliced):
    s = sliced["Path"]
    for step in slicer_case.TIME_STEPS:
        # The polygonal slice underestimates the area of the circular cross section.
        np.testing.assert_allclose(sliced["velocity_%05d" % step], slicer_case.expected_flow(step), rtol=0.05)
        np.testing.assert_allclose(sliced["pressure_%05d" % step], slicer_case.expected_pressure(step, s),
                                   rtol=1e-5)


def test_output_arrays(results, centerlines, sliced):
    output = slice_results(results, centerlines)
    assert "Unused" not in sliced
    assert "WSS_00000" not in sliced
    assert "pressure_cell" not in sliced
    assert output.GetCellData().GetNumberOfArrays() == 0
    assert output.GetNumberOfPoints() == centerlines.GetNumberOfPoints()
    for name in ["CenterlineSectionNormal", "MaximumInscribedSphereRadius", "BranchId", "Path"] + RESULT_ARRAYS:
        assert name in sliced


def test_empty_slice_uses_neighbor(results):
    # The first point is just outside of the inlet, so its plane does not cut the mesh.
    s = np.array([-0.06, 0.19, 0.44])
    output = point_arrays(slice_results(results, slicer_case.make_centerlines(s)))
    for step in slicer_case.TIME_STEPS:
        flow = output["velocity_%05d" % step]
        np.testing.assert_allclose(flow, slicer_case.expected_flow(step), rtol=0.05)


@pytest.mark.parametrize("split_lines", [False, True])
def test_empty_slice_uses_connected_neighbor(results, split_lines):
    s = np.r_[-0.06, slicer_case.centerline_arc_lengths(), slicer_case.LENGTH + 0.06]
    expected = point_arrays(slice_results(results, slicer_case.make_centerlines(s)))

    # Put the two outside endpoints next to each other in point ID order,
    # while preserving the original centerline connectivity.
    permutation = np.r_[0, len(s) - 1, np.arange(1, len(s) - 1)]
    centerlines = slicer_case.make_centerlines(s[permutation])
    point_ids = np.argsort(permutation)
    cells = list(zip(point_ids[:-1], point_ids[1:])) if split_lines else [point_ids]
    lines = vtk.vtkCellArray()
    for cell in cells:
        lines.InsertNextCell(len(cell))
        for point_id in cell:
            lines.InsertCellPoint(int(point_id))
    centerlines.SetLines(lines)

    actual = point_arrays(slice_results(results, centerlines))
    assert_same_results(actual, {name: values[permutation] for name, values in expected.items()})


def test_empty_slice_without_connected_neighbor_raises(results):
    centerlines = slicer_case.make_centerlines([-0.06, 0.19])
    centerlines.SetLines(vtk.vtkCellArray())
    with pytest.raises(RuntimeError, match="Could not extract a slice at centerline point 0"):
        slice_results(results, centerlines)


def test_empty_slice_skips_coincident_neighbor(results):
    # The first point is outside of the inlet and duplicated, so its closest connected
    # point is at the same position.
    s = np.array([-0.06, -0.06, 0.19, 0.44])
    output = point_arrays(slice_results(results, slicer_case.make_centerlines(s)))
    for step in slicer_case.TIME_STEPS:
        np.testing.assert_allclose(output["velocity_%05d" % step], slicer_case.expected_flow(step), rtol=0.05)


def test_empty_slice_prefers_neighbor_on_same_branch(results):
    s = np.array([-0.06, 0.19, 0.44])
    expected = point_arrays(slice_results(results, slicer_case.make_centerlines(s)))

    # Connect the first point to a closer point on another branch.
    centerlines = slicer_case.make_centerlines(np.r_[s, 0.0])
    centerlines.GetPoints().SetPoint(3, 0.2, 0.0, -0.06)
    centerlines.GetPointData().GetArray("CenterlineSectionNormal").SetTuple3(3, 1.0, 0.0, 0.0)
    centerlines.GetPointData().GetArray("BranchId").SetTuple1(3, 1)
    lines = vtk.vtkCellArray()
    for cell in [[0, 1, 2], [0, 3]]:
        lines.InsertNextCell(len(cell))
        for point_id in cell:
            lines.InsertCellPoint(point_id)
    centerlines.SetLines(lines)

    actual = point_arrays(slice_results(results, centerlines))
    assert_same_results({name: values[:3] for name, values in actual.items()}, expected)


def test_empty_slice_with_flipped_neighbor_normal(results):
    # Only the plane halfway to the neighbor cuts the mesh, where the blended normal
    # would be zero if the neighbor's normal were not flipped back.
    s = np.array([-0.25, 0.35, 0.6])
    expected = point_arrays(slice_results(results, slicer_case.make_centerlines(s)))

    # The neighbor's normal points the other way, which only flips the sign of its own flow.
    centerlines = slicer_case.make_centerlines(s)
    centerlines.GetPointData().GetArray("CenterlineSectionNormal").SetTuple3(1, 0.0, 0.0, -1.0)
    actual = point_arrays(slice_results(results, centerlines))
    sign = np.array([1.0, -1.0, 1.0])
    assert_same_results(actual, {name: values * sign if name.startswith("velocity") else values
                                 for name, values in expected.items()})


def test_zero_normal_raises(results):
    centerlines = slicer_case.make_centerlines([0.19, 0.44])
    centerlines.GetPointData().GetArray("CenterlineSectionNormal").SetTuple3(1, 0.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="zero at centerline point 1"):
        slice_results(results, centerlines)


def test_arrays_with_wrong_components_are_skipped(results, centerlines, sliced):
    mixed = vtk.vtkUnstructuredGrid()
    mixed.DeepCopy(results)
    num_points = mixed.GetNumberOfPoints()
    for name, values in [("velocity_magnitude", np.ones(num_points)), ("pressure_gradient", np.ones((num_points, 3)))]:
        array = n2v(values, deep=True)
        array.SetName(name)
        mixed.GetPointData().AddArray(array)

    actual = point_arrays(slice_results(mixed, centerlines))
    assert sorted(actual) == sorted(sliced)
    assert_same_results(actual, sliced, rtol=0)


def tetra_results(points):
    """ Create results on a single tetrahedron with unit pressure and velocity.
    """
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point)
    results = vtk.vtkUnstructuredGrid()
    results.SetPoints(vtk_points)
    tetra = vtk.vtkTetra()
    for i in range(4):
        tetra.GetPointIds().SetId(i, i)
    results.InsertNextCell(tetra.GetCellType(), tetra.GetPointIds())
    for name, values in [("pressure", np.ones(4)), ("velocity", np.ones((4, 3)))]:
        array = n2v(values, deep=True)
        array.SetName(name)
        results.GetPointData().AddArray(array)
    return results


def test_zero_area_slice_raises():
    # The plane cuts a flat tetrahedron, so the slice has polygons but no area.
    results = tetra_results([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)])
    centerlines = slicer_case.make_centerlines([0.0, 1.0])
    centerlines.GetPoints().SetPoint(0, 0.5, 0.5, 0.0)
    centerlines.GetPoints().SetPoint(1, 0.5, 0.5, 0.5)
    for i in range(2):
        centerlines.GetPointData().GetArray("CenterlineSectionNormal").SetTuple3(i, 1.0, 0.0, 0.0)
    with pytest.raises(RuntimeError, match="Could not extract a slice at centerline point 0"):
        slice_results(results, centerlines)


@pytest.mark.parametrize("normal, endpoint, expected_flow", [
    ([1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.00125, 0.125]),
    ([1.0, 1.0, 0.0], [0.5, 0.5, 0.0], [0.0768, 0.48]),
])
def test_tangent_slice_uses_neighbor(normal, endpoint, expected_flow):
    results = tetra_results([(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)])

    # The endpoint plane touches only a vertex or edge of the tetrahedron.
    # Moving 10% toward the interior point gives a positive-area slice.
    centerlines = slicer_case.make_centerlines([0.0, 1.0])
    centerlines.GetPoints().SetPoint(0, endpoint)
    centerlines.GetPoints().SetPoint(1, 0.5, 0.1, 0.1)
    for i in range(2):
        centerlines.GetPointData().GetArray("CenterlineSectionNormal").SetTuple3(i, *normal)

    actual = point_arrays(slice_results(results, centerlines))
    np.testing.assert_allclose(actual["pressure"], 1.0)
    np.testing.assert_allclose(actual["velocity"], expected_flow, rtol=1e-6)


def test_no_slice_raises(results):
    s = np.array([-5.0, -4.0])
    with pytest.raises(RuntimeError, match="Could not extract a slice"):
        slice_results(results, slicer_case.make_centerlines(s))


def test_parallel_matches_serial(results, centerlines, sliced):
    parallel = point_arrays(slice_results(results, centerlines, num_workers=4))
    assert_same_results(parallel, sliced, rtol=0)


def test_files(tmp_path, results, centerlines, sliced):
    slicer_case.write(results, tmp_path / "results.vtu")
    slicer_case.write(centerlines, tmp_path / "centerlines.vtp")
    slice_results_files(str(tmp_path / "results.vtu"), str(tmp_path / "centerlines.vtp"),
                        str(tmp_path / "output.vtp"))
    assert_same_results(point_arrays(read_vtp(tmp_path / "output.vtp")), sliced, rtol=1e-12)


SVSLICER = os.environ.get("SVSLICER") or shutil.which("svslicer")


@pytest.mark.skipif(SVSLICER is None, reason="svslicer executable not found")
def test_matches_svslicer_live(tmp_path, sliced):
    output_file = tmp_path / "svslicer.vtp"
    run_svslicer(SVSLICER, output_file)
    assert_same_results(sliced, point_arrays(read_vtp(output_file)))
