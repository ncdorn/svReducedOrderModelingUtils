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
