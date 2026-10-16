# Copyright (c) Stanford University, The Regents of the University of
#               California, and others.
#
# All Rights Reserved.
#
# See Copyright-SimVascular.txt for additional details.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject
# to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
# IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
# OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
The module maps 3D simulation results onto a centerline (formerly the svSlicer tool).

The 3D results are sliced with a plane at each centerline point, oriented by the
centerline's CenterlineSectionNormal array. Point data arrays whose names start
with "pressure" are averaged over the slice area and arrays whose names start with
"velocity" are integrated over the slice to give the volumetric flow through it.
The results are stored on the centerline under the original array names, so a
"velocity_00100" array on the output centerline holds the flow at that time step.

When a plane cuts the vessel more than once only the part of the slice closest
to the centerline point is used.
"""

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import numpy_to_vtk as n2v

from svromutils.manage import get_logger_name
from svromutils.utils import read_polydata, write_polydata

__all__ = ["slice_results", "slice_results_files"]

logger = logging.getLogger(get_logger_name())

# Prefixes (case-insensitive) of the 3D result point data arrays mapped to the centerline.
PRESSURE_PREFIX = "pressure"
VELOCITY_PREFIX = "velocity"

# Prefixes of the centerline point data arrays copied to the output centerline.
CENTERLINE_KEEP_PREFIXES = ["BifurcationId", "BifurcationIdTmp", "BranchId", "BranchIdTmp",
    "CenterlineId", "CenterlineSectionArea", "CenterlineSectionBifurcation",
    "CenterlineSectionClosed", "CenterlineSectionMaxSize", "CenterlineSectionMinSize",
    "CenterlineSectionNormal", "CenterlineSectionRemove", "CenterlineSectionShape",
    "GlobalNodeId", "MaximumInscribedSphereRadius", "Path"]

NORMAL_ARRAY = "CenterlineSectionNormal"
BRANCH_ARRAY = "BranchId"

# Weights used to move the slice plane toward the neighboring centerline point when
# the plane at the centerline point does not cut the mesh.
NEIGHBOR_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

def read_results(file_name):
    """ Read 3D simulation results from a .vtu file.
    """
    if not os.path.exists(file_name):
        raise RuntimeError("Could not find file: %s" % file_name)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()
    return reader.GetOutput()


def _starts_with(name, prefix):
    return name.lower().startswith(prefix.lower())


def _keep_arrays(data, prefixes):
    """ Remove the data arrays whose names do not start with one of the prefixes.
    """
    names = [data.GetArrayName(i) for i in range(data.GetNumberOfArrays())]
    for name in names:
        if not any(_starts_with(name, prefix) for prefix in prefixes):
            data.RemoveArray(name)


class _Slicer(object):
    """ The _Slicer class extracts and integrates the slices for single centerline points.
    """

    def __init__(self, results, centerlines):
        self.mesh = vtk.vtkUnstructuredGrid()
        self.mesh.ShallowCopy(results)
        _keep_arrays(self.mesh.GetPointData(), [PRESSURE_PREFIX, VELOCITY_PREFIX])
        self.mesh.GetCellData().Initialize()

        # Skip arrays that only share a prefix with the results, e.g. a scalar velocity_magnitude.
        point_data = self.mesh.GetPointData()
        self.array_names = []
        for name in [point_data.GetArrayName(i) for i in range(point_data.GetNumberOfArrays())]:
            num_comp = point_data.GetArray(name).GetNumberOfComponents()
            expected = 3 if _starts_with(name, VELOCITY_PREFIX) else 1
            if num_comp != expected:
                logger.warning("Skipping array '%s' with %d components, expected %d." % (name, num_comp, expected))
                point_data.RemoveArray(name)
            else:
                self.array_names.append(name)

        if centerlines.GetPointData().GetArray(NORMAL_ARRAY) is None:
            raise RuntimeError("The centerlines do not have a '%s' point data array." % NORMAL_ARRAY)
        self.positions = v2n(centerlines.GetPoints().GetData()).astype(float)
        self.normals = v2n(centerlines.GetPointData().GetArray(NORMAL_ARRAY)).astype(float)
        zero_normals = np.flatnonzero(np.linalg.norm(self.normals, axis=1) == 0.0)
        if len(zero_normals) > 0:
            raise RuntimeError("The '%s' array is zero at centerline point %d." % (NORMAL_ARRAY, zero_normals[0]))

        branch_ids = centerlines.GetPointData().GetArray(BRANCH_ARRAY)
        self.branch_ids = None if branch_ids is None else v2n(branch_ids)

        self.neighbors = [set() for _ in self.positions]
        lines = centerlines.GetLines()
        point_ids = vtk.vtkIdList()
        lines.InitTraversal()
        while lines.GetNextCell(point_ids):
            for i in range(1, point_ids.GetNumberOfIds()):
                start = point_ids.GetId(i - 1)
                end = point_ids.GetId(i)
                if start != end:
                    self.neighbors[start].add(end)
                    self.neighbors[end].add(start)

    def _extract_slice(self, position, normal):
        """ Extract the slice at a position, or None if it has no positive area.

        Only the connected part of the slice closest to the position is returned.

        Returns:
            tuple: The triangulated slice, its triangles' point IDs and their areas.
        """
        plane = vtk.vtkPlane()
        plane.SetOrigin(position)
        plane.SetNormal(normal)
        cutter = vtk.vtkCutter()
        cutter.SetCutFunction(plane)
        cutter.SetInputData(self.mesh)
        cutter.Update()
        if cutter.GetOutput().GetNumberOfPoints() == 0:
            return None

        clean_filter = vtk.vtkCleanPolyData()
        clean_filter.SetInputData(cutter.GetOutput())
        clean_filter.Update()

        conn_filter = vtk.vtkPolyDataConnectivityFilter()
        conn_filter.SetInputData(clean_filter.GetOutput())
        conn_filter.SetExtractionModeToClosestPointRegion()
        conn_filter.SetClosestPoint(position)
        conn_filter.Update()
        slice_data = conn_filter.GetOutput()
        polys = slice_data.GetPolys()
        if polys.GetNumberOfCells() == 0:
            return None

        if np.any(np.diff(v2n(polys.GetOffsetsArray())) != 3):
            triangle_filter = vtk.vtkTriangleFilter()
            triangle_filter.SetInputData(slice_data)
            triangle_filter.Update()
            slice_data = triangle_filter.GetOutput()
            polys = slice_data.GetPolys()

        # Degenerate cells, e.g. of a flat mesh element, can give polygons without area.
        triangles = v2n(polys.GetConnectivityArray()).reshape(-1, 3)
        points = v2n(slice_data.GetPoints().GetData()).astype(float)
        p0, p1, p2 = (points[triangles[:, i]] for i in range(3))
        areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
        if not areas.sum() > 0.0:
            return None
        return slice_data, triangles, areas

    def _neighbor(self, index):
        """ Get the index of the closest directly connected centerline point.

        Connected points at the same position are skipped and their own neighbors are
        used instead. If the centerlines have a BranchId array, points on the same branch
        are preferred so the plane is not moved into another branch at a bifurcation.
        """
        position = self.positions[index]
        coincident = {index}
        candidates = set()
        stack = [index]
        while stack:
            for i in self.neighbors[stack.pop()]:
                if i in coincident or i in candidates:
                    continue
                if np.array_equal(self.positions[i], position):
                    coincident.add(i)
                    stack.append(i)
                else:
                    candidates.add(i)
        if not candidates:
            return None

        def key(i):
            other_branch = self.branch_ids is not None and self.branch_ids[i] != self.branch_ids[index]
            return (other_branch, np.sum((self.positions[i] - position)**2), i)
        return min(candidates, key=key)

    def slice_point(self, index):
        """ Compute the integrated quantities for a centerline point.

        Returns:
            dict: Pressure (area average) and flow (integral of velocity . normal)
                  for each result array name.
        """
        position1 = self.positions[index]
        normal1 = self.normals[index]
        neighbor = self._neighbor(index)
        if neighbor is not None:
            # Orient the neighbor's normal like this point's so the blended normal is not zero.
            normal2 = self.normals[neighbor]
            if np.dot(normal1, normal2) < 0.0:
                normal2 = -normal2

        slice_data = None
        for weight in NEIGHBOR_WEIGHTS:
            if weight == 0.0:
                position, normal = position1, normal1
            elif neighbor is None:
                break
            else:
                position = (1.0 - weight) * position1 + weight * self.positions[neighbor]
                normal = (1.0 - weight) * normal1 + weight * normal2
            normal = normal / np.linalg.norm(normal)
            slice_data = self._extract_slice(position, normal)
            if slice_data is not None:
                break
            logger.warning("Slice at centerline point %d was empty (weight %g)." % (index, weight))

        if slice_data is None:
            raise RuntimeError("Could not extract a slice at centerline point %d." % index)

        return self._integrate(*slice_data, normal)

    def _integrate(self, slice_data, triangles, areas, normal):
        """ Integrate the point data arrays of a slice.

        Point values are linearly interpolated over each triangle, so the integral over
        a triangle is its area times the mean of its point values.
        """
        total_area = areas.sum()
        values = {}
        point_data = slice_data.GetPointData()
        for name in self.array_names:
            data = v2n(point_data.GetArray(name)).astype(float)
            if _starts_with(name, VELOCITY_PREFIX):
                data = data @ normal
            cell_mean = data[triangles].mean(axis=1)
            integral = np.dot(areas, cell_mean)
            if _starts_with(name, PRESSURE_PREFIX):
                integral /= total_area
            values[name] = integral
        return values


def _slice_all(slicer, num_workers):
    """ Compute the integrated quantities for all centerline points.

    The worker threads share the mesh; VTK releases the Python GIL while cutting.
    """
    num_points = len(slicer.positions)
    start = time.time()
    if num_workers <= 1 or num_points <= 1:
        values = [slicer.slice_point(index) for index in range(num_points)]
    else:
        # Slice the first point before starting the threads so the mesh data VTK computes
        # on first use (bounds, cell types) is not built by several threads at once.
        values = [slicer.slice_point(0)]
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            values += executor.map(slicer.slice_point, range(1, num_points))
    duration = time.time() - start
    logger.info("Extracted %d slices in %.2f s (%.4f s per slice)" % (num_points, duration,
        duration / max(num_points, 1)))
    return values


def _output_centerlines(centerlines, array_names, values):
    """ Create the output centerlines with the integrated quantities as point data arrays.
    """
    output = vtk.vtkPolyData()
    output.DeepCopy(centerlines)
    _keep_arrays(output.GetPointData(), CENTERLINE_KEEP_PREFIXES)
    output.GetCellData().Initialize()

    for name in array_names:
        array = n2v(np.array([value[name] for value in values], dtype=float), deep=True)
        array.SetName(name)
        output.GetPointData().AddArray(array)
    return output


def slice_results(results, centerlines, num_workers=1):
    """ Map 3D simulation results onto a centerline.

    Args:
        results (vtkUnstructuredGrid): 3D results with pressure* and velocity* point data arrays.
        centerlines (vtkPolyData): Centerlines with a CenterlineSectionNormal point data array.
        num_workers (int): Number of threads used to extract slices.

    Returns:
        vtkPolyData: The centerlines with the area averaged pressure and the flow stored
            under the names of the pressure* and velocity* arrays.
    """
    slicer = _Slicer(results, centerlines)
    values = _slice_all(slicer, num_workers)
    return _output_centerlines(centerlines, slicer.array_names, values)


def slice_results_files(results_file, centerlines_file, output_file, num_workers=1):
    """ Map 3D simulation results from a .vtu file onto the centerlines from a .vtp file.

    Args:
        results_file (str): 3D results file (.vtu).
        centerlines_file (str): Centerlines file (.vtp).
        output_file (str): Output centerlines file (.vtp).
        num_workers (int): Number of threads used to extract slices.
    """
    logger.info("Reading results file '%s'" % results_file)
    results = read_results(results_file)
    logger.info("Reading centerlines file '%s'" % centerlines_file)
    centerlines = read_polydata(centerlines_file)

    slicer = _Slicer(results, centerlines)
    logger.info("Mapping arrays: %s" % ", ".join(slicer.array_names))
    values = _slice_all(slicer, num_workers)
    output = _output_centerlines(centerlines, slicer.array_names, values)

    logger.info("Writing output file '%s'" % output_file)
    write_polydata(output_file, output)


def run_cli():
    parser = argparse.ArgumentParser(description="Map 3D simulation results onto a centerline.")
    parser.add_argument("results_file", help="3D results file (.vtu) with pressure* and velocity* arrays.")
    parser.add_argument("centerlines_file", help="Centerlines file (.vtp) with a CenterlineSectionNormal array.")
    parser.add_argument("output_file", help="Output centerlines file (.vtp).")
    parser.add_argument("--num-workers", type=int, default=1,
        help="Number of threads used to extract slices.")
    args = parser.parse_args()

    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('[slice-results] %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    slice_results_files(args.results_file, args.centerlines_file, args.output_file,
        num_workers=args.num_workers)
