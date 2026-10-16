# svReducedOrderModelingUtils
Python package used to create input files for the svOneDSolver and svZeroDSoler reduced  order modeling (ROM) solvers.

## Installation

The package is published on PyPI as `svromutils`:

```bash
pip install svromutils
```


## Mapping 3D results onto centerlines

`svromutils-slice-results` maps 3D simulation results onto a centerline. It replaces the C++ [svSlicer](https://github.com/SimVascular/svSlicer) tool and takes the same arguments:

```bash
svromutils-slice-results <path/to/3d_result.vtu> <path/to/centerlines.vtp> <path/to/output.vtp> --num-workers 4
```

At each centerline point the 3D mesh is sliced with a plane normal to the `CenterlineSectionNormal` array. If the plane cuts the vessel more than once, only the part closest to the centerline point is used. Point data arrays whose names start with `pressure` (case-insensitive) are averaged over the slice area. Arrays starting with `velocity` are integrated over the slice to give the flow through it. The results are stored on the output centerlines under the original array names, so `velocity_00100` holds the flow at time step 100. Arrays with the wrong number of components (for example a scalar `velocity_magnitude`) are skipped with a warning.

If a slice has no area, the plane is moved toward the closest directly connected centerline point and retried, up to halfway along that segment. Neighbors are determined from the centerline's line cells, independently of point ID order. Connected points at the same position are skipped, and if the centerlines have a `BranchId` array, neighbors on the same branch are preferred. If no slice with a positive area can be extracted, an error is raised.

The same function is available from Python:

```python
from svromutils.slicer import slice_results_files
slice_results_files("post.vtu", "centerlines.vtp", "centerlines_results.vtp", num_workers=4)
```

## Tests

```bash
pip install -e ".[test]"
pytest tests
```

The slicer tests compare against reference output from the C++ svSlicer. Set `SVSLICER=/path/to/svslicer` to also run the comparison against a local svSlicer build.
