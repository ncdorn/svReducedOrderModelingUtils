"""
Regenerate tests/data/svslicer_u_tube.vtp with the C++ svSlicer executable.

    python tests/make_svslicer_reference.py /path/to/svslicer

The reference in the repository was created with svSlicer
(https://github.com/SimVascular/svSlicer, commit ed72a43 plus the local fixes for
current VTK cell traversal and case-insensitive array names) built against VTK 9.5.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import slicer_case

DATA_DIR = Path(__file__).parent / "data"


def run_svslicer(svslicer, output_file):
    with tempfile.TemporaryDirectory() as tmp_dir:
        results_file = Path(tmp_dir) / "results.vtu"
        centerlines_file = Path(tmp_dir) / "centerlines.vtp"
        slicer_case.write(slicer_case.make_results(), results_file)
        slicer_case.write(slicer_case.make_centerlines(slicer_case.centerline_arc_lengths()),
                          centerlines_file)
        subprocess.run([str(svslicer), str(results_file), str(centerlines_file), str(output_file)],
                       check=True, stdout=subprocess.DEVNULL, env={**os.environ, "OMP_NUM_THREADS": "1"})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python make_svslicer_reference.py /path/to/svslicer")
    DATA_DIR.mkdir(exist_ok=True)
    run_svslicer(sys.argv[1], DATA_DIR / "svslicer_u_tube.vtp")
