"""The beginner notebooks in notebooks/learn/ run against the current code.

Every code cell of each notebook is executed in order in one namespace, as
Jupyter would, with IPython magics (lines starting with % or !) skipped and
plots closed instead of shown.  Only that the cells run is checked; the stored
outputs are regenerated with ``python notebooks/learn/build_learn.py
--execute``.
"""

import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
NOTEBOOKS = sorted(glob.glob(os.path.join(ROOT, "notebooks", "learn", "*.ipynb")))
# an IPython magic or shell command at the start of a line, but not a
# continuation line that starts with the != operator
MAGIC = re.compile(r"^(%|!(?!=))")


def _code(path):
    with open(path, encoding="utf-8") as f:
        cells = json.load(f)["cells"]
    for c in cells:
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            yield "\n".join(ln for ln in src.splitlines() if not MAGIC.match(ln))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=os.path.basename)
def test_notebook_runs(path, monkeypatch):
    monkeypatch.chdir(os.path.dirname(path))
    monkeypatch.setattr(plt, "show", lambda *a, **k: plt.close("all"))
    namespace = {"__name__": "__main__"}
    for i, code in enumerate(_code(path)):
        try:
            exec(compile(code, f"{os.path.basename(path)}[cell {i}]", "exec"), namespace)
        except Exception as e:
            pytest.fail(f"code cell {i} of {os.path.basename(path)} failed: {e!r}")
    plt.close("all")


def test_notebooks_exist():
    assert len(NOTEBOOKS) >= 3
