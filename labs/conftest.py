"""Shared pytest fixture for every lab.

Each lab folder holds `exercise.py` (your work: stubs that raise NotImplementedError),
`solution.py` (the reference), and a `test_labNN.py` that uses the `impl` fixture below.
The environment variable LAB_IMPL picks which file the tests import:

    pytest labs/03-linear-regression                         # tests your exercise.py (default)
    LAB_IMPL=solution pytest labs/                           # tests the reference solutions (CI does this)
"""
import importlib.util
import os
import pathlib

import pytest


def load_impl(test_path: pathlib.Path, which: str | None = None):
    which = which or os.environ.get("LAB_IMPL", "exercise")
    if which not in {"exercise", "solution"}:
        raise ValueError("LAB_IMPL must be 'exercise' or 'solution'")
    path = pathlib.Path(test_path).parent / f"{which}.py"
    name = f"{path.parent.name.replace('-', '_')}_{which}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def impl(request):
    return load_impl(request.path)
