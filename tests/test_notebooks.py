import importlib.util
from pathlib import Path

import pytest

NOTEBOOKS = sorted(Path("notebooks").glob("*.py"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_runs_headless(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    outputs, definitions = module.app.run()
    assert len(outputs) > 0
    assert definitions
