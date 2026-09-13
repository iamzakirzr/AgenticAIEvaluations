"""
Smoke test for the first thing a learner runs. FAST TIER.

If `make hello` is broken, someone's introduction to the repo is a stack
trace. That is worth one test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for path in (str(_ROOT), str(_ROOT / "02_langchain")):
    if path not in sys.path:
        sys.path.insert(0, path)


def test_hello_eval_runs_end_to_end(capsys):
    spec = importlib.util.spec_from_file_location(
        "hello_eval", _ROOT / "00_start_here" / "hello_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    module.main()
    output = capsys.readouterr().out

    # All seven steps must appear -- a silent early return would still "pass"
    # a bare smoke test.
    for step in range(1, 8):
        assert f"STEP {step}" in output, f"hello_eval stopped before step {step}"

    assert "hit rate" in output
    assert "SATURATED" in output, "the saturation lesson went missing"
    # Step 7 must actually demonstrate the unanswerable case, not just mention it.
    assert "SEARCH ALWAYS RETURNS RESULTS" in output, (
        "the search-always-returns-results lesson went missing from step 7"
    )
