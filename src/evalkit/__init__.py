"""evalkit: a DeepEval-based test automation framework for LLM apps and AI agents."""

import os

# Opt out of DeepEval/DeepTeam telemetry before either library is imported
# anywhere. An eval run must not phone home as a side effect.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPTEAM_TELEMETRY_OPT_OUT", "YES")

# DeepEval enforces its own per-call and per-metric timeouts, independent of
# the judge client's. Its defaults are sized for hosted APIs: on a CPU-only
# local judge, Faithfulness and Contextual Precision time out and are reported
# as errors. These are ceilings, so raising them costs nothing on a fast judge.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "3600")

__version__ = "0.1.0"
