"""evalkit: a DeepEval-based test automation framework for LLM apps and AI agents."""

import os

# Opt out of DeepEval/DeepTeam telemetry before either library is imported
# anywhere. An eval run must not phone home as a side effect.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPTEAM_TELEMETRY_OPT_OUT", "YES")

__version__ = "0.1.0"
