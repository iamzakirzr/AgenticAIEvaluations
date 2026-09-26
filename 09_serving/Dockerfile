# =============================================================================
# The agent gateway, containerised.
# =============================================================================
# Read 09_serving/README.md before deploying this anywhere reachable. The image
# is the easy part; the tool allowlist, egress control and gateway-level rate
# limiting are the parts that matter.
# =============================================================================

FROM python:3.11-slim

# Non-root. An agent runtime executes tools chosen by a language model
# influenced by untrusted text -- it is exactly the process you do not want
# running as root.
RUN useradd --create-home --uid 10001 agent
WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && uv pip install --system --no-cache -e .

COPY --chown=agent:agent . .
USER agent

ENV PYTHONUNBUFFERED=1 \
    DEEPEVAL_TELEMETRY_OPT_OUT=YES

EXPOSE 8001

# The container probe uses LIVENESS, never readiness -- see the README. A probe
# on /readyz would restart the container whenever a downstream MCP server
# blipped, converting a degraded dependency into a restart loop.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8001/healthz')"

# ONE worker on purpose. The in-process rate limiter multiplies by worker count
# (see test_in_process_limiting_is_per_worker). Scale with replicas behind a
# gateway that does the limiting, or move the limiter to shared state first.
CMD ["python", "09_serving/public_api.py"]
