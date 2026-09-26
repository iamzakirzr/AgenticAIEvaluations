"""
A live dashboard over the deployed agent -- mountable onto the lesson 09 app.

    make serve-agent      ->  http://localhost:8001/live

=============================================================================
WHAT IS ON IT, AND WHAT IS DELIBERATELY NOT
=============================================================================
ON IT:   traffic, error rate, refusal rate, latency percentiles, the server
         mix, active alerts, and the label queue.

NOT ON IT: any quality score. There are no reference answers in production, so
a faithfulness number here would be invented. The page says so in a banner
rather than leaving a reader to assume the absence is an oversight.

=============================================================================
ONE PIECE OF DASHBOARD DESIGN WORTH COPYING
=============================================================================
Every panel shows the SAMPLE COUNT next to the number. "Refusal rate 0%" over
three requests and over three thousand are different facts that render
identically, and the first one gets acted on at 3am by someone who did not
check. Putting n beside the value costs nothing and prevents that entirely.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from live_monitor import LiveMonitor, promote_candidates

_PAGE = """<!doctype html>
<meta charset="utf-8"><title>Agent -- live</title>
<style>
 body{font:14px system-ui;margin:0;padding:24px;background:#0f1115;color:#e6e6e6}
 h1{font-size:18px;margin:0 0 4px} .sub{color:#8b93a7;margin-bottom:20px}
 .banner{background:#2a2410;border:1px solid #6b5a1c;padding:10px 12px;border-radius:6px;margin-bottom:20px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
 .card{background:#171a21;border:1px solid #262b36;border-radius:8px;padding:14px}
 .k{color:#8b93a7;font-size:12px} .v{font-size:24px;margin-top:4px}
 .n{font-size:11px;color:#6b7383;margin-top:2px}
 .alert{border-left:3px solid #d9534f;padding:8px 12px;margin:8px 0;background:#1d1416}
 .page{border-left-color:#ff4d4f} .ticket{border-left-color:#e8a33d}
 table{width:100%;border-collapse:collapse;margin-top:8px}
 td,th{text-align:left;padding:6px 8px;border-bottom:1px solid #262b36;font-size:13px}
</style>
<h1>Agent &mdash; live</h1>
<div class="sub">auto-refreshes every 3s</div>
<div class="banner"><b>These detect change, not quality.</b>
There are no reference answers in production, so nothing here says an answer
was correct. Quality is measured offline against the golden set (04 and 05).</div>
<div id="cards" class="grid"></div>
<h2 style="font-size:15px;margin-top:24px">Active alerts</h2><div id="alerts"></div>
<h2 style="font-size:15px;margin-top:24px">Label queue</h2><div id="queue"></div>
<script>
const pct = v => v==null ? '--' : (v*100).toFixed(1)+'%';
const ms  = v => v==null ? '--' : Math.round(v)+'ms';
async function tick(){
  const s = await (await fetch('live/snapshot')).json();
  const a = await (await fetch('live/alerts')).json();
  const n = s.samples || 0;
  document.getElementById('cards').innerHTML = [
    ['requests', n, ''],
    ['error rate', pct(s.error_rate), 'n='+n],
    ['refusal rate', pct(s.refusal_rate), 'n='+n],
    ['blocked', pct(s.blocked_rate), 'n='+n],
    ['p50', ms(s.p50_ms), 'n='+n],
    ['p95', ms(s.p95_ms), 'n='+n],
    ['p99', ms(s.p99_ms), 'n='+n],
    ['tools / request', s.tool_calls_per_request ?? '--', 'n='+n],
  ].map(([k,v,sub]) =>
    `<div class="card"><div class="k">${k}</div><div class="v">${v}</div><div class="n">${sub}</div></div>`
  ).join('') + `<div class="card"><div class="k">server mix</div><div class="v" style="font-size:14px">${
    Object.entries(s.server_mix||{}).map(([k,v])=>k+' '+v).join('<br>') || '--'}</div></div>`;

  document.getElementById('alerts').innerHTML = a.alerts.length
    ? a.alerts.map(x=>`<div class="alert ${x.severity}"><b>${x.severity.toUpperCase()}</b>
       &nbsp;${x.name}<br><span class="k">${x.detail}</span></div>`).join('')
    : '<div class="k">none</div>';

  document.getElementById('queue').innerHTML = a.candidates.length
    ? '<table><tr><th>question</th><th>why it was picked</th></tr>' +
      a.candidates.map(c=>`<tr><td>${c.question}</td><td class="k">${c.reason}</td></tr>`).join('') +
      '</table><div class="k" style="margin-top:8px">Each needs a HUMAN label before it joins the golden set.</div>'
    : '<div class="k">none</div>';
}
tick(); setInterval(tick, 3000);
</script>
"""


def live_router(monitor: LiveMonitor) -> APIRouter:
    """Three endpoints, mountable on any FastAPI app that owns this monitor."""
    router = APIRouter()

    @router.get("/live", response_class=HTMLResponse)
    async def page() -> str:
        return _PAGE

    @router.get("/live/snapshot")
    async def snapshot() -> dict:
        return monitor.snapshot()

    @router.get("/live/alerts")
    async def alerts() -> dict:
        return {
            "alerts": [
                {"name": a.name, "severity": a.severity, "detail": a.detail}
                for a in monitor.check()
            ],
            "candidates": [
                {"question": c.question, "reason": c.reason, "request_id": c.request_id}
                for c in promote_candidates(monitor, limit=10)
            ],
        }

    return router


__all__ = ["live_router"]
