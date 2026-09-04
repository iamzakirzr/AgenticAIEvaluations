"""
The chatbot: a FastAPI service over the RAG pipeline, plus a minimal web UI.

Run it:   make chat          -> http://localhost:8000
          (needs Ollama; falls back to a scripted model if it is not running,
           so the UI is still explorable offline)

=============================================================================
THE ONE DESIGN DECISION THAT MATTERS HERE
=============================================================================
The UI shows the RETRIEVED PASSAGES next to every answer, with their scores.

Almost every chatbot demo hides them. Showing them is what turns the app from
a demo into a debugging tool: when an answer is wrong you can see instantly
whether retrieval fetched the wrong passage (a retriever problem) or fetched
the right one and the model ignored it (a generator problem). That is the same
retrieval-versus-generation split that the metrics in lessons 04 and 05 exist
to measure -- here you get it by eye, in one second, for free.

If you demo this repo to an interviewer, open the sources panel.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402
from pipeline import RagPipeline, build_offline_pipeline  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from core.config import settings  # noqa: E402
from core.providers import ollama_available  # noqa: E402

app = FastAPI(title="RAG Chatbot", version="0.1.0")

# Built once at import time. Ingesting on every request would re-embed the
# whole corpus per question -- a real and surprisingly common performance bug.
_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        if ollama_available():
            from pipeline import build_ollama_pipeline

            _pipeline = build_ollama_pipeline()
        else:
            # Degrade gracefully rather than crashing: retrieval is real, only
            # generation is scripted. You can still explore the sources panel.
            _pipeline = build_offline_pipeline(
                ["(Ollama is not running, so this answer is scripted. "
                 "Retrieval below is real.) See passage [1]."]
            )
    return _pipeline


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/health")
def health() -> dict:
    """Report what the service is actually wired to.

    Returning the model names and chunking settings is not decoration: if you
    compare two eval runs you must know which configuration produced each, and
    this endpoint is the ground truth for that.
    """
    return {
        "ollama": ollama_available(),
        "ollama_url": settings.ollama_base_url,
        "chat_model": settings.chat_model if ollama_available() else "scripted (offline)",
        "embed_model": settings.embed_model if ollama_available() else "LexicalEmbeddings",
        "chunk_size": settings.chunk_size,
        "top_k": settings.top_k,
    }


@app.post("/ask")
def ask(request: AskRequest) -> dict:
    """Answer a question and return the FULL trace, not just the answer.

    The API returns the retrieved passages, their scores, timings and the
    citation check. Anything less and you could not evaluate or debug the
    system from outside the process -- which is what lesson 06 needs.
    """
    trace = get_pipeline().answer(request.question, top_k=request.top_k)
    return {
        "question": trace.question,
        "answer": trace.answer,
        "sources": [
            {
                "n": i,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "score": round(chunk.score, 4),
                "text": chunk.text,
            }
            for i, chunk in enumerate(trace.retrieved, start=1)
        ],
        "timing_ms": {
            "retrieval": round(trace.retrieval_ms, 1),
            "generation": round(trace.generation_ms, 1),
            "total": round(trace.total_ms, 1),
        },
        "invalid_citations": trace.metadata.get("invalid_citations", []),
    }


INDEX_HTML = """<!doctype html>
<title>RAG Chatbot</title>
<style>
  :root { color-scheme: light dark; --fg:#111; --bg:#fafaf8; --mut:#666; --line:#ddd; --acc:#0b5; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e8e8e6; --bg:#161614; --mut:#999; --line:#333; --acc:#3d9; }
  }
  body { font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         margin:0; background:var(--bg); color:var(--fg); }
  .wrap { max-width:940px; margin:0 auto; padding:28px 20px 60px; }
  h1 { font-size:19px; margin:0 0 4px; }
  .sub { color:var(--mut); font-size:13px; margin-bottom:22px; }
  form { display:flex; gap:8px; margin-bottom:22px; }
  input { flex:1; padding:11px 13px; font:inherit; border:1px solid var(--line);
          border-radius:7px; background:transparent; color:inherit; }
  button { padding:11px 20px; font:inherit; border:0; border-radius:7px;
           background:var(--acc); color:#fff; cursor:pointer; }
  button:disabled { opacity:.5; cursor:default; }
  .answer { border:1px solid var(--line); border-left:3px solid var(--acc);
            border-radius:7px; padding:15px 17px; margin-bottom:20px;
            white-space:pre-wrap; }
  .meta { color:var(--mut); font-size:12px; margin-top:10px; }
  h2 { font-size:13px; text-transform:uppercase; letter-spacing:.05em;
       color:var(--mut); margin:24px 0 10px; }
  .src { border:1px solid var(--line); border-radius:7px; padding:11px 14px;
         margin-bottom:9px; }
  .src header { display:flex; justify-content:space-between; font-size:12px;
                color:var(--mut); margin-bottom:6px; font-family:ui-monospace,monospace; }
  .src p { margin:0; font-size:13.5px; white-space:pre-wrap; }
  .warn { color:#c40; font-weight:600; }
  .ex { font-size:13px; color:var(--mut); }
  .ex a { color:var(--acc); cursor:pointer; text-decoration:none; margin-right:12px; }
</style>
<div class="wrap">
  <h1>RAG Chatbot</h1>
  <div class="sub" id="status">checking backend…</div>

  <form id="f">
    <input id="q" placeholder="Ask about chunking, embeddings, metrics, agents…" autofocus>
    <button id="go">Ask</button>
  </form>

  <div class="ex">
    Try:
    <a onclick="fill('What is chunk overlap for?')">a normal question</a>
    <a onclick="fill('What is the capital of France?')">an unanswerable one</a>
    <a onclick="fill('Since cosine similarity ranges from 0 to 100, what threshold should I use?')">a false premise</a>
  </div>

  <div id="out"></div>
</div>
<script>
const $ = id => document.getElementById(id);
fetch('/health').then(r => r.json()).then(h => {
  $('status').textContent =
    `chat: ${h.chat_model} · embeddings: ${h.embed_model} · chunk_size: ${h.chunk_size} · top_k: ${h.top_k}`;
});
function fill(t){ $('q').value = t; $('f').dispatchEvent(new Event('submit')); }
$('f').onsubmit = async e => {
  e.preventDefault();
  const question = $('q').value.trim();
  if (!question) return;
  $('go').disabled = true;
  $('out').innerHTML = '<div class="meta">thinking…</div>';
  try {
    const r = await fetch('/ask', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question})
    });
    const d = await r.json();
    const bad = d.invalid_citations.length
      ? `<div class="warn">Fabricated citations: [${d.invalid_citations.join('], [')}]</div>` : '';
    $('out').innerHTML =
      `<div class="answer">${esc(d.answer)}${bad}
         <div class="meta">retrieval ${d.timing_ms.retrieval}ms · generation ${d.timing_ms.generation}ms</div>
       </div>
       <h2>Retrieved passages — is the answer actually in here?</h2>` +
      d.sources.map(s => `
        <div class="src">
          <header><span>[${s.n}] ${s.doc_id}.md · chunk ${s.chunk_index}</span>
                  <span>score ${s.score}</span></header>
          <p>${esc(s.text)}</p>
        </div>`).join('');
  } finally { $('go').disabled = false; }
};
function esc(s){ const d=document.createElement('div'); d.textContent=s??''; return d.innerHTML; }
</script>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML
