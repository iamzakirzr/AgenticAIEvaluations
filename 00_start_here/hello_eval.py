"""
YOUR FIRST EVALUATION -- 60 lines, 2 seconds, no model, no API key.

Run it:   make hello
          (or:  .venv/bin/python 00_start_here/hello_eval.py)

=============================================================================
READ THIS IF YOU COME FROM QA / SDET
=============================================================================
You already know how to do this. Watch:

    A NORMAL TEST                      AN EVALUATION
    -------------------------          ------------------------------------
    input: a request                   input: a question
    call the system under test         call the system under test
    assert output == expected          score output against expected
    pass / fail                        a number between 0 and 1
    a fixture file of test data        a "golden dataset"
    the assertion                      the "metric"
    a flaky test                       a non-deterministic judge
    boundary value analysis            adversarial / unanswerable questions
    code coverage                      dataset category coverage
    a CI gate on the test suite        a CI gate on the metric

The ONLY genuinely new idea is that the assertion returns a SCORE instead of
a boolean, because "is this answer good?" has no exact expected value. Every
other instinct you have transfers directly -- and most people building these
systems are worse at test design than you are.

This file is one complete evaluation, start to finish, with nothing hidden.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for path in (str(_ROOT), str(_ROOT / "02_langchain")):
    if path not in sys.path:
        sys.path.insert(0, path)

from pipeline import build_offline_pipeline

from core.golden import load_golden
from core.metrics import evaluate_retrieval


def main() -> None:
    print("=" * 74)
    print("YOUR FIRST EVALUATION")
    print("=" * 74)

    # -----------------------------------------------------------------------
    # STEP 1: the system under test.
    # -----------------------------------------------------------------------
    # A RAG chatbot: it searches a small knowledge base, then answers from what
    # it found. `build_offline_pipeline` gives us REAL search with a SCRIPTED
    # answer, so this runs instantly with no model installed -- exactly like
    # stubbing a slow downstream service in an integration test.
    system = build_offline_pipeline(["A scripted answer, citing passage [1]."])
    print(f"\nSTEP 1  System under test ready: {len(system.documents)} chunks indexed.")

    # -----------------------------------------------------------------------
    # STEP 2: the test data. In eval this is called a GOLDEN DATASET.
    # -----------------------------------------------------------------------
    # It is a fixture file. Each row has a question, the correct answer, and
    # -- crucially -- WHICH DOCUMENT the answer lives in. That last field is
    # what lets us score search quality with no AI involved at all.
    golden = load_golden()
    answerable = [item for item in golden if item.is_answerable]
    print(f"STEP 2  Golden dataset loaded: {len(golden)} questions "
          f"({len(answerable)} answerable, {len(golden) - len(answerable)} deliberately not).")

    # -----------------------------------------------------------------------
    # STEP 3: run the system over every question and record what it retrieved.
    # -----------------------------------------------------------------------
    print("STEP 3  Running the system over every question...")
    results = {}
    for item in answerable:
        trace = system.answer(item.question)
        # trace.retrieved_doc_ids = what the system FOUND
        # item.reference_doc_ids  = what it SHOULD have found
        results[item.id] = (trace.retrieved_doc_ids, item.reference_doc_ids)

    # -----------------------------------------------------------------------
    # STEP 4: score. This is the assertion, except it returns numbers.
    # -----------------------------------------------------------------------
    report = evaluate_retrieval(results)

    print("\nSTEP 4  Scores")
    print("-" * 74)
    print(f"  hit rate   {report.hit_rate:6.3f}   did we find the right document AT ALL?")
    print(f"  recall     {report.recall:6.3f}   what fraction of needed documents did we find?")
    print(f"  precision  {report.precision:6.3f}   what fraction of what we fetched was useful?")
    print(f"  MRR        {report.mrr:6.3f}   how HIGH did the right document rank?")
    print(f"  nDCG       {report.ndcg:6.3f}   rank quality, discounted by position")

    # -----------------------------------------------------------------------
    # STEP 5: the part beginners skip -- LOOK AT THE FAILURES.
    # -----------------------------------------------------------------------
    failures = report.failures()
    print(f"\nSTEP 5  Questions where search missed something: {len(failures)}")
    if failures:
        lookup = {item.id: item for item in answerable}
        for item_id in failures[:5]:
            print(f"    {item_id}  {lookup[item_id].question[:60]}")
    else:
        print("    none -- but read the caveat below before celebrating.")

    # -----------------------------------------------------------------------
    # STEP 6: the professional habit. Is this metric even useful?
    # -----------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("STEP 6  THE QUESTION THAT SEPARATES A TESTER FROM A SCORE-READER")
    print("=" * 74)
    print(
        "  hit rate and recall are 1.000 -- perfect. So is search perfect?\n"
        "\n"
        "  No. They are SATURATED: they sit at their maximum, so they cannot go\n"
        "  UP, and only a catastrophic change makes them go down. A metric with\n"
        "  no headroom cannot detect a regression, and gating a build on it is\n"
        "  theatre.\n"
        "\n"
        "  You already know this instinct from testing: an assertion that passes\n"
        "  no matter what the code does is not a test. Same idea, new context.\n"
        "\n"
        f"  Precision ({report.precision:.3f}) and MRR ({report.mrr:.3f}) DO have room to\n"
        "  move, so those are the ones worth gating on. 01_embeddings proves it\n"
        "  by sweeping chunk size and showing which numbers actually respond.\n"
    )

    # -----------------------------------------------------------------------
    # STEP 7: the questions with no answer. The most important rows.
    # -----------------------------------------------------------------------
    unanswerable = [item for item in golden if not item.is_answerable]
    example = unanswerable[0]
    trace = system.answer(example.question)
    print("=" * 74)
    print("STEP 7  THE ROWS THAT CATCH THE WORST BUG")
    print("=" * 74)
    print(f'  Question:  "{example.question}"')
    print("  The corpus does NOT contain this. Correct behaviour is to refuse.")
    print(f"  Search returned {len(trace.retrieved)} passages anyway, "
          f"top score {max(c.score for c in trace.retrieved):.3f}")
    print(
        "\n"
        "  Read that again: SEARCH ALWAYS RETURNS RESULTS. It has no concept of\n"
        "  'no match'. It handed the model four irrelevant passages with a\n"
        "  completely straight face.\n"
        "\n"
        "  That is why AI systems make things up. Not because the model is\n"
        "  stupid -- because nothing in the pipeline is allowed to say 'I don't\n"
        "  know' unless you explicitly build it.\n"
        "\n"
        "  A test set of only answerable questions would never find this. In QA\n"
        "  terms: you tested the happy path and shipped.\n"
    )

    print("=" * 74)
    print("WHAT YOU JUST DID")
    print("=" * 74)
    print(
        "  Ran a real system over a labelled dataset, scored it on five metrics,\n"
        "  found which questions failed, noticed that two of the metrics were\n"
        "  useless for gating, and identified the failure mode a naive test set\n"
        "  would miss. No AI model was involved at any point.\n"
        "\n"
        "  Next:  00_start_here/README.md   -- the 4-week plan\n"
        "         01_embeddings/            -- how search actually works\n"
    )


if __name__ == "__main__":
    main()
