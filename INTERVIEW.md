# Interview Preparation — AI/LLM Evaluation for QA and SDET Roles

Answer each out loud *before* reading the answer. Reciting definitions is worth
little; the questions below are the ones where a real answer separates you.

Every "strong answer" here is backed by something runnable in this repo, so you
can say "I built that" rather than "I read that".

---

## How to use this

Three tiers. Do them in order.

- **Tier 1 — Fundamentals.** You will be asked these. Non-negotiable.
- **Tier 2 — Judgement.** Where most candidates fall down.
- **Tier 3 — Tool specifics.** Only some interviews go here.

Then the **60-second project pitch** and the **questions to ask them**.

---

# Tier 1 — Fundamentals

### Q1. "How is testing an AI system different from testing normal software?"

**Weak answer:** "It's non-deterministic, so you can't use exact assertions."

That's true and shallow — it's the first sentence, not the answer.

**Strong answer:**

> Four differences, in order of how much they change my work.
>
> One, **assertions become scores.** There's no exact expected value for "is
> this a good answer", so instead of pass/fail I get 0.87 and have to decide
> what threshold means "shipped".
>
> Two, **the oracle can be wrong.** Often the thing grading the output is
> another model. So I have to test my measuring instrument before I trust its
> numbers — I measure judge/human agreement with Cohen's kappa.
>
> Three, **non-determinism is real but manageable.** Temperature 0 removes most
> of it. What's left I handle the way I'd handle a flaky test: measure the
> variance, then gate on a noise band rather than a fixed threshold.
>
> Four, **in production there's no reference answer**, so about half my metrics
> stop working. Faithfulness and refusal rate survive; context recall and answer
> correctness don't, because they need labels.
>
> What *doesn't* change is test design. Negative cases, boundary values,
> coverage of behaviours, a cheap tier and an expensive tier. That part is
> ordinary QA, and it's usually the part these systems are missing.

---

### Q2. "Explain RAG."

**Strong answer:**

> Four steps. Split documents into chunks; embed each chunk into a vector; embed
> the user's question and find the nearest chunks; paste those into the prompt
> and ask the model to answer from them. It exists because the model doesn't
> know your data and retraining isn't practical.
>
> The property I care about as a tester is that **step three always returns
> results**. Vector search has no concept of "no match" — ask it something the
> corpus doesn't cover and it returns the k least-irrelevant chunks with a
> straight face, no error. The model then answers from garbage unless the prompt
> explicitly permits refusal.
>
> So the first thing I check on any RAG system is whether the test set contains
> questions the corpus *can't* answer. If it doesn't, the suite cannot detect
> the worst failure mode, and the scores are meaningless.

**Follow-up they'll ask: "How would you test it?"**

> Split it in two, because the fixes are completely different. Retrieval
> metrics — recall@k, MRR, nDCG — tell me whether the right document was found
> at all; they need no model, so they're free, deterministic, and can gate every
> PR. Generation metrics — faithfulness, answer relevancy — need a judge, so
> they're slow and noisy and I run them nightly.
>
> If retrieval is bad, no prompt engineering will save you. So I fix that first.

---

### Q3. "What is an embedding?"

**Strong answer:**

> Text converted to a list of numbers positioned so that similar meanings land
> near each other. Similarity is the cosine of the angle between two vectors —
> −1 to 1, where 1 is the same direction. Cosine rather than Euclidean distance
> because distance is sensitive to magnitude, which mostly reflects document
> length rather than meaning.
>
> The practical thing I watch for: **the query and the documents must be
> embedded by the same model.** Different models put text in unrelated vector
> spaces. When ingestion upgrades its embedding model and the query service
> doesn't, nothing raises — dimensions may even match — and every result is
> noise. I've built a fingerprint check that refuses to serve on a mismatch,
> because failing loudly at startup beats failing silently on every request.

---

### Q4. "Name some RAG metrics and what each catches."

Have this table in your head:

| Metric | Question it answers | Needs a judge? |
|---|---|---|
| recall@k | Did we fetch the right document at all? | No |
| precision@k | How much of what we fetched was useful? | No |
| MRR | How *highly* did the right document rank? | No |
| Faithfulness | Are the answer's claims supported by the context? | Yes |
| Answer relevancy | Does the answer address the question? | Yes |
| Context precision | Were the retrieved chunks relevant, rank-weighted? | Yes |
| Context recall | Was everything needed actually retrieved? | Yes |
| Noise sensitivity | Do irrelevant chunks cause wrong claims? (**lower is better**) | Yes |

Say the last row out loud in the interview. Knowing one metric inverts shows
you've used them rather than listed them.

---

### Q5. "What's the difference between faithfulness and correctness?"

This is the highest-value question in the whole list. Most candidates conflate
them.

**Strong answer:**

> Faithfulness asks "does the answer follow from the retrieved context?" It never
> asks "is the answer true". So **an answer that faithfully repeats a wrong
> document scores 1.0.**
>
> That's why the combination matters: high faithfulness with low correctness
> almost always means retrieval fetched the wrong document and the generator
> summarised it perfectly. Faithfulness alone would call that a success.
>
> I have a test that asserts exactly this — it feeds a deliberately wrong
> context and checks faithfulness stays *high*, because if it dropped, the metric
> would be measuring something other than its name.

---

# Tier 2 — Judgement

### Q6. "Your faithfulness score is 0.87. Is that good?"

The question is a trap. The answer is another question.

**Strong answer:**

> On its own, that number tells me nothing, and I'd want four things before
> acting on it.
>
> **What's the spread?** One run has no error bar. If I run the same evaluation
> three times and get 0.87, 0.79, 0.91, then 0.87 is noise and I can't detect
> anything smaller than about 0.12.
>
> **What was it yesterday?** An absolute number is far less useful than a delta
> against a baseline recorded under the same configuration.
>
> **Is the judge any good?** If it scores kappa 0.3 against my own labels, 0.87
> is a number generated by a coin.
>
> **What's the distribution?** A mean of 0.87 could be everything at 0.87, or
> half at 1.0 and half at 0.74. Those need completely different fixes, and the
> mean hides which one you have.

---

### Q7. "How would you put LLM evaluation in CI?"

**Weak answer:** "Run the eval suite on every PR and fail below a threshold."

That fails within a fortnight and it's worth saying why.

**Strong answer:**

> Two tiers, and only one of them gates a PR.
>
> The **PR gate** runs only what needs no model: retrieval metrics, dataset
> integrity, prompt structure, citation validation, agent trajectories. Free,
> deterministic, about fifteen seconds. That's most of what matters and it never
> flakes.
>
> The **judged tier** runs nightly against a real model. It doesn't gate a PR,
> because a judged metric varies between identical runs — gate on it and the
> build goes red on sampling noise, someone adds `continue-on-error: true`, and
> then nobody reads CI at all. That outcome is worse than not having the check.
>
> For the nightly gate I compare against a recorded baseline with a **noise
> band** — a regression is a drop larger than `max(min_delta, 2 × observed
> stdev)`. And the baseline records a **fingerprint** of the models and chunking
> that produced it; if that changed, the gate *disables itself* and reports the
> numbers as informational rather than failing a build for a change someone made
> on purpose.

**Follow-up: "What stops someone just re-recording the baseline to go green?"**

> Nothing technical, which is why re-recording goes through a pull request with
> a required reason and a review checklist. Moving a baseline is how a
> regression disappears — one "re-record" at a time, every step inside the noise
> band. It has to be a reviewed decision with a visible diff.

---

### Q8. "How do you know your LLM judge is any good?"

If you can answer this well you are ahead of most candidates.

**Strong answer:**

> I calibrate it. Hand-label a sample — say 40 outputs, pass/fail — run the judge
> over the same sample, and compute **Cohen's kappa**: agreement corrected for
> chance.
>
> Raw agreement lies on imbalanced data. If 90% of answers are good, a judge that
> says "pass" unconditionally gets 90% agreement and has learned nothing. Kappa
> correctly scores that 0.0.
>
> My rule: above 0.6 I'll gate on it, 0.4 to 0.6 I'll track the trend but not
> fail builds, below 0.4 the metric is noise and I say so.
>
> I'd rather report "the local 8B judge scored kappa 0.31 on faithfulness so I
> gated on retrieval metrics instead" than show a green dashboard nobody checked.

**Follow-up: "What biases do judges have?"**

> Position bias — prefers whichever answer comes first; fix by running each
> comparison twice with the order swapped. Verbosity bias — rates longer answers
> higher. Self-preference — a model rates its own output more favourably, which
> is why the generator and judge should be configured separately. And leniency —
> everything clusters at the top of the scale.
>
> Mitigations that work: temperature 0, explicit rubrics rather than "rate the
> quality", reasoning *before* the verdict rather than after, and decomposing the
> judgement — faithfulness works precisely because it checks claims one at a time
> instead of judging holistically.

---

### Q9. "Design a test suite for a customer-support RAG chatbot."

They want structure. Give them the dataset first — that's the tell.

**Strong answer:**

> I'd start with the dataset, because that determines what the suite can detect
> at all. Four categories:
>
> - **Single-hop** — the answer is in one document. Baseline competence.
> - **Multi-hop** — needs two documents combined. Breaks naive top-k retrieval.
> - **Unanswerable** — the corpus genuinely doesn't cover it, and the correct
>   behaviour is refusal. **The most important category**, because a suite
>   without it cannot detect confident fabrication.
> - **Adversarial** — false premises, leading questions, prompt injection in the
>   input. Catches sycophancy and jailbreaks.
>
> Every row records which documents *should* be retrieved, which makes recall@k
> and MRR computable with no model at all.
>
> Then three tiers of check. Free and deterministic: retrieval metrics, refusal
> detection by regex, citation validity, PII leakage. Judged nightly:
> faithfulness, answer relevancy, context precision and recall. And safety, run
> on every change: toxicity, PII, prompt-injection resistance, staying in scope.
>
> For a support bot specifically I'd add multi-turn checks — does it remember
> what the customer said three turns ago, does it stay in role, does it refuse
> to give refund amounts it isn't authorised to.

---

### Q10. "The model provider ships a new version. What do you do?"

**Strong answer:**

> Treat it exactly like a dependency upgrade with no changelog, because that's
> what it is.
>
> Run the full evaluation on both versions against the same golden dataset with
> the same configuration, and diff the per-category scores rather than the
> overall mean — a model can improve on average while getting worse at refusing.
>
> Critically, I'd **change one thing at a time**. If the new model is also the
> judge, I can't attribute any movement. So I pin the judge, swap only the
> generator, measure; then swap the judge separately.
>
> And I'd check latency, tokens and cost, not just quality. A 2% faithfulness
> gain for triple the latency is usually a bad trade, and that's a conversation
> you can only have with numbers.

---

### Q11. "How do you test an agent?"

**Strong answer:**

> You score the **trajectory**, not just the final answer. An agent that reached
> the right answer after nine wasted tool calls and one wrong turn looks
> identical, by outcome, to one that went straight there.
>
> Four failures I test explicitly, all deterministically with a scripted model:
> infinite loops — caught by a step limit and by detecting repeated identical
> calls; wrong tool selection — compare called tools against expected;
> right tool with wrong arguments, which is a distinct failure; and stopping
> early.
>
> Loop detection matters most in production because it's invisible to
> outcome-only testing — a looping agent never returns a *wrong* answer, it just
> never returns, and it surfaces as a timeout rather than a quality bug.
>
> A design trick that makes this cheap: make refusal a **tool call** rather than
> free text. Then "did it refuse?" is a boolean in the trajectory instead of a
> judged question.

---

### Q12. "What would you do first, joining a team with an LLM feature and no evaluation?"

**Strong answer:**

> Build the golden dataset. Nothing else is useful without it, and it's the part
> nobody else will do.
>
> I'd get 50 real questions from actual traffic or support tickets — not
> invented ones, because invented questions test what we imagined users ask.
> Label the expected answer and expected source documents. Deliberately include
> unanswerable and adversarial cases.
>
> Then the cheapest possible gate: retrieval metrics in CI. No model, no cost, no
> flakes, and it catches the largest class of failures. That earns the credibility
> to ask for judged metrics later.
>
> In parallel I'd add tracing, because production traffic is the best source of
> new dataset rows — the gap between what we imagined and what users actually ask
> is where the failures live.
>
> What I would *not* do first is stand up a judged metric dashboard. It's the
> most visible thing and the least trustworthy until the dataset and the judge
> are sound.

---

# Tier 3 — Tool specifics

### Q13. "Have you used DeepEval or RAGAS? What's the difference?"

> Both. They overlap heavily on RAG metrics and differ in shape.
>
> DeepEval is **pytest-shaped**: an `LLMTestCase` with `input`, `actual_output`,
> `retrieval_context`, metrics you attach, and `assert_test` so it runs in an
> ordinary pytest suite. It has G-Eval, which lets you define a metric by writing
> the criterion in a sentence — useful when your failure mode is domain-specific.
>
> RAGAS is **dataset-shaped**: `SingleTurnSample` objects collected into an
> `EvaluationDataset` and scored in bulk. It has noise sensitivity, which
> DeepEval doesn't, and factual correctness split into precision and recall so
> you can tell "we added wrong claims" from "we missed right ones".
>
> They use different names for the same fields — `actual_output` versus
> `response`, `retrieval_context` versus `retrieved_contexts` — so I keep a
> separate small adapter for each rather than a unified wrapper. A unified
> interface has to pick one vocabulary and then misleads anyone reading the other
> library's docs.
>
> The reason to run both: when they disagree sharply on the same case, at least
> one judge is unreliable. Neither can tell you that alone.

**If they push on version specifics — this is where you sound like you've
actually shipped it:**

> A couple of real traps. `ragas` declares `langchain-community` unpinned, and
> when that package removed its Vertex AI chat model, `import ragas` broke
> outright — nothing to do with your code. And RAGAS's current metric API is
> `ragas.metrics.collections`; the path in essentially every tutorial online is
> deprecated. On the DeepEval side, `ToolCorrectnessMetric` is a purely
> deterministic comparison that nonetheless constructs an OpenAI client in its
> constructor and fails without an API key.

---

### Q14. "LangChain vs LangGraph?"

> LangChain is orchestration — common interfaces over models, embeddings, vector
> stores, splitters and prompts, so swapping a provider is a one-line change.
> LangGraph, from the same authors, is a state machine for agents: nodes, edges,
> conditional edges, and a checkpointer for persistence.
>
> Use LangChain when the flow is a fixed pipeline. Use LangGraph when it loops or
> branches based on model output.
>
> One version warning worth knowing: **LangChain 1.x is a substantial rewrite of
> 0.3**, and most tutorials online still target the old one, so copied snippets
> fail on imports.
>
> The LangGraph feature I'd highlight is `interrupt()` — it suspends the graph
> and persists state so a human can approve a dangerous action, then resumes from
> exactly that point, possibly in a different process. Two things catch people:
> it requires a checkpointer, and **the node re-runs from the top on resume**, so
> anything before the interrupt happens twice. I put the approval check before
> any side effect for that reason. And my approval logic **fails closed** —
> anything that isn't an explicit recognised "yes" means no, because a guard that
> fails open is worse than no guard.

---

### Q15. "What does an observability tool add that offline evaluation doesn't?"

> Real questions, and the discovery that **in production there is no reference
> answer**. Nobody labelled the user's input, so reference-based metrics —
> context recall, answer correctness, recall@k — are simply unavailable. What
> survives is reference-free: faithfulness, answer relevancy, toxicity, PII, and
> cheap deterministic signals like refusal rate and retrieval confidence.
>
> Practically, three things matter. **Redact PII before the span is created**,
> not before it's exported — once it's a span attribute another thread is already
> shipping it to a third party. **Sample with a tail**, not just a head: pure
> random sampling at 10% throws away nine of every ten incidents, so I always
> keep errors, refusals, slow requests and low-confidence retrievals. And **alert
> in both directions** — everyone alerts on refusals going up; the one people
> forget is refusals *collapsing*, which means the model stopped refusing and
> started confabulating.
>
> The biggest payoff is the loop back to offline: production questions become new
> golden dataset rows. Though the reference answer still has to be written by a
> human — auto-filling it from the system's own output builds a dataset that
> certifies current behaviour as correct, which hides every existing bug forever.

---

### Q16. "How would you test a multi-step prompt chain?"

> Not by scoring the final output. A four-link chain has five failure modes —
> each link, plus the composition — and all of them present identically as "the
> answer is wrong", because only the last link's output is visible.
>
> Three things. **Trace every link**, so a failure has an address instead of a
> symptom; that also distinguishes "the link ran and failed" from "the link
> never ran", which a chain-level assertion cannot, and in a branching chain the
> second is the more common cause. **Contract-check each boundary** — the `|`s
> are integration seams, and contract tests at seams are ordinary SDET work. The
> nastiest one is a non-empty check: an empty link output doesn't raise, it gets
> formatted into the next prompt as nothing at all and the model answers a
> question it was never asked. **Measure per-link success**, because that tells
> you which link to fix.
>
> Then the arithmetic, which usually changes the design: four links at 95% each
> is an 81% chain. Reliability is bounded above by your worst link, so "improve
> every prompt" is the wrong response. And 99% over five links needs 99.8% per
> link — if you can't build that link, the fix isn't a better prompt, it's fewer
> links.

---

### Q17. "You've connected an agent to several MCP servers. What's your test strategy?"

> Start with what changed: my agent's tool surface is now defined by a process I
> don't control. A server upgrade can rename a tool or reword a description and
> my agent's behaviour changes with no diff in my repository. I can't unit-test
> their servers, so I test the seam.
>
> Three things. **Snapshot the tool contract** — names, required arguments, and
> descriptions. Descriptions especially, because the description is the prompt
> the model routes on, so a reworded one is a behaviour change with a
> byte-identical schema. **Measure tool-selection accuracy** against labelled
> cases, which needs no judge: I label which server should serve each question
> and count. And **test degradation** — with one server down, does the agent
> still start?
>
> Tool selection is the metric I'd lead with, because asking the wrong server
> produces a fluent, well-cited answer that is *faithful* to the passages it was
> given. No RAG metric can see it; the error is upstream of everything they
> measure.
>
> The traps I'd mention, all of which I hit: two servers exporting the same tool
> name collide silently and tool choice falls back to list order; one dead server
> raises an ExceptionGroup out of the shared TaskGroup and you get no tools from
> *any* server; MCP tools are async-only, so `.invoke()` raises at the first tool
> call after binding and planning succeed — which means it passes unit tests and
> fails in integration.

---

### Q18. "What does it take to expose an agent publicly?"

> First, the framing: it isn't a deployment task. A public agent endpoint is a
> remote tool execution service, driven by untrusted text, billed to me. A
> stranger with the URL gets arbitrary execution of every tool I bound, my
> inference spend with no natural ceiling, and a prompt-injection surface that
> includes my own retrieved documents — the user's message isn't the only
> untrusted input.
>
> An API key tells me *who*, which is necessary and not close to sufficient. The
> control people leave out is a **per-key tool allowlist**: a leaked read-only
> key is a leak, a leaked key that can fetch URLs is an SSRF proxy on my egress
> IP. I'd enforce it twice — preventively by binding only permitted tools, and
> detectively by checking afterwards, because if the detective control ever fires
> the preventive one has a hole and I want my alert to tell me, not my bill.
>
> Then: a spend budget separate from the rate limit, because rate isn't cost and
> one agent request can fan out into a dozen model calls; an input length cap,
> since the largest input I accept is the largest single bill I can be handed; a
> hard timeout, because an agent loop with no deadline is unbounded; and output
> guards for PII and system-prompt leaks.
>
> One detail I'd flag: in-process rate limiting multiplies by worker count. Four
> uvicorn workers serve four times your stated limit, and nearly every FastAPI
> tutorial has that bug. Limit at the gateway or use shared state.
>
> And I'd say plainly that this is the minimum, not a security review. Genuinely
> public means a WAF, real identity, and egress control in front of it.

---

### Q19. "How do you monitor an LLM system in production?"

> Lead with the limit: **you cannot measure quality live**, because there are no
> reference answers. Refusal rate, latency, tool mix, citation validity — every
> one is a proxy. They detect *change*, not correctness, and conflating the two
> is how you get a green dashboard above a system that's quietly wrong. I'd put
> that on the dashboard itself rather than let a reader assume otherwise.
>
> Inside that limit: **error-budget burn rate on two windows**, not a threshold.
> "Alert above 1%" fires at 3am for a two-minute blip and stays silent through a
> week at 0.9% that eats the whole quarterly budget. With a 99% SLO the budget is
> 1%; burn rate is observed rate over budget. 14.4× on a one-hour window pages,
> 6× on six hours tickets. Fast catches the outage, slow catches the bleed.
>
> **Refusal rate in both directions.** Everyone watches the spike. The collapse
> is the dangerous one: the system stopped abstaining and started confabulating,
> and latency improves while the refusal graph goes down, which most dashboards
> draw green.
>
> For a multi-tool agent, **tool-mix drift** — traffic silently moving from the
> internal server to the web one is a behaviour change with no error and no code
> change. I'd measure it with total variation distance rather than KL, because KL
> is undefined when a category disappears entirely and that's exactly the
> interesting case.
>
> And the loop back: anomalous traffic — blocked, errored, refused, unusually
> long — becomes candidates for the golden set. Humans label them. Auto-filling
> the reference from the system's own output builds a dataset that certifies
> current behaviour as correct and hides every existing bug forever.


---

# The 60-second project pitch

Practise until it's smooth. Concrete numbers and one honest limitation.

> I built a curriculum repo for evaluating RAG systems and agents with
> LangChain, LangGraph, DeepEval, RAGAS and LangWatch, running against local
> open-source models through Ollama.
>
> It goes from an embeddings walkthrough through to an agent on a public
> endpoint with a live dashboard, including MCP, so the same system is built,
> exposed and then watched.
>
> The design decision I'd highlight is the two-tier test strategy. Over 500
> tests run with no model, no GPU and no API key — retrieval quality, dataset
> integrity, prompt structure, agent trajectories, PII redaction, real MCP
> protocol round trips, API auth, the regression-gate maths. Judged metrics are
> behind a marker and run nightly, never on a PR, because gating on a noisy
> judge teaches everyone to ignore CI.
>
> Two findings I'm most pleased with. First, recall on my dataset is *saturated*
> — it moves by 0.012 across a sixteen-fold change in chunk size — so gating on
> it would be theatre, and the gate uses MRR instead. Second, my CI once posted
> a pull-request comment claiming faithfulness improved from 0.900 to 1.000 on a
> run that computed no faithfulness at all: a unit test had left fixture data in
> the artifacts directory and the workflow published it. I fixed the call sites,
> added an autouse fixture that fails any test writing to that directory, and
> made CI produce the report rather than find it.
>
> A third, if there's time: I wanted a similarity threshold so the system would
> refuse when retrieval was weak. I measured it first — 42 real questions against
> 200 random strings — and the distributions overlap: gibberish tops out at
> 0.433, real questions bottom out at 0.208. No threshold separates them. So I
> shipped the score as a hint rather than a gate, and pinned the overlap in a
> test so nobody "fixes" it later.
>
> The honest limitation is that the corpus is only eight documents and the
> questions reuse its vocabulary, which is exactly why recall saturates. Real
> domain documents would be the next change.

That second finding is the strongest thing in the pitch. It shows you catch the
class of bug the whole discipline is about — a number that looks like a
measurement and isn't — **in your own work**.

---

# Questions to ask them

Good questions here are also answers. Each of these signals experience.

1. "Do you have a golden dataset, and who maintains it?" — tells you if
   evaluation is real or aspirational.
2. "Have you measured how much your metrics move between identical runs?" —
   almost nobody has, and it reveals whether their gates are meaningful.
3. "Do judged metrics gate a PR, or run on a schedule?" — if they gate PRs, ask
   how often people override the check.
4. "How do you know your judge agrees with a human?" — the single best question
   you can ask. Watch their face.
5. "What happens to a trace from production — does it ever become a test case?"
6. "When the model provider ships a new version, what's the process?"

---

# Two-minute pre-interview refresher

- Assertions become **scores**; the oracle can be **wrong**; production has **no
  reference answer**.
- **Search always returns results.** No concept of "no match". That's why RAG
  hallucinates.
- **Faithfulness ≠ correctness.** Faithfully repeating a wrong document scores
  1.0.
- **Noise sensitivity: lower is better.** The metric that inverts.
- **Cohen's kappa**, not raw agreement. >0.6 gate, <0.4 don't trust.
- **A saturated metric can't detect a regression.** Check it can move.
- **Temperature 0** in every test, especially the judge.
- **Gate on a noise band**, not a fixed threshold; disable the gate when the
  config fingerprint changes.
- **Agents: score the trajectory.** Loops are invisible to outcome-only tests.
- **Unanswerable questions** are the most important rows in the dataset.
