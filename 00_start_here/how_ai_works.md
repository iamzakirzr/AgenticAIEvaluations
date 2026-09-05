# How AI and ML Actually Work — for Testers

No maths beyond arithmetic. Everything here is what you need to *test* these
systems; nothing is here to make you a researcher.

---

## 1. Machine learning in one page

Normal software:

```
you write the rules  →  data goes in  →  answers come out
```

Machine learning:

```
data + answers go in  →  the machine derives the rules  →  new data gets answers
```

Those derived "rules" are millions or billions of numbers called **weights**.
Nobody wrote them and nobody can read them. That's the whole reason AI testing
is different: **you cannot inspect the logic, so you can only characterise the
behaviour.**

Which is, incidentally, black-box testing. You've done this before.

### Training vs inference — the distinction that matters most

| | Training | Inference |
|---|---|---|
| When | Once, before release | Every request |
| Changes the weights? | Yes | **No** |
| Who does it | The model provider | You |
| What you test | Not this | **This** |

The model does **not** learn from your users. It does not remember yesterday.
Any "memory" you see is your own code storing conversation history and pasting
it back into the prompt. (`03_langgraph` builds exactly that.)

---

## 2. What a language model actually does

One thing, repeatedly: **given some text, predict the next token.**

```
"The capital of France is"  →  " Paris"  (91%)  " a"  (3%)  " located"  (2%)  ...
```

Then it appends the choice and does it again. That's it. Everything else —
answering questions, writing code, following instructions — is that loop
scaled up on enough training data.

### Three consequences you will meet in testing

**It's a probability distribution, so output varies.** `temperature` controls
how much. At `0` it always takes the top choice, which is as deterministic as
you get. **Set temperature to 0 in every test**, especially for a judge —
sampling noise in your measuring instrument is pure downside.

**It has no idea whether it's right.** There is no truth check inside. "Paris"
and a confidently invented API method are produced by exactly the same
mechanism. This is why hallucination isn't a bug to be patched out — it's the
mechanism working as designed on a question it can't ground.

**It cannot say "I don't know" unless that's a likely continuation.** Text on
the internet rarely says "I don't know", so the model rarely does either. You
have to *instruct* it to, and then *measure* whether it complied. That single
prompt line is the highest-value change in a RAG system, and
`02_langchain/prompts.py` sets up an A/B to prove it.

---

## 3. Tokens

Models don't see characters or words. Text is split into **tokens** — roughly 4
characters of English.

```
"Testing AI systems"  →  ["Test", "ing", " AI", " systems"]   (4 tokens)
```

You care because tokens are the unit of:

- **Cost** — billed per input and output token
- **Limits** — the context window is a token budget
- **Latency** — output tokens are generated one at a time, so a long answer is
  slow by construction

Rough conversion: **1 token ≈ 4 characters ≈ 0.75 words**. Good enough for
capacity planning; `core/resilience.py` uses exactly this ratio.

---

## 4. Embeddings — the part you'll test most

An **embedding** converts text into a list of numbers, positioned so that
similar meanings are close together.

```
"How do I reset my password?"  →  [0.021, -0.113, 0.087, ... ]   (768 numbers)
"I forgot my login"            →  [0.019, -0.108, 0.091, ... ]   ← nearby
"What is the refund policy?"   →  [-0.204, 0.331, -0.052, ...]   ← far away
```

"Close together" is measured by **cosine similarity** — the angle between the
two arrows. 1.0 = same direction, 0 = unrelated, −1 = opposite.

### Why this is the foundation of search

Convert every document chunk to a point. Convert the question to a point. Return
the nearest chunks. That is semantic search, complete.

`01_embeddings/walkthrough.py` builds it from scratch in numpy and prints every
step: tokenizing, hashing words into slots, weighting rare words higher (IDF),
normalising, then ranking by cosine. **Run it.** Fifteen minutes there is worth
more than any amount of reading here.

### The failure that created neural embeddings

The simple version counts words. So "car" and "automobile" are *completely
unrelated* to it — different words, different slots. Neural embedding models are
trained specifically so synonyms and paraphrases land near each other.

The repo asserts this from both sides: a fast test proves the word-counting
version fails on `car`/`automobile`, and an `@ollama` test proves a real model
doesn't.

---

## 5. RAG, and its one dangerous property

**Retrieval-Augmented Generation.** The model doesn't know your data, and
retraining is impractical, so you *show* it the relevant text at question time.

```
                    ┌──── indexing (done once, ahead of time) ────┐
   documents  →  split into chunks  →  embed each  →  vector store
                                                            │
                    ┌──── serving (per question) ────────────┘
   question  →  embed  →  find nearest chunks  →  paste into prompt  →  answer
```

### The property that causes most AI incidents

**Search always returns results.** It has no concept of "no match". Ask about
something the corpus doesn't cover and it returns the *k least irrelevant*
chunks, with scores, with no error.

The model then receives four irrelevant passages and an instruction to answer.
Unless you explicitly permitted refusal, it writes something plausible.

Run `make hello` and look at step 7 — it demonstrates this on a real question
the corpus can't answer.

**What this means for you as a tester:** a test set of only answerable questions
cannot detect the single worst failure mode. This repo's golden dataset is 12%
deliberately unanswerable for exactly that reason. In your language: you have to
write the negative tests, and nobody else will.

---

## 6. Agents

An **agent** is a model that can call your functions.

```python
@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for passages relevant to a query."""
```

The model sees a JSON schema built from the **name**, the **type hints** and the
**docstring**. So the docstring is a *prompt*. If an agent keeps choosing the
wrong tool, rewrite the docstring before you touch the system prompt — almost
nobody tries that first.

The loop:

```
        ┌──────────────────────────────────────┐
        ↓                                      │
   ask the model  →  tool calls requested?  →  run them  →  feed results back
                            │ no
                            ↓
                          done
```

**There is no planner.** The exit condition is literally "the model stopped
asking for tools". Which is why agents fail in ways pipelines can't:

| Failure | Why outcome-testing misses it |
|---|---|
| Infinite loop | It never returns a wrong answer — it just never returns |
| Wrong tool | The final answer might still be right, by luck |
| Right tool, wrong arguments | Distinct from choosing wrong; needs its own check |
| Stops too early | Answer looks complete, isn't |

That's why agent evaluation scores the **trajectory**, not just the output. All
four are reproduced and tested in `03_langgraph` with no model at all.

---

## 7. Why testing this is genuinely harder — and where you're already ahead

| Problem | Why it's hard | What you already know |
|---|---|---|
| No exact expected value | "Good answer" isn't a string comparison | Fuzzy/property-based assertions |
| Non-determinism | Same input, different output | Flaky test triage: find and remove the variance source |
| The oracle is fallible | Your judge is another AI | Never trust an unverified test oracle |
| Slow and costly | Every assertion is a network call | Test pyramid: cheap tests first, expensive last |
| Infinite input space | Users type anything | Equivalence partitioning, boundary analysis |
| No reference in prod | Nobody labels live traffic | Monitoring vs testing — you know the difference |

Every entry in the right-hand column is a skill you already have. The people
building these systems frequently don't. **That is your interview advantage —
lead with it.**

---

## 8. The five things to remember

1. **Temperature 0 in tests.** Non-determinism in your measuring instrument is
   pure downside.
2. **Search always returns something.** Refusal must be built and measured.
3. **Faithfulness is not correctness.** Faithfully repeating a wrong document
   scores perfectly.
4. **A saturated metric cannot detect a regression.** Check whether your number
   can even move before you gate on it.
5. **Measure your judge before you trust its numbers.** Cohen's kappa, against
   labels you wrote by hand.

---

Next: **[glossary.md](glossary.md)**, then `make hello`, then
**[01_embeddings](../01_embeddings/)**.
