# LLM as a Judge

## The idea and its central risk

Using a language model to score another model's output is called LLM-as-a-judge.
It is the only practical way to measure qualities like helpfulness, faithfulness
and tone at scale, because writing rules for them is impossible and human
labelling is slow.

The central risk is that the judge is itself a fallible model. A metric produced
by an uncalibrated judge is a number that looks rigorous and may measure nothing.
Treating judge scores as ground truth without ever checking them against human
labels is the most common serious mistake in applied evaluation.

## Known biases

**Position bias** is the tendency to prefer whichever response appears first
when comparing two candidates. The standard mitigation is to run every pairwise
comparison twice with the order swapped and keep only the verdicts that agree.

**Verbosity bias** is the tendency to rate longer answers higher regardless of
quality.

**Self-preference bias** is the tendency of a model to rate its own outputs more
favourably than other models' outputs. This is why using the same model as both
generator and judge inflates scores, and why the generator and judge should be
configured separately.

**Leniency bias** is the general tendency of judges to cluster scores at the top
of the scale, so that a five-point scale effectively becomes a two-point one.

**Formatting bias** is sensitivity to superficial presentation such as bullet
points or markdown headers, independent of content.

## Making judges more reliable

Score discrete categories rather than continuous numbers. A judge asked for a
score between 0 and 1 produces arbitrary precision it cannot justify; a judge
asked to choose between defined labels is far more consistent.

Provide an explicit rubric that defines what each score means, rather than a
vague instruction to rate quality.

Require the judge to give its reasoning before its score, not after. A verdict
produced first and justified afterwards is a rationalisation.

Decompose the judgement. Rather than asking "is this answer good", ask a series
of narrow, checkable questions. Faithfulness works this way: it extracts
individual claims and verifies each one separately, which is far more reliable
than a single holistic judgement.

Set temperature to zero. A judge with sampling noise makes your metric noisy,
which destroys your ability to detect small regressions.

## Calibration

Calibration means measuring how well your judge agrees with human labels. The
procedure is to label a sample of outputs by hand, run the judge on the same
sample, and compute an agreement statistic.

**Cohen's kappa** measures agreement between two raters, corrected for the
agreement that would occur by chance. It ranges from -1 to 1. A value of 0 means
the judge is no better than random guessing. Values above 0.6 are conventionally
described as substantial agreement, and values above 0.8 as almost perfect. A
judge scoring below about 0.4 should not be trusted to gate anything.

Raw percentage agreement is misleading on imbalanced data. If 90 percent of
answers are good, a judge that says "good" every time achieves 90 percent
agreement and a kappa of 0, correctly revealing that it has learned nothing.

## Small local judges

Open-weight models in the 7 to 8 billion parameter range can judge simple,
well-decomposed criteria acceptably but are unreliable on tasks requiring
structured output over several steps, such as claim-by-claim faithfulness. Two
failure modes dominate: malformed JSON that cannot be parsed, and inconsistent
verdicts across identical repeated calls.

A judge failure must never be silently recorded as a score of zero. That
converts an infrastructure problem into a fake quality regression and corrupts
the baseline you compare against. Failed judgements should be recorded as
failures and reported separately from scores.
