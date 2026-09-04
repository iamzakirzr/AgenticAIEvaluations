# Agent Evaluation

## Why agents need different metrics

A RAG pipeline has one step, so evaluating its single output is sufficient. An
agent takes a variable number of steps, chooses tools, and can loop. The final
answer being correct does not mean the agent behaved well: it may have called an
expensive tool nine times, or reached the right answer by luck after a wrong
turn. Agent evaluation therefore scores the trajectory, not only the outcome.

## Trajectory metrics

**Tool Correctness** compares the tools the agent actually called against the
tools it should have called for that task. It can be scored strictly, requiring
exact order and arguments, or loosely, requiring only that the correct set of
tools was used. DeepEval implements this as a non-LLM metric when the expected
tool list is supplied, which makes it deterministic and cheap.

**Tool Call Accuracy** in RAGAS compares the sequence of tool invocations,
including their arguments, against a reference sequence.

**Task Completion** judges whether the agent achieved the user's underlying
goal, inferred from the full trace of messages and tool calls rather than from
the final message alone.

**Agent Goal Accuracy** in RAGAS comes in two forms. The with-reference variant
compares the end state against an annotated desired outcome. The without-
reference variant asks a model to infer the goal from the conversation and judge
whether it was met, which is useful when you have traces but no labels.

**Step Efficiency** penalises unnecessary steps: an agent that reaches the goal
in three tool calls is better than one that takes eleven, all else being equal.

**Loop Detection** identifies agents stuck repeating the same action. This is
one of the most common production agent failures and is invisible to
outcome-only evaluation, because the agent usually eventually times out rather
than returning a wrong answer.

**Plan Adherence** and **Plan Quality** apply to agents that produce an explicit
plan before acting, checking respectively whether the agent followed its plan
and whether the plan was sound in the first place.

**Argument Correctness** checks whether the parameters passed to a tool were
right, which is a distinct failure from calling the wrong tool entirely.

## Conversational metrics

For multi-turn chatbots:

**Knowledge Retention** checks whether the assistant remembers facts the user
stated earlier in the conversation and does not ask for them again.

**Role Adherence** checks whether the assistant stayed within its assigned
persona and scope across every turn.

**Conversation Completeness** checks whether the user's requests across the
whole conversation were ultimately satisfied.

**Topic Adherence** in RAGAS measures whether the assistant stayed within a set
of permitted subject areas, which is the standard way to evaluate a scoped
support bot that must refuse off-topic requests.

## Safety and misuse metrics

**Bias** and **Toxicity** score generated text for prejudiced or harmful
content.

**PII Leakage** checks whether the output exposes personally identifiable
information.

**Misuse** checks whether the assistant allowed itself to be used for something
outside its intended purpose.

**Non-Advice** checks that the assistant did not give regulated professional
advice, such as financial or medical guidance, when it is not permitted to.

**Role Violation** checks for breaks in character, including responses to
prompt-injection attempts embedded in retrieved documents.

## The core principle

Outcome metrics tell you whether the agent worked. Trajectory metrics tell you
whether it worked for the right reasons. Only the second kind lets you predict
whether it will keep working on inputs you have not tested.
