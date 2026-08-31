---
name: llm-inference-course-authoring
description: Create or revise Chinese LLM inference course chapters for software engineers who know Python but have little algorithm or Transformer background. Use when designing a course, writing or revising lesson content, adding diagrams, Colab labs, quizzes, or references in this repository.
paths:
  - docs/**
  - notebooks/**
---

# LLM Inference Course Authoring

Write a progressive, practical course about LLM inference for working programmers.

## Audience and learning contract

Assume the learner:

- can read and write Python;
- understands functions, classes, arrays, HTTP APIs, and basic command-line work;
- may not know linear algebra, deep learning, Transformer architecture, GPU hardware, or algorithm notation;
- wants to understand and operate LLM inference systems rather than train a foundation model.

Do not assume the learner has read papers. Define a new term in plain Chinese on first use. Use an English term in parentheses only when it helps them search documentation.

The desired outcome for each chapter is:

1. the learner can explain the idea in ordinary language;
2. the learner can identify where that idea appears in a real system such as Hugging Face, vLLM, SGLang, or llama.cpp;
3. when the chapter includes a code lab, the learner can run a small experiment.

Read [the chapter template](references/chapter-template.md) and [the pedagogy rubric](references/pedagogy-rubric.md) before authoring a chapter.

## Required workflow

### 1. Establish scope

Before drafting:

1. Read the target chapter and its adjacent chapters.
2. State the one learner question this chapter answers.
3. State prerequisites in programmer language, not academic course names.
4. Identify what is deliberately deferred to later chapters.
5. Check whether the user asked for a course outline only (do not edit files), or for course content/site changes.

Keep each chapter focused on one core idea. Split a chapter when its learner question cannot be stated in one sentence.

### 2. Research before claims

For technical claims that may change, research and cross-check:

- primary paper or official technical report for the algorithm;
- official documentation for framework commands and APIs;
- source repository or release documentation for implementation behavior.

Use secondary blogs only to clarify intuition, never as the sole source for performance claims. Link every practical command to an official source where possible.

Qualify benchmarks with model, hardware, precision, batch/concurrency, context length, and metric. Do not present a reported speedup as a general fact.

### 3. Draft in the teaching sequence

Default order:

1. **Why this matters** — a concrete developer-facing problem.
2. **Mental model** — a short explanation or analogy.
3. **Visual** — a diagram that shows data/control flow.
4. **Walkthrough** — explain the moving parts in execution order.
5. **Hands-on lab** — only when running code materially improves understanding; make it a small runnable Colab exercise.
6. **Production connection** — where the mechanism appears in real tooling.
7. **Common mistakes** — 2–4 likely misconceptions.
8. **Recap and self-check** — what to explain without looking.
9. **References** — curated, annotated sources.

Do not lead with definitions, equations, or a wall of code.

The chapter skeleton is fixed: concept explanation → visual → optional code lab → self-check → references. Do not remove the visual, self-check, or references sections. Omit the code lab and Colab link together when the chapter is conceptual or reading-oriented.

### 4. Use formulas sparingly

Prefer:

- a picture, table, concrete number, or worked example;
- executable Python that prints the result;
- prose such as “bandwidth divided by bytes read per step.”

Use a formula only if omitting it would make the mechanism misleading or prevent the learner from estimating a real system.

When a formula is necessary:

1. introduce the practical question it answers;
2. give a plain-language reading before the formula;
3. put each variable in a table: symbol, name, units, plain meaning, realistic example;
4. show one numerical substitution line by line;
5. state what the formula ignores and when it becomes unreliable;
6. follow it with an interactive calculator or a tiny Python cell when feasible.

Never introduce more than one new formula in a section. Avoid derivations in the main path; place optional derivations in a clearly labelled advanced callout.

### 5. Design code labs

Only a chapter with a code-focused lab must have a matching notebook at `notebooks/<chapter>_<topic>.ipynb`. Put its visible Colab link immediately before the relevant code snippet or numbered experiment step, never in the chapter introduction. Do not create empty or token “example” notebooks for a conceptual chapter.

Labs must:

- run on free Colab CPU by default unless GPU is central to the lesson;
- avoid downloading a multi-GB model for a first exercise;
- set seeds where output comparison matters;
- use small, inspectable dimensions;
- print or visualize the state that teaches the idea;
- explain the purpose of each code cell immediately before it;
- end with 2–3 small modifications the learner can try.

For an optional GPU exercise, label GPU requirements and give a non-GPU alternative.

Do not give a code block without explaining: what it receives, what it returns, and which few lines matter.

### 6. Design diagrams

Use an SVG/Vue component for dynamic data-flow diagrams in this VitePress site. A diagram must answer a question that prose alone makes harder:

- where data moves;
- what is cached, copied, or recomputed;
- which steps are repeated;
- which resource is the bottleneck.

Label arrows and phases in Chinese. Keep visual reading order left-to-right or top-to-bottom. Use no more than 7–9 nodes in a first-exposure diagram. On mobile, intentional horizontal scrolling is acceptable; page-wide overflow is not.

Diagrams are explanatory, not decorative. If a diagram does not teach a relationship, omit it.

### 7. Write quizzes and completion checks

Use 3–5 questions per chapter:

- at least one “predict the result” question;
- at least one misconception check;
- no trivia about paper authors or dates.

Completion criteria must test an observable ability: running a notebook, explaining a diagram, calculating a rough capacity, or changing a parameter and predicting its consequence.

### 8. Verify before handoff

For site changes:

1. validate every notebook as JSON;
2. run `npm run build`;
3. run relevant browser checks on desktop and mobile;
4. verify Colab URLs match committed notebook paths and the configured GitHub repository;
5. commit logical changes and push to the configured remote only when credentials permit.

Do not claim GitHub Pages is updated unless its deployment is verified.

## Repository conventions

- Site pages: `docs/chapters/`.
- Runnable labs: `notebooks/`.
- VitePress global components: `docs/.vitepress/theme/components/`.
- Course-wide references: `docs/resources.md`.
- Chinese is the primary language. Keep established English technical names searchable.
- Use existing VitePress Vue components (`Quiz`, `Checklist`, diagrams, calculators) when suitable rather than hand-rolling duplicate widgets.

## Quality gate

Before submitting a chapter, check all statements:

- Could a Python programmer with no Transformer background explain the opening problem?
- Does the first screen give intuition before jargon?
- Does every acronym have a first-use expansion?
- Does every formula meet the six-step explanation rule?
- If there is a code lab: can it run in a free Colab session?
- If there is code: does it have cell-level explanation and an expected result?
- Does each diagram have readable labels and a clear lesson?
- Are claims traceable to official docs, papers, or source?
- Does the chapter clearly say what it does **not** cover yet?

If any answer is no, revise before calling the chapter complete.
