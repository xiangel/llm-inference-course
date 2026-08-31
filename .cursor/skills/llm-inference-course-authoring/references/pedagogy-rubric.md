# Pedagogy rubric for programmer beginners

## Writing level

Write for an engineer who is comfortable with:

- functions, classes, loops, dict/list/array concepts;
- Python package installation and notebooks;
- APIs, processes, logs, and basic performance measurements.

Do not assume familiarity with:

- vectors, matrices, dot products, softmax, derivatives;
- neural-network training;
- GPU architecture and CUDA;
- Transformer vocabulary.

If an unfamiliar concept is unavoidable:

1. describe its job first;
2. show it in the diagram or code;
3. name it second;
4. give the formal detail only when it unlocks the next step.

## Good and weak openings

**Good**

> You send a 4,000-token prompt to a model. The first word arrives after a pause, then the rest stream steadily. This chapter explains why those two waits have different causes.

**Weak**

> In this chapter, we formally derive the autoregressive factorization of a Transformer language model.

## Formula rubric

**Good**

> Question: will this model fit on a 24 GB GPU at an 8K context?  
> Plain reading: the cache grows once for every stored token, every layer, and every concurrent request.  
> Formula: [one formula].  
> Variables: [table].  
> Worked example: [numbers].  
> Boundary: weights and framework overhead are not included.

**Weak**

> [Formula]  
> Therefore the complexity is O(...).

## Code rubric

Each code cell must answer one question:

| Cell | Required explanation |
| --- | --- |
| Setup | What packages/data/dimensions are intentionally small and why |
| Core mechanism | Input, output, 1–3 lines to inspect |
| Observation | What output proves the concept |
| Exercise | A change the learner can make and what should change |

Prefer a 30-line, inspectable program to a production framework call that hides the mechanism.

## Diagram rubric

Before including a diagram, name its learning goal:

| Diagram | Learner should be able to say |
| --- | --- |
| End-to-end model flow | “Text becomes IDs, then vectors, then the model picks one next ID.” |
| Attention flow | “This token can use earlier tokens but cannot read later tokens.” |
| KV Cache flow | “Old K/V are reused; only the new token gets fresh K/V.” |
| Roofline | “This workload waits for memory or compute depending on which side of the ridge it sits.” |
| Scheduler flow | “Finished requests leave, new ones enter between decode steps.” |

## Source hierarchy

1. Official docs, source code, release notes.
2. Original paper / technical report.
3. Maintainer technical blog.
4. Educational blog or video, used only for intuition.

Never use a secondary source as the only evidence for a framework default or performance claim.
