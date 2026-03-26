# Measuring Progress Toward AGI — Cognitive Abilities

> Design high-quality benchmarks that go beyond recall to evaluate how frontier models truly reason, act, and judge.

**Host:** Google DeepMind × Kaggle  
**Total Prize Pool:** $200,000

---

## Description

Current AI models often succeed by exploiting familiar data or memorized patterns, making existing evaluations poor judges of how models truly think. This hackathon challenges you to bridge that gap.

Imagine a student who gets an A+ on a history test not because they understand the underlying events, but because they memorized the textbook. Current AI models can be similar: they display remarkable flashes of brilliance and crystallized knowledge, but often rely on surface-level patterns rather than fluid intelligence. This makes it difficult to distinguish when a model is truly solving a novel problem versus when it is simply recalling something it has seen during training.

The core problem is that we lack an empirical framework to measure these limitations. We need evaluations that isolate specific cognitive abilities, resist shortcut solutions, and expose systematic failure modes. Without such benchmarks, progress toward human-level generality becomes difficult to interpret, comparisons between models become noisy, and important weaknesses remain hidden until deployment.

Your task is to create high-quality benchmarks using [Kaggle Benchmarks](https://www.kaggle.com/discussions/product-announcements/667898) to test true understanding, focusing on the cognitive faculties highlighted in Google DeepMind's paper — [Measuring progress toward AGI: A cognitive framework](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/measuring-progress-toward-agi/measuring-progress-toward-agi-a-cognitive-framework.pdf).

The five cognitive tracks are: **Learning, Metacognition, Attention, Executive Functions, and Social Cognition.** Designing these rigorous standards will build detailed cognitive profiles of frontier models and reveal exactly how close we are to achieving Artificial General Intelligence (AGI).

A successful submission should answer a simple question: *"What can this benchmark tell us about model behavior that we could not see before?"*

---

## Timeline

| Date | Milestone |
|------|-----------|
| March 17, 2026 | Start Date |
| April 16, 2026 | Final Submission Deadline |
| April 17 – May 31, 2026 | Judging Period* |
| June 1, 2026 | Anticipated Results Announcement |

\* Judging period subject to change based on the number of submissions received. All deadlines are at 11:59 PM UTC.

---

## Submission Requirements

> **Note:** Upon joining this hackathon, your Kaggle account will be provisioned with extra quota ($50/day, $500/month) to run the AI models for your benchmark.

A valid submission must contain the following:

1. **Kaggle Writeup**
   - \[Mandatory\] Kaggle Benchmark, attached as a project link under "Benchmark"
   - \[Optional\] Media Gallery
   - \[Optional\] Attached Public Notebook

**Your final submission must be made prior to the deadline. Any un-submitted or draft Writeups will not be considered by the judges.**

To create a new Writeup, click "New Writeup" on the [Writeups page](https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups). After saving, a "Submit" button will appear in the top right corner.

> Note: If you attach a private Kaggle Resource to your public Writeup, your private Resource will automatically be made public after the deadline.

### a. Kaggle Benchmark \[mandatory\]

The most important part of your submission. You must create a Kaggle Benchmark with underlying tasks — all authored by you — and link the benchmark in the writeup as a project link. Keep tasks and the benchmark **private** until the deadline; after submission, all tasks and benchmarks are published publicly.

[Kaggle Benchmarks](https://www.kaggle.com/benchmarks) is a product that lets you build, run, and share custom benchmarks for evaluating AI models at no cost, powered by the [kaggle-benchmarks SDK](https://github.com/Kaggle/kaggle-benchmarks/tree/ci).

**Resources:**
- [Kaggle Benchmarks guide](https://www.kaggle.com/docs/benchmarks#intro)
- [Getting started notebook](https://www.kaggle.com/code/nicholaskanggoog/kaggle-benchmarks-getting-started-notebook?scriptVersionId=290215074)
- [YouTube tutorial](https://www.youtube.com/watch?v=VBlyJJ7PTD8)
- [Open source GitHub repo](https://github.com/Kaggle/kaggle-benchmarks) & [DeepWiki](https://deepwiki.com/Kaggle/kaggle-benchmarks)
- [Benchmarks cookbook](https://github.com/Kaggle/kaggle-benchmarks/blob/ci/cookbook.md)
- [Example tasks](https://github.com/Kaggle/kaggle-benchmarks/tree/ci/documentation/examples)

### b. Media Gallery \[optional\]

Attach any images and/or videos associated with your submission. A cover image is required to submit your Writeup.

### c. Public Notebook \[optional\]

Submit your code as a public notebook in the `Project Files` field. It must be publicly accessible. Private Kaggle Notebooks are automatically made public after the deadline.

### d. Public Project Link \[mandatory\]

A URL to your benchmark. Under "Attachments", click "Add a link" and select your benchmark to add it to the project.

---

## Evaluation

### Minimum Requirements

- Target one primary domain (to keep the signal sharp)
- Clearly state which capability is being isolated
- Explain what new insight the benchmark reveals about model behavior within that domain

### Scoring Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Dataset quality & task construction** | 50% | Verifiably correct answers (no ambiguity), sufficient sample size, clean and readable code, robust input prompts and output verification |
| **Writeup quality** | 20% | Clear problem statement, task & benchmark construction details, dataset provenance, technical implementation details, results and insights, organizational affiliations, references & citations |
| **Novelty, insights & discriminatory power** | 30% | What the benchmark reveals that existing ones cannot; meaningful signal with a gradient of performance (a benchmark where all models score 0% or 100% is equally useless) |

### Writeup Template

Use the following structure (3 pages or less):

```
### Project Name
### Your Team
### Problem Statement
### Task & benchmark construction
### Dataset
### Technical details
### Results, insights, and conclusions
### Organizational affiliations
### References & citations
```

---

## Prizes

### Grand Prizes

**Four (4) grand prizes of $25,000 each** are awarded to the best submissions across all tracks (no track restriction).

### Track Prizes

Each of the five tracks has **two prizes of $10,000 each** ($20,000 per track). No repeat winners between grand prizes and track prizes, for a total of **14 unique winners**.

---

## Tracks

### Learning · $20,000

**Can the model acquire and apply new knowledge and skills — not just recall what it was trained on?**

Learning is the ability to acquire new knowledge or skills through experience. It is fundamental to adaptive intelligence: a system that cannot learn from new experiences is inherently brittle. Current benchmarks test what models *know* (crystallized knowledge) rather than their capacity to learn on the fly. This track asks participants to create evaluations that isolate learning processes — including reinforcement-based learning, concept formation, and skill learning.

**Example evaluation targets:**
- Can the model learn a new rule or concept from a handful of examples and generalize it correctly?
- Does the model retain information provided earlier in a long interaction, or does it drift and hallucinate?
- Can the model update its beliefs when given corrective feedback, or does it perseverate on initial answers?

---

### Metacognition · $20,000

**Does the model know what it knows — and what it doesn't?**

Metacognition is a system's knowledge about its own cognitive processes and its ability to monitor and control them. It is often under-evaluated in AI: we rarely test whether models can accurately judge their own confidence, detect errors, or adjust strategies when failing. This track asks participants to build evaluations that probe metacognitive knowledge, monitoring, and control.

**Example evaluation targets:**
- Is the model's stated confidence well-calibrated with its actual accuracy?
- Can the model identify which questions it is likely to get wrong before answering?
- When the model makes an error, does it detect and correct it — or does it confabulate a justification?
- Does the model know the boundaries of its own knowledge (e.g., distinguishing "I know this" from "I'm guessing")?

---

### Attention · $20,000

**Can the model focus on what matters and ignore what doesn't?**

Attention is the ability to focus cognitive resources on specific aspects of information or task demands. While sharing a name with the transformer mechanism, cognitive attention specifically refers to how a system allocates processing resources across competing information. In AI, failures appear as distraction by irrelevant context or missing critical details. This track probes selective attention (filtering), sustained attention, and attention shifting (flexibility).

**Example evaluation targets:**
- Does the model get distracted by irrelevant but salient information inserted into a prompt?
- Does performance degrade systematically as input length increases, even when task difficulty is held constant?
- Can the model shift focus between sub-tasks in a complex, multi-part prompt without losing track?
- How does the model perform when critical information is buried among large amounts of irrelevant context?

---

### Executive Functions · $20,000

**Can the model plan, inhibit impulses, and adapt flexibly — or does it default to habitual responses?**

Executive functions include planning, inhibitory control, and cognitive flexibility. These are often conflated with "reasoning," but are distinct: a model may excel at logic yet struggle with multi-step plans or overriding habitual responses. This track asks for evaluations that isolate these processes to reveal a model's true ability to orchestrate complex thoughts and actions.

**Example evaluation targets:**
- Can the model formulate and execute a multi-step plan, adjusting when intermediate steps fail?
- When a habitual response pattern is wrong in a new context, can the model override it?
- Can the model switch between different task rules or frameworks without perseverative errors?
- How does the model handle situations where multiple plausible actions conflict?
- Can the model perform intermediate computations without losing track (working memory)?

---

### Social Cognition · $20,000

**Can the model understand and navigate social situations — not just produce polite text?**

Social cognition is the ability to interpret and respond to social information. For AI, it underpins inferring user intent, predicting reactions, and navigating competing perspectives. This track asks for evaluations that probe genuine social abilities beyond surface-level politeness.

**Example evaluation targets:**
- Can the model infer a speaker's intention when it diverges from their literal statement?
- Can the model track and reason about multiple agents with different (and possibly false) beliefs?
- Does the model adjust its communication style appropriately for different social contexts and audiences?
- Can the model navigate a negotiation scenario where goals are partially misaligned?
- Does the model recognize and respond appropriately to implicit social norms?

---

## Judges

- Yibin Lin — Software Engineer, Kaggle
- Prathamesh Bang
- Marc Coram
- Yao Yan
- Martyna Plomecka — Research Scientist, Google DeepMind
- Nicholas Kang — Product Manager, Kaggle
- Long Phan
- Lionel Levine
- Xin Liu
- Kiran Vodrahalli
- Isabelle
- Oran Kelly

---

## Citation

Martyna Plomecka, Yao Yan, Nicholas Kang, Ryan Burnell, María Cruz, and Sara Wolley. *Measuring Progress Toward AGI - Cognitive Abilities.* https://kaggle.com/competitions/kaggle-measuring-agi, 2026. Kaggle.