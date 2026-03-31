## Module: Natural	Language	Processing	

### Weight: 50%	of	overall	module	mark

### Submission	Deadline: [10	April	 2026 ]

## 1.	Context	and	Motivation

```
Large Language Models (LLMs) often demonstrate significant gaps in culture-specific
everyday knowledge, particularly for under-resourced regions and languages. Many existing
benchmarks rely heavily on Western-centric data (e.g., Wikipedia), failing to capture everyday
realities such as local foods, school traditions, work practices, or holidays.
```
```
This project is inspired by BLEnD (Benchmark for LLMs on Everyday Knowledge in Diverse
Cultures and Languages). You will build and evaluate an NLP system that demonstrates:
```
- Technical proficiency in multilingual inference
- Careful evaluation and reproducibility
- Critical awareness of cultural bias and responsible AI

## 2.	Project	Structure

```
You must complete one track in depth.
Completing both tracks may be considered for distinction-level performance.
```
### Track	A	— Short	Answer	Questions	(SAQ)

```
Input: Question + locale (e.g., ga-IE, ar-DZ, zh-CN)
Output: Short answer in the same local language
```
```
Evaluation:
An answer is correct if it matches any human-annotated reference after appropriate
normalisation.
```
# Natural Language Processing –

# NLP Project 1 (50%) — Cross-Cultural Knowledge

# Evaluation (BLEnD / SemEval-Style)


You must:

- Implement a robust answer normalisation strategy (Unicode normalisation, token
    normalisation, stemming/lemmatisation if appropriate, or embedding-based
    similarity).
- Justify your chosen matching method.

### Track	B	— Multiple	Choice	Questions	(MCQ)

**Input:** Question (English) + target locale + 4 culturally distinct options
**Output:** Selected option (A/B/C/D or 0–3)

**Evaluation:**
Accuracy per locale and overall.

You must:

- Enforce structured output (model must return only the selected option).
- Analyse performance differences across locales.

## 3.	Minimum	Requirements

You must evaluate on **at least four locales** , including:

- At least **one non-Latin script locale** (e.g., Arabic, Chinese, Japanese, Thai)
- At least **one under-represented or low-resource locale**
- Two additional locales of your choice

## 4.	System	Requirements

You must implement a **local or cloud-based inference pipeline** using open-source/open-
weight models.

Model	Constraints

- Commercial APIs (e.g., GPT-4, Claude) are **not permitted**.
- Recommended model size: **7B–13B class**.
- Larger models may be used but are not required.
- All decoding must use **temperature = 0** for reproducibility.


## 5.	Required	Project	Components

### A.	Baseline	(Required)

Implement a clear baseline system.
Examples:

- Direct prompting baseline
- Simple multilingual transformer inference
- Basic retrieval-augmented approach
- Structured constrained decoding

### B.	Improvements	Beyond	Baseline	(Required)

You must implement **at least two meaningful improvements** , such as:

- Persona-based prompting
- Structured JSON outputs
- Retrieval augmentation (RAG)
- Confidence estimation or abstention mechanism
- Locale-aware prompting
- Calibration strategy
- Error-driven prompt refinement

You must demonstrate whether these improvements help (or explain why not).

### C.	Evaluation	&	Analysis	(Required)

Your evaluation must include:

- Overall performance
- Performance by locale (table required)
- Gap vs best-performing locale
- At least **10 analysed failure examples**
- At least one case where the model produces a fluent but culturally incorrect answer

## 6.	Responsible	AI	Section	(Required)

Your report must include a dedicated section discussing:

- Cultural bias and uneven performance
- Risks of deployment in under-resourced contexts
- Overconfidence and hallucination risks
- Trust calibration (uncertainty, abstention, guardrails)
- Transparency: what users should be told


Concrete mitigation strategies must be proposed.

## 7.	Reproducibility	Requirements

You must ensure:

- Fixed random seeds
- Temperature = 0 decoding
- Clear model version documentation
- Exact locale list documented
- Clear data preprocessing steps

Work must be reproducible from your README instructions.

## 8.	Deliverables

Submit a single ZIP file containing:

### 1)	Report	(PDF,	6–10	pages)

Must include:

1. Introduction & task description
2. Dataset and locale selection
3. Methodology
    o Baseline
    o Improvements
4. Evaluation
    o Tables (overall + per-locale)
5. Error analysis (≥10 examples)
6. Responsible AI reflection
7. Limitations & future work

Formatting:

- LREC-style formatting recommended (https://lrec2026.info/authors-kit/)
- LaTeX encouraged but not mandatory (clear academic formatting required)

### 2)	Code

- Clean, readable Python code
- Logical structure (functions, modularity)
- requirements.txt or environment.yml


- README.md with step-by-step reproduction instructions

### 3)	Prediction	Files

- predictions_saq.csv (if applicable)
- predictions_mcq.csv (if applicable)

## 9.	Marking	Rubric	(50%	of	the NLP	Module)

```
Component Weight
Reproducibility & Engineering Quality 20%
Baseline Implementation 10%
Improvements Beyond Baseline 20%
Evaluation & Per-Locale Analysis 25%
Error Analysis Depth 15%
Responsible AI Reflection 10%
```
### Distinction-Level	Indicators

- Completing both tracks
- Confidence calibration or abstention modelling
- Statistical comparison across locales
- Clear identification of systemic cultural bias patterns
- Particularly strong Responsible AI reasoning

## 10.	Academic	Integrity

- All code must be your own.
- Any AI assistance must be disclosed.
- Reports will be checked for plagiarism.

## 11.	Late	Submission	Policy

Standard MTU late submission policy applies:

- Up to 2 weeks late with 10% penalty.
- Medical certificates must be submitted to the department secretary if applicable.


## 12.	Final	Submission	Checklist

Before uploading to Canvas:

☐ Report PDF included

☐ Code included

☐ requirements.txt included

☐ README included

☐ Predictions file included

☐ Locales clearly listed

☐ Responsible AI section included


