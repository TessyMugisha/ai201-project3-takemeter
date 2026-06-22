# TakeMeter

A fine-tuned text classifier that sorts TV/film discussion comments by **discourse type** — is a comment an argued judgment, a bare opinion, or an emotional reaction? Built on `distilbert-base-uncased`, fine-tuned on 203 hand-labeled comments from the Paradise (Hulu) Reddit community, and compared against a zero-shot Llama 3.3 70B baseline.

> **Note:** Sections marked _[FILL AFTER COLAB]_ contain placeholders to be replaced with real numbers after running the fine-tuning notebook. Everything else is final.

## What this project is

Online TV/film communities produce a wide range of comment quality on the same event: deep breakdowns of why the writing worked, one-line verdicts, and pure emotional outbursts all sit side by side in the same thread. TakeMeter classifies a comment into one of three discourse types so a community tool could, for example, surface high-effort critique or filter for live reactions.

The hard part of this project is not the training pipeline — it is the label design and annotation. Most of the effort went into defining a taxonomy whose boundaries hold up, then hand-labeling 203 real comments against those definitions.

## Labels

I use three mutually exclusive labels. I chose **critique** rather than "analysis" because TV/film discourse argues from reasoning and craft observation (pacing, character consistency, structure, plausibility), not from verifiable statistics the way a sports community would.

| Label | Definition | Example |
|---|---|---|
| **critique** | A judgment about the show backed by reasoning that does real work — a "because," a specific example, a craft observation, or a comparison that explains why. | *"The presidential assassination being undone by an allergy to tree nuts is horrifically lazy writing."* |
| **hot_take** | A bold or confident verdict stated without supporting reasoning. The claim may be true, but the comment asserts rather than argues. | *"This show kinda fucking sucks."* |
| **reaction** | An immediate emotional response to something in the show. The dominant register is feeling, not judgment. | *"I'm shaking after that episode, every atom in my body feels electric, I LOVE THIS SHOW."* |

**Decision rule (the hard boundary, hot_take vs. critique):** Remove the opinion from the comment. If real, load-bearing reasoning remains, it is critique. If nothing remains — or only a vague phrase that just restates the opinion — it is hot_take. The test is whether the reasoning carries weight, not whether a reason-shaped clause is present.

**Tiebreaker (mixed emotion + judgment):** Label by what the comment is built around. Emotion as the point with judgment as an aside → reaction. Reasoning as the point with emotion as set-dressing → critique/hot_take. When genuinely 50/50, default to the higher-effort label (critique > hot_take > reaction).

Full design reasoning, the stress-test that produced the tiebreaker, and the evaluation-metric rationale are in [`planning.md`](./planning.md).

## Data

**Source:** Public comments from Reddit threads in the Paradise (Hulu) community — premiere, episode/finale discussion, and opinion/debate threads. Comments were copied from public threads and hand-labeled.

**Collection method:** Manual collection into a CSV. I used an AI assistant as a labeling *partner* (see AI Usage) — it proposed a label with reasoning for each comment, and I reviewed and overrode its calls against the decision rules. Every final label was reviewed by me. The reasoning for each call is preserved in the `notes` column of [`takemeter_labeled.csv`](./takemeter_labeled.csv).

**Targeting by thread type:** The key collection lesson was that *thread type controls which label you get*. Finale and discussion threads overflow with critique; premiere and live threads produce reactions; "hate-watch" and "unpopular opinion" threads produce hot_takes. I targeted thread types deliberately to balance the set.

**Exclusions:** Plot speculation/theorizing, pure inside-jokes and quoted lines, and off-topic logistics chatter were not collected, because they do not fit the three-label taxonomy. Excluding rather than force-fitting kept each label meaningful.

**Label distribution (203 total):**

| Label | Count | Share |
|---|---|---|
| critique | 91 | 45% |
| reaction | 60 | 30% |
| hot_take | 52 | 26% |

No label exceeds the 70% imbalance threshold. critique is the largest class (it is the easiest discourse type to find), and hot_take is the smallest — a real finding, since most people who post strongly enough to comment include *some* justification, which tips them into critique.

**Split:** The notebook splits this single CSV into train/validation/test at 70% / 15% / 15% automatically.

### Three difficult examples and how I decided

1. **"not got solution for time travel mysteries, but other than that this season was awesome"** — Could read as reaction ("awesome" is emotive) or hot_take (a verdict on the season with a caveat). **Decided: hot_take**, because the comment is built around an evaluative claim about the season, with the caveat functioning as part of the verdict rather than an emotional outburst.

2. **"As a med student all of Annie's doctoring scenes were driving me crazy"** — The credential implies reasoning (medical inaccuracy) but none is actually given, and "driving me crazy" is emotive. Sits between reaction, hot_take, and critique. **Decided: hot_take** — it is a brief evaluative complaint with no stated reasoning; the credential gestures at a reason without supplying one.

3. **"Just saw the end of season 2 and am so upset that I won't watch season 3 and wish I never started the series. I feel like they jumped the shark and took the easy way out. So disappointed"** — Heavy emotion ("so upset," "so disappointed") but also evaluative claims ("jumped the shark," "easy way out"). **Decided: reaction**, applying the tiebreaker: the comment is built around emotional disappointment, and the evaluative phrases are bare assertions rather than reasoning, so emotion is the dominant register.

(Additional borderline cases, including a sarcasm-encodes-critique example, are flagged in the CSV `notes` column.)

## Model and training

- **Base model:** `distilbert-base-uncased` — a small, fast pre-trained model that understands English but has no notion of these three labels until fine-tuned.
- **Training:** Fine-tuned on the 70% train split using the starter Colab notebook on a free T4 GPU.
- **Hyperparameters:** _[FILL AFTER COLAB — state the values you used. Notebook defaults are 3 epochs, learning rate 2e-5, batch size 16. Document at least one decision and why, e.g., "kept 3 epochs because validation loss flattened after epoch 3 and more began to overfit on only ~140 training examples."]_

## Baseline

A zero-shot baseline using Groq's `llama-3.3-70b-versatile`, prompted with the three label definitions and instructed to output only the label name, classifying the same locked test set with no task-specific training. This measures how hard the task is for a strong general model and gives the fine-tuned numbers meaning.

## Evaluation report

_[FILL AFTER COLAB — all numbers below come from notebook Sections 4, 5, and 6.]_

### Overall accuracy

| Model | Test accuracy |
|---|---|
| Zero-shot Llama 3.3 70B (baseline) | _[FILL]_ |
| Fine-tuned DistilBERT | _[FILL]_ |

Majority-class baseline for reference: always guessing critique scores ~0.45 on this distribution, so a useful model must clear that comfortably.

### Per-class metrics

**Fine-tuned DistilBERT:**

| Label | Precision | Recall | F1 |
|---|---|---|---|
| critique | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| hot_take | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| reaction | _[FILL]_ | _[FILL]_ | _[FILL]_ |

**Zero-shot Llama (baseline):**

| Label | Precision | Recall | F1 |
|---|---|---|---|
| critique | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| hot_take | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| reaction | _[FILL]_ | _[FILL]_ | _[FILL]_ |

### Confusion matrix (fine-tuned model)

_[FILL AFTER COLAB — write the matrix out as text from the notebook output. Rows = true label, columns = predicted. The committed `confusion_matrix.png` is the supplementary image version.]_

| True ↓ / Pred → | critique | hot_take | reaction |
|---|---|---|---|
| **critique** | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| **hot_take** | _[FILL]_ | _[FILL]_ | _[FILL]_ |
| **reaction** | _[FILL]_ | _[FILL]_ | _[FILL]_ |

**Prediction to confirm or refute:** I expect hot_take to be the most-confused label, leaking into critique (model over-credits a thin reason) and into reaction (short punchy verdicts look like outbursts). _[FILL — state whether the matrix confirmed this.]_

### Three wrong predictions analyzed

_[FILL AFTER COLAB — pick 3 misclassified test examples from Section 4. For each, give: the comment, true label, predicted label, and a real analysis using the guiding questions below. Run your wrong-prediction list through an AI tool to surface patterns first, then verify them yourself.]_

1. **Comment:** _[FILL]_ — True: _[FILL]_, Predicted: _[FILL]_.
   **Why it failed:** _[FILL — which boundary? is it sarcasm, short length, a thin reason that looks load-bearing? Is it a labeling problem or a data-distribution problem?]_

2. _[FILL]_

3. _[FILL]_

### Sample classifications

_[FILL AFTER COLAB — run 3–5 example comments through the fine-tuned model and show predicted label + confidence. For at least one correct prediction, explain why it's reasonable.]_

| Comment | Predicted | Confidence | Note |
|---|---|---|---|
| _[FILL]_ | _[FILL]_ | _[FILL]_ | _[for one correct row, one sentence on why the prediction is reasonable]_ |

### What the model captured vs. what I intended

_[FILL AFTER COLAB — this is the key reflection. Likely themes to confirm against your results:]_

- I intended the model to learn the abstract distinction "argued judgment vs. bare verdict vs. emotional response." Because the dataset is **single-show (all Paradise)** and the critiques lean heavily on **finale-complaint vocabulary** ("rushed," "lazy writing," "plot holes," "deus ex machina"), the model may have partly learned *Paradise-specific and complaint-specific surface features* rather than the abstract categories. _[Confirm: do errors cluster on non-finale or non-Paradise-style phrasing?]_
- hot_take is the smallest class and the conceptual middle of the taxonomy, so it is the most likely to be under-learned. _[Confirm against per-class F1 and the matrix.]_

## Spec reflection

_[FILL — one way the spec helped, one way you diverged.]_ One genuine divergence already present: the example taxonomy in the spec used an "analysis" label implying verifiable evidence; I diverged to "critique" because TV/film discourse argues from reasoning and craft, not data — a deliberate adaptation to the community rather than a copy of the example.

## AI usage

I used an AI assistant in several disclosed ways:

1. **Label stress-testing (before annotation):** I gave the assistant my label definitions and decision rule and asked it to generate boundary-case comments designed to break the rule, then classified them myself. This confirmed my core definitions held but revealed I had no rule for mixed emotion+judgment comments; I added the tiebreaker rule as a direct result.
2. **Annotation assistance (disclosed):** I used the assistant as a labeling partner — it proposed a label and reasoning for each real comment, and I reviewed and overrode its calls against my decision rules. Every final label was reviewed by me; the per-comment reasoning is preserved in the CSV `notes` column.
3. **Failure analysis (planned):** _[FILL AFTER COLAB — after training, I gave my wrong-prediction list to the assistant to surface patterns, then verified each against the confusion matrix and examples before writing it up. State what it found and what you corrected or discarded.]_

## Repository contents

- `planning.md` — design thinking, label definitions, edge-case rules, evaluation-metric rationale, AI tool plan.
- `takemeter_labeled.csv` — 203 labeled comments (`text`, `label`, `source_url`, `notes`).
- `README.md` — this file.
- `evaluation_results.json` — _[committed after Colab]_ baseline-vs-fine-tuned metrics.
- `confusion_matrix.png` — _[committed after Colab]_ confusion matrix image.