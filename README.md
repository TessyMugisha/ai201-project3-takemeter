# TakeMeter

A fine-tuned text classifier that sorts TV/film discussion comments by **discourse type** — is a comment an argued judgment, a bare opinion, or an emotional reaction? Built on `distilbert-base-uncased`, fine-tuned on 203 hand-labeled comments from the Paradise (Hulu) Reddit community, and compared against a zero-shot Llama 3.3 70B baseline.

## What this project is

Online TV/film communities produce a wide range of comment quality on the same event: deep breakdowns of why the writing worked, one-line verdicts, and pure emotional outbursts all sit side by side in the same thread. TakeMeter classifies a comment into one of three discourse types so a community tool could, for example, surface high-effort critique or filter for live reactions.

The hard part of this project is not the training pipeline. It is the label design and annotation. Most of the effort went into defining a taxonomy whose boundaries hold up, then hand-labeling 203 real comments against those definitions.

## Labels

I use three mutually exclusive labels. I chose **critique** rather than "analysis" because TV/film discourse argues from reasoning and craft observation (pacing, character consistency, structure, plausibility), not from verifiable statistics the way a sports community would.

| Label | Definition | Example |
|---|---|---|
| **critique** | A judgment about the show backed by reasoning that does real work — a "because," a specific example, a craft observation, or a comparison that explains why. | *"The presidential assassination being undone by an allergy to tree nuts is horrifically lazy writing."* |
| **hot_take** | A bold or confident verdict stated without supporting reasoning. The claim may be true, but it asserts rather than argues. | *"This show kinda fucking sucks."* |
| **reaction** | An immediate emotional response to something in the show. The dominant register is feeling, not judgment. | *"I'm shaking after that episode, every atom in my body feels electric, I LOVE THIS SHOW."* |

**Decision rule (the hard boundary, hot_take vs. critique):** Remove the opinion from the comment. If real, load-bearing reasoning remains, it is critique. If nothing remains — or only a vague phrase that just restates the opinion — it is hot_take.

**Tiebreaker (mixed emotion + judgment):** Label by what the comment is built around. Emotion as the point with judgment as an aside → reaction. Reasoning as the point with emotion as set-dressing → critique/hot_take.

Full design reasoning, the stress-test that produced the tiebreaker, and evaluation-metric rationale are in [`planning.md`](./planning.md).

## Data

**Source:** Public comments from Reddit threads in the Paradise (Hulu) community — premiere, episode/finale discussion, and opinion/debate threads. Comments were copied from public threads and hand-labeled.

**Collection method:** Manual collection into a CSV. I used an AI assistant as a labeling *partner* (see AI Usage) — it proposed a label with reasoning for each comment, and I reviewed and overrode its calls against the decision rules. Every final label was reviewed by me. The reasoning for each call is preserved in the `notes` column of [`takemeter_labeled.csv`](./takemeter_labeled.csv).

**Targeting by thread type:** Early collection skewed heavily toward critique, because finale and discussion threads are where people argue about quality. I corrected this by deliberately targeting different thread *types* for different labels: premiere and live-episode threads for reactions, and "hate-watch / unpopular opinion" threads for hot_takes. Thread type controls which label you harvest — that was the most important collection lesson.

**Exclusions:** Plot speculation, pure inside-jokes and quoted lines, and off-topic logistics chatter were not collected, because they do not fit the three-label taxonomy.

**Label distribution (203 total):**

| Label | Count | Share |
|---|---|---|
| critique | 91 | 45% |
| reaction | 60 | 30% |
| hot_take | 52 | 26% |

No label exceeds the 70% imbalance threshold. critique is the largest class; hot_take is the smallest — a real finding, since most people who post strongly enough to comment include *some* justification, which tips them into critique.

**Split:** The notebook splits this single CSV into train/validation/test at 70% / 15% / 15% automatically. Train = 142, validation = 30, test = 31. Splits are stratified to preserve label proportions.

### Three difficult examples and how I decided

1. **"not got solution for time travel mysteries, but other than that this season was awesome"** — Could read as reaction ("awesome" is emotive) or hot_take. **Decided: hot_take**, because the comment is built around an evaluative claim about the season; the caveat functions as part of the verdict rather than an emotional outburst.

2. **"As a med student all of Annie's doctoring scenes were driving me crazy"** — The credential implies reasoning but none is actually given. **Decided: hot_take** — a brief evaluative complaint with no stated reasoning; the credential gestures at a reason without supplying one.

3. **"Just saw the end of season 2 and am so upset that I won't watch season 3 and wish I never started the series. I feel like they jumped the shark and took the easy way out. So disappointed"** — Heavy emotion but also evaluative claims. **Decided: reaction**, applying the tiebreaker: the comment is built around emotional disappointment, and the evaluative phrases are bare assertions rather than reasoning, so emotion is the dominant register.

## Model and training

- **Base model:** `distilbert-base-uncased` — a small, fast pre-trained model that understands English but has no notion of these three labels until fine-tuned.
- **Training:** Fine-tuned on the 70% train split using the starter Colab notebook on a free T4 GPU. Training time was under 5 minutes.

### Hyperparameter decisions

I diverged from the notebook defaults on two values, and the divergence was forced by the model's behavior:

- `num_train_epochs` = **12** (default was 3)
- `learning_rate` = **5e-5** (default was 2e-5)
- Other defaults kept: batch size 16, weight decay 0.01, `load_best_model_at_end=True`.

**Why:** My first training run with the defaults **collapsed to the majority class** — the model predicted `critique` for every test example, scoring 45% (exactly the base rate). The training loss barely moved across 3 epochs (1.09 → 1.07, where random-chance loss for 3 classes is ln(3) ≈ 1.10), confirming the model never escaped its starting prior. With only 142 training examples, the default settings train too gently to escape majority-class collapse.

Re-running at lr 5e-5 for 12 epochs broke the collapse: validation accuracy climbed from 0.47 in epoch 1 to 0.87 in epoch 12, and training loss dropped to 0.03. `load_best_model_at_end=True` kept the best checkpoint.

## Baseline

A zero-shot baseline using Groq's `llama-3.3-70b-versatile`, prompted with the three label definitions, the decision rule, and the tiebreaker, instructed to output only the label name. It classified the same locked test set with no task-specific training.

## Evaluation report

### Overall accuracy

| Model | Test accuracy |
|---|---|
| Zero-shot Llama 3.3 70B (baseline) | **0.871** |
| Fine-tuned DistilBERT | **0.806** |
| Difference | **−0.065** |

Majority-class baseline (always guess critique): ~0.45.

**Headline finding:** the zero-shot 70B baseline **outperformed** the fine-tuned DistilBERT by ~6.5 points. This is a substantive, reportable result, not a failure to hide: it directly answers the project's "was fine-tuning worth it?" question. With only 142 training examples, fine-tuning a small specialized model was not enough to surpass a strong general model on this task.

### Per-class metrics

**Fine-tuned DistilBERT (test set, n=31):**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| critique | 1.00 | 0.93 | 0.96 | 14 |
| hot_take | 0.62 | 0.62 | 0.62 | 8 |
| reaction | 0.70 | 0.78 | 0.74 | 9 |
| **accuracy** | | | **0.81** | 31 |
| macro avg | 0.78 | 0.78 | 0.77 | 31 |

**Zero-shot Llama 3.3 70B (baseline, same test set):**

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| critique | 0.82 | 1.00 | 0.90 | 14 |
| hot_take | 1.00 | 0.75 | 0.86 | 8 |
| reaction | 1.00 | 0.89 | 0.94 | 9 |
| **accuracy** | | | **0.87** | 31 |
| macro avg | 0.94 | 0.88 | 0.90 | 31 |

### Confusion matrix (fine-tuned model)

Rows = true label, columns = predicted label.

| True ↓ / Pred → | critique | hot_take | reaction |
|---|---|---|---|
| **critique** (14) | 13 | 1 | 0 |
| **hot_take** (8) | 0 | 5 | 3 |
| **reaction** (9) | 0 | 2 | 7 |

The image version (`confusion_matrix.png`) is committed in this repo as a supplementary copy.

### What the matrix reveals

The error pattern is sharp and directional: **the fine-tuned model handles critique near-perfectly but confuses hot_take and reaction with each other.** All 6 errors are in the bottom-right 2×2 block. Critique never gets confused with reaction in either direction.

This is *different* from the baseline's error pattern. The baseline's errors clustered on **hot_take → critique** (it over-predicted critique because thin reasoning fooled it). The fine-tuned model has the opposite weakness: it **separates critique cleanly but cannot tell short hot_takes from short reactions apart.**

This matches my prior hypothesis only partially. I had predicted hot_take would be the weakest class (correct — F1 0.62, the lowest), but I had expected leakage into critique, not into reaction. The actual leakage direction is the real finding.

### Three wrong predictions analyzed

I chose three of the six errors that best illustrate the dominant failure pattern.

**Error 1 — hot_take → reaction (confidence 0.93)**
> *"James Marsden is a delight to watch. While I really like SKB, I find him monotone and wooden here."*

True: hot_take. Predicted: reaction.
**Why it failed:** The comment is two bare verdicts (one positive, one negative) with no supporting reasoning — a textbook hot_take by the decision rule. The model likely latched onto "delight to watch" / "really like" (emotionally-toned positive language) and mapped that to reaction. It mistook *evaluative warmth* for *emotional outburst*. The verdict structure that defines hot_take isn't captured; surface affect dominates the prediction.

**Error 2 — hot_take → reaction (confidence 0.90)**
> *"I don't know if I've ever watched a show that simultaneously has such good acting and such bad writing."*

True: hot_take. Predicted: reaction.
**Why it failed:** Same pattern. This is a confident contrasting verdict with no reasoning, but the rhetorical opener "I don't know if I've ever..." reads as personal exclamation. The model anchored on the personal/exclamatory voice rather than on the verdict's structure.

**Error 3 — reaction → hot_take (confidence 0.94)**
> *"so i need season three next week"*

True: reaction. Predicted: hot_take.
**Why it failed:** The reverse direction of the same boundary. This is a short emotional expression of impatience, but it reads grammatically like a declarative claim ("I need X"). The model has learned to associate short declarative comments with hot_take and missed the implicit emotional register.

### Pattern across all 6 errors

Verified by re-reading the misclassified examples: **5 of 6 errors are hot_take ↔ reaction confusions, and the errors are concentrated on short comments (under 25 words) where the difference between "bare verdict" and "bare feeling" lives almost entirely in tone, not structure.** The sixth error was a critique misclassified as hot_take, where the model under-credited a `(see breaking bad, the wire)` reasoning aside as decorative rather than load-bearing.

This is consistent with what 142 training examples can support: the model learned the **structural** signal that separates critique (longer, with reasoning vocabulary) from the two short classes, but it had too little data to learn the **tonal** signal that separates the two short classes from each other.

### Is this a labeling problem or a data problem?

I re-read these 6 examples against my decision rule. I would label each of them the same way again — they are consistent with the taxonomy, not annotation noise. The failure is therefore a *data-distribution* problem: the model needs more short comments and more examples that sit on the hot_take/reaction boundary specifically, so it can learn the tonal cue rather than relying on length and reasoning-word presence.

**What would fix it:** more balanced class counts (currently 91/60/52) with extra focus on short hot_takes and short reactions, plus borderline examples of each. The labels are sound; the data shape is the bottleneck.

### Sample classifications

Five test-set comments and the fine-tuned model's predictions:

| Comment | Predicted | Confidence | True | Note |
|---|---|---|---|---|
| *"The presidential assassination being undone by an allergy to tree nuts is horrifically lazy writing."* | critique | high | critique | ✓ **Reasonable** — strong craft observation ("lazy writing") with a specific evidentiary anchor. The model correctly reads this as load-bearing reasoning. |
| *"I'm giving season 2 room to play around. Many great shows have funny season 2s where they experiment (see breaking bad, the wire)..."* | hot_take | 0.57 | critique | ✗ The Breaking Bad/Wire reference is real load-bearing evidence, but the model under-credited it. Low confidence (0.57) reflects genuine uncertainty. |
| *"James Marsden is a delight to watch..."* | reaction | 0.93 | hot_take | ✗ See Error 1 — surface affect overrode verdict structure. |
| *"so i need season three next week"* | hot_take | 0.94 | reaction | ✗ See Error 3 — declarative grammar overrode emotional register. |
| *"Cool. Don't watch it bro"* | reaction | 0.85 | hot_take | ✗ Brief dismissive verdict labeled as emotional outburst — same hot_take↔reaction pattern. |

### What the model captured vs. what I intended

I intended the model to learn the abstract distinction "**argued judgment vs. bare verdict vs. emotional response**." Looking at the confusion pattern, here is what it actually appears to have learned:

- It **captured critique cleanly** — but probably along a partly *surface* axis. Critique examples are systematically longer and use a specific vocabulary cluster ("rushed," "plot holes," "lazy writing," "deus ex machina," "because," "writers"). When the model says "critique" it is right 100% of the time, but it may be detecting *finale-complaint phrasing on a single show*, not the abstract category.
- It **collapsed the hot_take/reaction distinction** — the part of the taxonomy that depends on *tone* rather than *vocabulary*. A short declarative ("show is overrated") and a short exclamation ("I cant!!") look similar in token features; the difference lives in register, which 52 hot_take training examples weren't enough to teach.
- It **did not learn the decision rule itself.** I designed the taxonomy around "remove the opinion — does load-bearing reasoning remain?" The model has no representation of that test; it works on surface features. Error #4 above (the Breaking Bad / The Wire critique misread as hot_take) is the clearest case — the model under-credited a real piece of evidence because it was a brief parenthetical rather than a written-out argument.

The gap is that my taxonomy is **definitional** (built around what reasoning *does*) while the model is **distributional** (it learns what reasoning *looks like* in this dataset). On a single show with concentrated finale-complaint vocabulary, those two boundaries diverge — and the divergence is exactly where the errors live.

### A note on test-set size

The test set is 31 examples (the notebook's 15% split of 203). With 8 hot_takes and 9 reactions in the test set, a single misclassification moves a per-class recall by ~12 percentage points. The per-class numbers are real but noisy, and I treat the pattern (hot_take ↔ reaction confusion) as more reliable than any individual percentage.

## Spec reflection

**How the spec helped:** the spec's insistence that I write the decision rule *before* collecting 200 examples, and that I name my hardest expected edge case in advance, was the single most valuable structural constraint. It forced me to formalize the "does the reasoning carry weight?" test before I had a stake in any particular call, which kept my labeling consistent across 203 comments.

**How my implementation diverged:** the spec's example taxonomy used "analysis" with verifiable evidence (stats, historical comparisons). I diverged to **critique** because TV/film discourse argues from reasoning and craft, not from cite-able data — a deliberate adaptation to the community I chose. I also diverged from the notebook's default hyperparameters (3 epochs at 2e-5) to 12 epochs at 5e-5 after the defaults caused majority-class collapse; the spec invites this kind of decision and explicitly asks for a documented hyperparameter change.

## AI usage

I used an AI assistant in four disclosed ways:

1. **Label stress-testing (before annotation):** I gave the assistant my label definitions and decision rule and asked it to generate boundary-case comments designed to break the rule, then classified them myself. This confirmed my core definitions held but revealed I had no rule for mixed emotion+judgment comments. I added the tiebreaker rule ("default to the higher-effort label when 50/50") as a direct result.

2. **Annotation assistance (disclosed):** I used the assistant as a labeling *partner*, not an autolabeler. I pasted batches of real thread comments; the assistant proposed a label with its reasoning for each; I reviewed and overrode its calls against my decision rule. Every final label was reviewed by me. The per-comment reasoning is preserved in the CSV `notes` column.

3. **Training-failure diagnosis:** When the first training run collapsed to majority class, I described the symptom (43% accuracy across all epochs, all predictions = critique) to the assistant. It diagnosed under-training (loss near ln(3), barely moving) and proposed bumping epochs and learning rate. I ran 12 epochs at 5e-5 on its recommendation. The fix worked. I verified the diagnosis myself by watching the training loss climb out of ln(3) in the rerun.

4. **Failure analysis (after evaluation):** I gave the assistant my list of 6 wrong predictions and asked it to surface patterns. It identified the hot_take ↔ reaction confusion concentrated on short comments. I verified this against the confusion matrix and the actual text of each misclassified example before writing it up. One proposed pattern (that the model was triggered by exclamation marks specifically) I discarded because the misclassified examples did not consistently contain them.

## Repository contents

- `planning.md` — design thinking, label definitions, edge-case rules, evaluation-metric rationale, AI tool plan.
- `takemeter_labeled.csv` — 203 labeled comments (`text`, `label`, `source_url`, `notes`).
- `README.md` — this file.
- `evaluation_results.json` — baseline-vs-fine-tuned metrics from the notebook.
- `confusion_matrix.png` — confusion matrix image.