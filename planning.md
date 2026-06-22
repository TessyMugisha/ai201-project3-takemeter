# TakeMeter — Planning

A fine-tuned classifier that sorts TV/film discussion comments by *discourse type*: is a given comment an argued judgment, a bare opinion, or an emotional reaction?

## Community

I chose the Paradise (Hulu) viewing community on Reddit — the episode-discussion, finale, and general-reaction threads where people talk about the show. Paradise is a mystery/sci-fi drama with a passionate fanbase, which makes it a strong fit for a discourse-quality task: the same event (a finale, a character death, a plot twist) produces a wide spread of comment types in the same thread. Some people write multi-paragraph breakdowns of why the writing worked or failed; some drop a one-line verdict; some just post how they felt in the moment. That natural variety — substantive argument sitting right next to gut reaction sitting next to bare opinion — is exactly the range a classifier needs to learn a meaningful boundary, rather than a community where every comment looks the same.

One deliberate scoping decision: mystery/sci-fi threads are also flooded with *plot speculation* ("here's my theory about the time travel," "is Jane secretly Alex?"). Speculation is a fourth mode of discourse that does not fit a quality/opinion taxonomy, so I excluded it during collection rather than forcing it into a label or adding a catch-all bucket. This is documented in the data collection plan below.

## Labels

I use three labels. I chose **critique** rather than the "analysis" label from the project examples, because film/TV discourse rarely argues from verifiable statistics the way a sports community does. A good TV argument is built from *reasoning and craft observation* — pacing, character consistency, structure, plausibility — not from cite-able data. "Critique" captures "argued judgment from reasoning," which is the real high-effort discourse mode in this community.

**critique** — a judgment about the show backed by reasoning that does actual work: a stated "because," a specific example, a craft observation, or a comparison that explains *why*. If you removed the opinion, real supporting reasoning would remain.
- *"The finale fell flat because they rushed three seasons of setup into 40 minutes and never paid off the brother's arc."*
- *"The presidential assassination being undone by an allergy to tree nuts is horrifically lazy writing."*

**hot_take** — a bold or confident verdict about the show stated without supporting reasoning. The claim may be true, but the comment asserts rather than argues. If you removed the opinion, nothing (or only more opinion) would remain.
- *"This show kinda fucking sucks."*
- *"Season 2 is better than Season 1 and it's not close."*

**reaction** — an immediate emotional response to something that happened in the show. The dominant register is feeling, not judgment.
- *"I'm shaking after that episode, every atom in my body feels electric, I LOVE THIS SHOW."*
- *"NOOO not him, I bawled my eyes out."*

## Hard edge cases

The boundary that genuinely fights back is **hot_take vs. critique**, with a secondary problem of **emotion-plus-judgment** mixed comments. Both came up repeatedly during annotation. My decision rules:

**Decision rule (hot_take vs. critique):** Remove the opinion from the comment. If real, load-bearing reasoning remains — a because, a specific example, a craft observation, a comparison that explains why — it is **critique**. If nothing remains, or only a vague reason-shaped phrase that merely restates the opinion ("I like it because it's good"), it is **hot_take**. The test is whether the reasoning *carries weight*, not whether a reason-shaped clause is present.

- *"I love season 2, it's getting us the background of how it came to be"* → **hot_take.** The trailing clause looks like a reason but just restates the opinion. Decorative.
- *"I'm giving season 2 room to play around. Many great shows have funny season 2s where they experiment (see Breaking Bad, The Wire)"* → **critique.** The two examples are real evidence; the argument survives without the opinion.

**Tiebreaker rule (mixed emotion + judgment):** When a comment contains both an emotional reaction and an evaluative claim, label it by which one the comment is *built around*. If the reasoning is the point and the emotion is set-dressing → critique/hot_take. If the emotion is the point and the judgment is a passing aside → reaction. When genuinely 50/50, default to the higher-effort label (critique > hot_take > reaction), because the comment is contributing argument to the discourse.

**Special pattern — sarcasm encoding a critique:** A throwaway-looking line can carry a real argument. *"Glad that knee is better, Xavier"* is sarcasm pointing at a forgotten injury (a continuity error). When the sarcasm *is* the argument, label it critique, not reaction.

These rules were stress-tested before annotation (see AI Tool Plan). The two cases I will treat as documented difficult examples in the README are: a strong-feeling-but-no-reasoning comment that *looks* like critique but is hot_take ("the finale was bad, like really bad, this one stings different"), and a mixed comment that opens as reaction but pivots into a real craft objection ("I gasped when Sinatra got shot, but then — she's obviously going to live, so why bother with the fake-out").

## Data collection plan

**Source:** Public comments from Reddit threads in the Paradise community — premiere threads, episode/finale discussion threads, and opinion/debate threads. Comments were copied from the threads and hand-labeled.

**Targeting by thread type:** Early collection skewed heavily toward critique, because finale and discussion threads are where people argue about quality. I corrected this by deliberately targeting different thread *types* for different labels: premiere and live-episode threads for reactions (people responding emotionally in the moment), and "is anyone else not into this / hate-watch / unpopular opinion" threads for hot_takes (bare verdicts). This is the single most important collection lesson — thread type controls which label you harvest.

**Exclusions:** Plot speculation/theorizing, pure inside-jokes and quoted lines, and off-topic logistics chatter (e.g., which network airs the show) were not collected, because they do not fit any of the three labels. Excluding rather than force-fitting keeps each label meaningful.

**Final counts (203 total):** critique 91, reaction 60, hot_take 52.

**Underrepresentation plan (executed):** hot_take was persistently the smallest and hardest label to find, because most people who post strongly enough to comment include *some* justification, which tips them into critique. When a label ran low I went back for thread types that produce it, capping critique once it was plentiful and continuing to collect reaction and hot_take until all three were healthy.

## Evaluation metrics

Accuracy alone is not sufficient here because the classes are imbalanced (91 / 60 / 52) and the cost of confusion is uneven across labels. I will report:

- **Overall accuracy** — headline number, for both the fine-tuned model and the zero-shot baseline.
- **Per-class precision, recall, and F1** — because I specifically expect the model to perform unevenly. With critique over-represented, a lazy model can score decently by over-predicting critique; per-class F1 exposes that, especially for the smaller hot_take class.
- **Confusion matrix** — to see *which* labels get swapped. My prediction: hot_take is the most-confused label, bleeding into both critique (when the model over-credits a thin reason) and reaction (when a verdict is short and punchy). The matrix will confirm or refute this directly.

The metric I will watch most closely is **hot_take F1 (and where its errors land in the matrix)**, since hot_take is both the smallest class and the conceptual middle of the taxonomy.

## Definition of success

For a real community tool (e.g., auto-tagging or surfacing high-effort comments), "good enough" would be:

- Overall test accuracy meaningfully above the majority-class baseline. With critique at ~45% of the data, always-guess-critique scores ~0.45, so a useful model must clear that comfortably — target **≥ 0.70 overall**.
- **No label collapses:** every class achieves **F1 ≥ 0.55**, so the tool is actually usable for all three types rather than just the majority one. A model that nails critique and reaction but scores near zero on hot_take would not be deployable, even with high overall accuracy.
- The fine-tuned model should at least **match or beat the zero-shot Llama baseline** on overall accuracy; if it does not, that is itself a reportable finding about whether fine-tuning on 200 examples was worth it for this task.

These criteria are specific enough to judge objectively at the end: I can read overall accuracy off the report, check that all three per-class F1s clear 0.55, and compare the two models head-to-head on the same test set.

## AI Tool Plan

There is no implementation code to generate in this project, so AI assistance is concentrated in the three places it actually helps:

**Label stress-testing (done before annotation).** I gave an AI assistant my label definitions and the hot_take-vs-critique decision rule and had it generate boundary-case comments designed to break the rule, then classified them myself. The exercise confirmed my three core definitions were sound but revealed I had no rule for comments mixing emotion and light judgment. I added the tiebreaker rule (default to the higher-effort label when 50/50) as a direct result. Two of the stress-test cases became my documented difficult examples.

**Annotation assistance (disclosed).** I used an AI assistant as a *labeling partner*, not an autolabeler: I pasted batches of real thread comments, the assistant proposed a label with its reasoning for each, and I reviewed and overrode calls I disagreed with. Every label in the final dataset was reviewed by me against the decision rules above. The reasoning behind each call is preserved in the `notes` column of the dataset CSV, and the genuinely borderline calls are flagged there.

**Failure analysis (planned).** After training, I will give the AI assistant my list of wrong predictions and ask it to surface patterns (e.g., "misclassified hot_takes tend to contain the word 'because'"), then verify each proposed pattern against the actual confusion matrix and example comments before writing it into the evaluation report. I will only report patterns I can confirm myself.