# Routing eval — load precision, forbidden load, and the neighbouring skills

The `description` is the only part of this Skill always in context, and it is a router trigger rather than a summary
of the recipe. This file is the test set for it: run the queries **verbatim**, record whether the Skill loaded, and
score against the expectations below. A description change ships only with a re-run of this file.

**Debt, recorded rather than hidden.** The description was tightened on 2026-10-12 from 74 words to 49: the removed
sentence ("User veto, cancellation, and scope constraints win") was a behavioural rule, not a trigger, and the body
already carries it verbatim. Removing a non-routing sentence cannot add a false load, but it can remove a match, and
by this file's own rule the change wants the run below. No run exists yet.

## How to run

- Each case × 3 repetitions × **every orchestrator model family in use** — a router that only works on one family is
  not shipped.
- Record `loaded / not loaded`, the artefact produced, and the first sentence that shows which way it went.
- Ship only if every positive loads and no forbidden case loads. A neighbouring skill losing recall is a failure of
  this file, not of that skill.

## Positives — the Skill must load

| # | Query | Why it is in the set |
|---|---|---|
| C1 | "We're thinking of cutting the enterprise tier price 15%. Should we?" | A pricing judgment with real stakes; the shape the description names first |
| C2 | "Second opinion on this acquisition before we sign." | The independent-review trigger, in the words a user uses |
| C3 | "Red-team our compliance position on this data-processing clause." | The red-team trigger plus a stake the description names |
| C4 | "Our two advisors disagree on the contract terms — which is right?" | Sources in conflict, which is the reconciling trigger |
| C5 | "I need a go/no-go on this strategy by Friday." | The deadline shape; tests that "go/no-go" alone carries |

## Forbidden loads — the Skill must not load

| # | Query | Whose job it is instead |
|---|---|---|
| F1 | "What did we charge the enterprise tier last year?" | A lookup; the description's own negative |
| F2 | "Summarise this 40-page contract." | A summary, not a judgment |
| F3 | "Write the pricing page copy for the new tier." | Production, not judgment |
| F4 | "What is the current USD/SEK rate?" | A factual question |
| F5 | "Write up the go/no-go decision we made as a Word memo." | `office-documents` — the decision is made; the job is the file |
| F6 | "Chart our conversion rate by month." | `charts` |
| F7 | "Build me a dashboard of the pricing funnel." | `interactive-html` |
| F8 | "What does our handbook say about expenses?" | A knowledge-base lookup, not a judgment |

## Neighbouring-domain confusion — the pairs to watch

- **C5 against F1**: "go/no-go" must mean a decision with stakes, not any request phrased as a question. If F1 starts
  loading, the trigger list needs the lookup negative stated before the stake list.
- **C3 against F2**: compliance language appears in both a red-team request and a summary request; the verb is what
  separates them.
- **F5 against C1–C5**: a decision word inside a production request is the closest neighbour this Skill has, and it must
  route to `office-documents` every time. If it loads here, the stakes list has become a decision list.

## Results log (append per run)

| date | description as run | model family | positives loaded | forbidden loads | neighbouring-skill recall delta | verdict | notes |
|---|---|---|---|---|---|---|---|
| | | | /5 | /8 | | | |
