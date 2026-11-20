# Built-in Skills — the routing eval suite

Routing is a **joint** property, not a per-skill one. Every description is injected into the index of every session
for every user, and adding or editing one Skill's description moves where the other Skills load — the guidance calls
it action at a distance. A Skill that starts loading for another Skill's questions is not a bug in that other Skill,
and a description change that is reviewed alone is not reviewed.

This file is the suite: one truth table over all built-in Skills. It has a **machine half** that runs in CI (the
anchor column, checked against each `description` by `tests/unit/test_skills_routing_eval.py`) and a **run half** that
needs the orchestrator model families.

## How to run

`make eval-serve` runs an app for the driver with no sign-in, on a database of its own (`$EVAL_DB`, default
`dlightrag_eval`, created when absent and never dropped); then
`uv run python scripts/run_skill_routing_eval.py --base-url http://127.0.0.1:8110 --out /tmp/skill-eval [--only ID]`
starts one research-mode run per case and writes `events.txt`, `transcript.json`, `artifacts.json` and `score.json`
under `<out>/<case>/`.

- Queries **verbatim**, × 3 repetitions, × every orchestrator model family in use. A router that only works on one
  family is not shipped.
- Record `loaded / not loaded`, the artefact produced, and the first sentence that shows which way it went.
- **Ship only if** every positive loads, **zero** forbidden cases load, and no neighbouring Skill loses more than 5pp
  recall. The anchor column is the cheap part: it fails when a description edit drops a phrase a positive case
  depends on, which is the most common way a working router breaks.

## Anchors — the CI half

Each anchor is a phrase that must appear in that Skill's `description`. Editing a description so that an anchor
disappears means either the case was reworded deliberately (edit this file in the same change) or the router lost a
trigger it was relying on.

| Skill | Anchor in its description |
|---|---|
| `council` | `go/no-go` |
| `charts` | `chart, graph or plot` |
| `interactive-html` | `interactive HTML report` |
| `office-documents` | `Word, Excel, PowerPoint or PDF` |
| `skill-creator` | `make, edit or remove a personal DlightRAG skill` |

## Positives — the Skill must load

| ID | Query | Owner |
|---|---|---|
| P4 | "I need a go/no-go on this pricing decision by Friday." | `council` |
| P5 | "Second opinion on this acquisition before we sign." | `council` |
| P6 | "Chart our revenue by quarter." | `charts` |
| P7 | "Plot the margin trend for the last eight quarters." | `charts` |
| P8 | "Put the assumptions in a Word document." | `office-documents` |
| P9 | "Give me the model as an Excel file." | `office-documents` |
| P10 | "Build an interactive report with filters and tabs." | `interactive-html` |
| P11 | "我要一个可筛选的可视化报告。" | `interactive-html` |
| P12 | "Turn this weekly routine into a skill." | `skill-creator` |
| P13 | "Delete the skill I made last week." | `skill-creator` |

## Forbidden loads — the query must not load that Skill

| ID | Query | Must not load | Owner instead | Why |
|---|---|---|---|---|
| F3 | "Summarise this 40-page contract." | `council` | none | A summary, not a judgment |
| F4 | "Write the pricing page copy for the new tier." | `council` | none | Production, not judgment |
| F5 | "What did we charge the enterprise tier last year?" | `council` | none | A lookup |
| F6 | "Build a dashboard of the pricing funnel." | `charts` | `interactive-html` | A dashboard is not a single chart |
| F7 | "Give me one chart as a PNG." | `interactive-html` | `charts` | One figure is not a report |
| F8 | "Render this as HTML." | `office-documents` | `interactive-html` | HTML is not an Office file |
| F9 | "Turn this into a PDF." | `interactive-html` | `office-documents` | PDF is an Office-type document |

## Adjacency — the pairs most likely to confuse

- **P6/P7 vs P10/P11**: one chart is `charts`; several charts with filters, tabs or sliders are `interactive-html`.
  The rendering Skill's description carries the negative explicitly.
- **F6/F7**: the same boundary from both sides — a dashboard is not a chart, a PNG is not a report.
- **F8/F9**: HTML is not an Office file and a PDF is; `office-documents` covers PDF explicitly.
- **P12/P13 vs F5**: asking to make or delete a skill is `skill-creator`; asking what a skill says or what we charged
  is a lookup.

## Run log (append per run)

| date | Skills as run (commit) | model families | positives loaded | forbidden loads | neighbour recall delta | verdict | notes |
|---|---|---|---|---|---|---|---|
| | | | /13 | /9 | | | |

