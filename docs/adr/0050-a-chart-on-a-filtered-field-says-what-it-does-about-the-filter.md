# ADR 0050 — a chart that sits on a filtered field says what it does about the filter

A slicer was wired to a chart only when the chart's `filters` named it, and nothing checked the
converse: a chart whose rows carried the slicer's field and did not name it built green. The
contract is now that the slicer reaches every such chart and each of them declares itself, either
in `filters` or in `unfiltered`, and that the build refuses anything else and shows the author
what the filter does to the chart before it ships.

## Status

Accepted and implemented. A behaviour change in `html-report` and one added key in a chart block;
no migration, because a fragment is written and built in the same turn.

## Context

The mechanism is opt-in one chart at a time. `core.js` filters rows for the ids in a chart's own
`filters`, and the runtime redraws a chart for those ids and for the ones its text names with a
`{id}` token. So a chart that omits an id is simply not wired to it, silently.

Nothing in the loop said so. Measured on the golden dashboard with one `filters` entry removed, the
build printed `1 page, 3 charts, 1 slicers, 0 warnings` and exited 0. The author's two channels were
blind to it as well: `--preview` drew the default state only, and a chart that ignores a control
looks exactly like one that follows it. The Skill's prose already carried the rule — "list it in the
`filters` of every chart it should drive" — and the omission survived it, because the invariant was
restated once per chart and verified by nobody. A rule that a long generation must repeat N times
loses a few of them, and the loss is invisible in every medium the author checks.

The predicate the check needs already existed. `_check_fields` refuses a `filters` entry whose
field no row of the chart has, which is `listed ⊆ present`, computed from `_row_fields(chart)`. The
failure is the other half of the same relation: `present \ listed`.

## Decision

**A chart on a filtered field declares itself.** If a filter slicer's `data-field` is in the fields
of a chart's rows, the chart must name that slicer in `filters`, or in `unfiltered` — a list of
ids, or an object of id to reason, which keeps an exemption's reason in the source where a reader
of the fragment can weigh it. Anything else is an `html-report:` error naming both remedies. A
slicer inside a `data-page` section reaches the charts of that page and of no page, and no others,
because a control inside a page cannot be worked from another.

**The exemption is not silence.** `core.prepare` appends `不受「…」筛选影响` — in the report's
language, from the slicer's own label — to the caption of a chart that names it in `unfiltered`. A
chart standing still while its neighbours move is the one place a correct report reads as a broken
one, and this is the same objection the Skill already made to a filter that quietly leaves a chart
alone.

**The declaration is tested, not trusted.** `_check_wiring_responds` prepares each chart that lists
a slicer twice, at the default state and at one non-default choice of every slicer it names, through
the `core.prepare` the browser itself runs, and warns when the two are identical: a slicer that
moves no row the chart draws is either wired to nothing or should not be listed. It compares the
prepared plan — the rows, the tokens, the caption, the box — rather than a picture, so a chart that
answers only in its caption still counts as answering.

**The preview shows the filter working.** `--preview` writes, beside the two widths, one picture per
slicer state the chart follows, named `<chart>-900@<slicer>=<value>.png`. Interactivity is then
visible in the step the Skill already asked for — look at the charts before attaching them — rather
than in a claim about them.

## Consequences

- The invariant lives in the tool and is stated once in each place an author reads: the error
  message at the moment of the omission, `--help`, `interactive-html`'s hub, and `fragment.md`. The
  error message is the one that matters: it arrives where the author is looking.
- Auto-wiring by field was considered and rejected as the default. It would make forgetting
  impossible, but it would silently collapse the charts that compare across the sliced dimension —
  one bar per region becoming one bar — and over-wiring is invisible in exactly the way omission
  was. The declaration is explicit, and the exemption is the small, deliberate case.
- A report build now prepares each chart once more per slicer it lists, and `--preview` writes up to
  one more picture per chart per slicer. Both are the same order of work the build already did.
- The check binds only where it can bite: a filter slicer (never a metric switch or a slider), a
  field the chart's own rows carry, and a slicer the reader can reach from where the chart stands.
