# An interactive report is built by a toolkit

An agent asked for an interactive HTML report writes the report's content and structure as an
HTML fragment, and a toolkit in the image turns it into the document: one ECharts runtime, the
Mineral look, charts laid out at the reader's width, slicers, pages, and the checks. The agent
does not draw charts, axes, tooltips or filters itself.

## Status

Accepted and implemented. The toolkit is `html-report` beside `echarts-render` in the image, the
built-in Skill `interactive-html` tells an agent when and how to use it, and the chart palette of
both tools moved to one Mineral source.

Amended 2026-10-19: the Skill starts a report from a fixture, and the toolkit checks a finished one.

## Context

An HTML Artifact runs, only after the reader activates it, in an iframe sandboxed to
`allow-scripts` under a CSP that allows inline script and style and nothing else: no network, no
frames, no workers, no storage, no dialogs. A report the model writes by hand has to carry its own
chart code, and in practice it did: a small chart library per report (ticks, axes, tooltips) drawn
as SVG, which cannot be filtered, is not themed, and fails on a phone.

The chart tool already in the image did not help. `echarts-render` draws one chart at 800 x 500; its
`--html` page carries its own copy of ECharts and cannot be composed into a report. Embedding such
pages as scaled iframes gave fixed-size charts shrunk by CSS, so side-by-side charts were squashed
on a phone, titles and legends were large against the chart body, and nothing could be filtered.
Nothing in the system gave a report responsive layout, slicers, pages or a sandbox-safe runtime.

## Decision

- **A fragment in, a document out.** `html-report build SRC OUT` reads an HTML fragment (prose,
  tables, KPI strip, slicers, optional `data-page` sections, one JSON block per chart) and writes
  one self-contained document with the full ECharts build inlined once, the stylesheet and the
  runtime. It fails with one line per problem when the fragment cannot run in the sandbox and warns
  about advice (a chart without a title and source, too many series, boilerplate). `--preview`
  draws every chart to PNG through the same code the browser runs.
- **Charts are re-laid-out, never scaled.** The runtime measures each figure and applies a profile
  (narrow, medium, wide) to fonts, legend, grid, labels and aspect through ECharts itself. The
  caption is HTML, made from the option's `title`; the plot never shrinks for the title. Columns of
  charts appear only where each gets its own width.
- **One column: the page's width is the measure.** A paragraph, a list and a timeline end where a
  table, a card and a figure end, so text never stops short of the data beneath it and leaves the
  rest of the row empty. The page's own `max-width` keeps the line readable; it, not a prose cap of
  its own, is what bounds the measure, and a fragment that wants a narrower one writes it.
- **One option language.** A chart is the same ECharts option JSON `echarts-render` takes, with
  data as tidy rows, `seriesBy` and `aggregate` to turn rows into series, slicer ids as tokens, and
  a `palette` choice. `core.js` is shared by the browser runtime and the node preview, so they cannot
  diverge.
- **Mineral is the default look, in the reader's own mode.** The stylesheet uses the product's role
  names and values for both colour schemes, hairlines instead of shadows, and the contrast floors of
  the Web theme. The document wears the mode on its root (`data-color-mode`): the product's Artifact
  frame hands over the reader's choice and keeps it current, so an embedded report follows the
  product rather than the operating system; opened on its own the report follows the system scheme
  and follows it as it changes. It is a default, not a lock: a fragment may override the CSS
  variables.
- **One palette, narrow spectrum respected.** Stone plus gold cannot tell eight series apart, so the
  chart palette is designed rather than derived: gold leads a categorical set chosen by measured
  distance under colour-vision deficiency (six series are distinguishable for every reader; the build
  warns above six), a `highlight` mode (one gold series over graded stone), a gold `sequential` ramp
  and a stone-centred `diverging` ramp. `chart-render/palette.json` is the only source; `theme.json`
  refers to its roles, and PNG charts from `echarts-render` use it too.
- **No boilerplate.** The toolkit adds nothing to the page (no footer, credit, timestamp or
  watermark), and the build warns on AI-generation lines, disclaimers, privacy or copyright notices
  and "generated on" footers.
- **Examples are fixtures.** The four example fragments live in the Skill's `references/`, are what an
  agent reads through `load_skill`, and are the fixtures of the browser tests, so what a model is shown
  is what is tested. The wheel check fails if one is missing.

## Considered options

- **Make `echarts-render --html` emit fragments.** Rejected: it would still give one chart with no
  layout, filters or pages, and every report would need its own wiring.
- **Generate the whole report from a JSON spec.** Rejected: a report's narrative and structure are
  its content, and a spec language narrows them; the toolkit owns only what is deterministic.
- **Load ECharts or fonts from a CDN.** Rejected: the sandbox forbids the network.
- **Keep the old categorical palette for PNG charts.** Rejected: one palette keeps a PNG in an answer
  and an interactive chart in a report the same family.

## Consequences

- The image gains `html-report`, its runtime files and the shared palette. A built report is about
  1.2 MB with ECharts inlined; the Artifact limit is 20 MiB.
- The existing PNG charts change colour.
- CI installs `resvg` and the font for the chart and report tests, and the chart toolkit's Node
  dependency, through `scripts/install-chart-tools.sh` and a `chart-render-install` target.
- A report is built in the Agent Workspace by ordinary tools, then attached like any file.

## Amendment 2026-10-19: the toolkit starts a report from a fixture and checks a finished one

The toolkit assumed a model would read an example and write a fragment. On 2026-10-19 a live Run
loaded `interactive-html`, then wrote `scripts/build_report.py` to emit a 30 KB self-contained page
with five inline SVG charts; its reasoning said "no html-report build to avoid schema risk". The
examples existed and the build was available; neither was used. Reading an example is an
instruction a model can skip. Starting from the example is a file it can edit.

**Decision — a starter and a check.**

- **Starter.** The Skill copies the nearest example into `report/src.html` with `load_skill`'s
  `write_to`, which already copies a document's exact bytes into the Agent Workspace. The Agent
  edits content freely — sections, prose, numbers, charts — and keeps the machine contract: the
  fragment is built by `html-report`, carries at least one paired chart, and declares its slicers.
  A copy is not a read (ADR 0057); the example stays a fixture of the browser tests, and an
  unmodified copy cannot be published as the deliverable because a Skill document is refused by
  digest (ADR 0046).
- **Check.** `html-report build` stays permissive about chart count — it is a generic fragment
  builder, and a chartless fragment is legitimate — and warns at zero. New `html-report check
  REPORT` inspects a finished document: its `report-source` block is present and rebuilds to the
  same bytes (ADR 0051), and the fragment carries at least one paired chart. Every failure is an
  error naming the rebuild, because the Agent can still correct it; the Skill runs `check` before
  `attach_artifact`, and the platform never blocks (ADR 0058).

Rejected: making `build` error at zero charts. It would put a Skill's delivery rule into a generic
tool and refuse legitimate chartless reports; the check is the Skill's gate, and the build keeps its
job.

Consequences: a hand-built page fails `check` with a message that names the build; a page built from
the starter passes; the examples become starters without a second copy of their shape.
