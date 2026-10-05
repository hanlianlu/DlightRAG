# An interactive report is built by a toolkit

An agent asked for an interactive HTML report writes the report's content and structure as an
HTML fragment, and a toolkit in the image turns it into the document: one ECharts runtime, the
Mineral look, charts laid out at the reader's width, slicers, pages, and the checks. The agent
does not draw charts, axes, tooltips or filters itself.

## Status

Accepted and implemented. The toolkit is `html-report` beside `echarts-render` in the image, the
built-in Skill `interactive-html` tells an agent when and how to use it, and the chart palette of
both tools moved to one Mineral source.

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
- **One option language.** A chart is the same ECharts option JSON `echarts-render` takes, with
  data as tidy rows, `seriesBy` and `aggregate` to turn rows into series, slicer ids as tokens, and
  a `palette` choice. `core.js` is shared by the browser runtime and the node preview, so they cannot
  diverge.
- **Mineral is the default look.** The stylesheet uses the product's role names and values for both
  colour schemes (the report follows the system scheme), hairlines instead of shadows, and the
  contrast floors of the Web theme. It is a default, not a lock: a fragment may override the CSS
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
