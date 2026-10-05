# Fragment reference

The fragment is the body of the page. `html-report` adds the document, the Mineral stylesheet, ECharts and the runtime around it, and adds nothing else: no footer, no credit, no date.

## Layout

- `header.report-head`: `h1`, `p.lede` (the conclusion), `p.meta` (period and source).
- `section.kpis` holding `.kpi`: `<b>` the value, `<span>` the label, `<small>` the comparison, with `<span class="delta up|down|flat">` carrying the sign and the number.
- `main` holds free content, or `section[data-page="id"][data-title="Title"]` blocks: two or more become tabs (short ASCII ids). A page holds one focus: a few charts, the prose that reads them, maybe a table.
- `.grid` puts charts side by side only where each gets about 360px of its own width; `figure.chart.wide` takes the whole row. A plain `<table>` is wrapped to scroll; `.num` right-aligns a numeric cell.
- Prose parts: `.callout[data-kind="insight|caution|risk|note"]` (the label is added for you), `p.note`, `ol.timeline > li[data-tone] > time, b, p, span.tag`, `span.tag[data-tone="1".."6"]` (a plain `.tag` is neutral and the first choice; a tone is for a real category, always with its text).

## A chart

```html
<figure class="chart" data-chart="trend"></figure>          <!-- add .wide for a full row -->
<script type="application/json" id="chart-trend">
{ "filters": ["region"], "aspect": "16:9", "minHeight": 240, "orient": "auto",
  "option": { ...an ECharts option as JSON, the same you would give echarts-render... } }
</script>
```

- Data: tidy rows (one object per row), once, in `<script type="application/json" id="data-sales">[...]</script>`, used by any chart with `"dataset": {"from": "sales"}`; or inline as `"dataset": {"source": [...]}`.
- `series[].encode` names row fields (`{"x": "quarter", "y": "revenue"}`; a pie takes `itemName` and `value`). `seriesBy: "field"` makes one series per distinct value of that field; `seriesOrder: [...]` fixes their order; `aggregate` (`sum` default, `avg`, `min`, `max`, `count`, `first`, `last`) combines rows that share an x. It draws bar, line, scatter, pie and funnel from rows; for heatmap, radar, boxplot, candlestick, sankey and the like write `series.data` yourself and the theme and layout still apply.
- `title.text` and `title.subtext` become the caption above the chart, so do not repeat them in HTML. The legend, axes, fonts, sizes and colours come from the runtime: do not set `fontSize`, `grid`, `legend` placement, or `color`.
- `"palette"`: unset (categorical, up to six series), `"highlight"` (one gold series, the rest stone), `"sequential"`, `"diverging"`.
- `orient`: `auto` lays long or many vertical-bar categories out horizontally on narrow widths; `keep` stops that.
- A formatter is a template string (`"{b}: {c}"`), never a function.

## Slicers

`<div class="slicer" data-slicer="region" data-field="region" data-label="地区"></div>` filters rows by a field. `data-mode="multi"` for several choices (single mode has an All choice; `data-all="false"` removes it), `data-values='["华东","华北"]'` to fix the choices and their order, `data-ui="chips|select"` (chips up to six values, a select above). A metric switch: `data-type="metric" data-options='[{"label":"销售额","y":"revenue"},{"label":"利润","y":"profit"}]'`, used by writing `"{metric}"` in an `encode` value or a title. A chart obeys a slicer only if its `filters` lists the slicer's id, so list it in every chart it should drive. A slicer hides on a page where none of its charts shows.

## Controls for what-if pages

`data-type="slider"` with `data-min`, `data-max`, `data-step`, `data-value`, `data-unit`: a range input whose value you read in your own script. Every slicer id is also a `{id}` token in chart titles and strings. Read `references/example-scenario.html` before building one.

## Your own script

Only for arithmetic and wiring, after the runtime is ready:

```js
Report.ready(() => {
  const draw = () => Report.chart("fx").setRows(rowsFrom(Report.slicer("spread").value()));
  document.addEventListener("report:slicer", draw); draw();
});
```

`Report.chart(id)` gives `{instance, rows(), setRows(rows), refresh()}`; `Report.slicer(id)` gives `{value(), set(v)}`. No storage, dialogs, network, `eval`, or drawing of your own.

## What the build says

`html-report build` fails, naming the chart or line, on: bad JSON, a figure without its block (or the reverse), duplicate ids, a series type that cannot be drawn, an option ECharts cannot draw, no data, a `filters` entry that names no slicer, a field missing from the rows, and anything the sandbox refuses (external URLs, `<link>`, `<iframe>`, `<form>`, storage, dialogs, `fetch`, `eval`). It warns about a chart without `title.text` and `subtext`, more than six series, two value axes, long vertical-bar labels, pages that hold no chart or table, drawing code in your script, and boilerplate (AI-generated lines, disclaimers, privacy or copyright notices, "generated on" footers). Warnings are advice: take them unless the user asked for the thing.
