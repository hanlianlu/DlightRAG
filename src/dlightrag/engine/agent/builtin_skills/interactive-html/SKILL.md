---
name: interactive-html
description: Use when the user asks for an interactive HTML report, dashboard or web page (交互式报告, 可视化报告, 仪表盘, 网页报告, 可筛选, 多页签, 推演页面), or a report that combines several charts with filters, tabs or controls. Not a single chart image (charts), Office files (office-documents) or Markdown.
---

# Interactive HTML

A report is built, not drawn. You write what it says and how it is organised, as an HTML fragment; `html-report` supplies the page: the Mineral look, one ECharts runtime, responsive charts, filters, tabs, and the checks. Never write a chart, an axis, a tooltip, a tab bar or a filter yourself, never embed `echarts-render --html` pages as iframes, and never scale a chart with CSS. A hand-drawn chart cannot be filtered, is unreadable on a phone, and the reader's sandbox refuses much of what you would reach for. Also load `charts`: its rules for the chart option (title and subtext, at most six series, JSON only, long names on the y axis) apply unchanged.

Before writing, read the example nearest to the request, with `load_skill` and its `path` (the `read` tool cannot see them): `references/example-brief.html` (one page), `example-dashboard.html` (filters and a metric switch), `example-multipage.html` (pages, a timeline, tags), `example-scenario.html` (sliders and a what-if). Each is a complete fragment that builds. `references/fragment.md` lists every attribute, and `html-report --help` states the same rules; `html-report` is a command to run with bash, not a skill, and its source is not where to learn them.

## Shape it first

- The answer leads: the header's lede is the conclusion, the KPI strip carries the four to six numbers that prove it, each chart answers one question and its title says what to see when it is there to make a point.
- One scrolling page unless the content holds separate questions a reader will come back to. Then pages (two to five), each with its own focus. A brief needs no tabs and no filters.
- A filter earns its place only on a dimension that changes the answer. Put it where it applies, and list it in the `filters` of every chart it should drive: a filter that quietly leaves one chart unchanged is a bug the reader sees at once.
- Tables for exact values, charts for shape, prose for what the numbers mean. Short paragraphs.
- Data goes in as tidy rows (`dataset.source`), aggregated with pandas first and checked against the source. Never invent a number: say in a `.note` what is missing.
- Write in the user's language, with the unit and the source in each chart's `subtext` and the as-of date in the header.

## Build

Write the fragment to `report/src.html`, then:

```
html-report build report/src.html artifacts/report.html --preview tmp/preview
```

Fix every `html-report:` error it prints, `view` the 360 and 900 previews of the charts that matter, then `attach_artifact` the built file. `artifacts/` holds only that file: previews and scratch work stay in `tmp/`, or the artifact is refused as unsafe. A follow-up edits `report/src.html` and rebuilds; the built file is never patched by hand. You cannot open the report: say what you checked (the build and the previews), never that you tested it in a browser, and do not install a browser or packages to try. The reader activates the artifact to interact with it.

## The look

The default is Mineral, DlightRAG's identity: stone neutrals, one gold accent, hairlines instead of shadows, quiet. The page and every chart already wear it in light and dark, so set no colours, fonts, sizes, legend or grid in an option. Leave Mineral only when the user brings their own brand or the subject needs it, and then override the CSS variables once in a single `<style>` block and keep text at 4.5:1.

Colour carries meaning, so spend it: `"palette": "highlight"` when the point is one series, `"sequential"` for magnitude, `"diverging"` around a midpoint, the default for unordered groups, and fold the rest into 其他. Category and status are always written out; colour is never the only carrier.

## Gotchas

- Say only what concerns the subject. No "AI generated", disclaimer, privacy or compliance notice, cookie or copyright line, "generated on" footer, credit or call to action: not even a one-line note describing how the page was made. A source line, an as-of date and the assumptions behind these numbers are content.
- localStorage, alert, confirm, fetch, CDN scripts, web fonts, external images, forms: the sandbox refuses them and the build fails on them. An option holds no JavaScript: a formatter is a template string.
- Give a chart no second title in HTML: the caption is made from its `title`. Give it no fixed width or height: use `.grid`, `aspect` and `minHeight`.
- One dataset and a filter, never one pre-drawn picture per filter state.
- Custom script is for arithmetic and wiring (a scenario slider feeding `Report.chart(id).setRows`), never for drawing.
