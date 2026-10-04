---
name: charts
description: Use when the user asks for a chart, graph or plot, or when a figure would show the answer better than text.
---

# Charts

Charts are Apache ECharts options rendered by `echarts-render`, which draws them on the server in the house theme with a Chinese font; there is no matplotlib and no browser. Build the `option` as a Python dict from pandas (`.tolist()`, NaN as `None`), render it, look at the PNG with `view`, and attach it with `attach_artifact`:

```python
chart = json.dumps(option, ensure_ascii=False)
subprocess.run(["echarts-render", "-", "artifacts/name.png"], input=chart, text=True, check=True)
```

- **Theme**: leave colors, fonts, background, legend and layout to it. Give every chart `title.text`, and a `title.subtext` with the unit and the source in words.
- **Option**: it is JSON, so a formatter is a template string such as `"{b}: {c} 亿元"`, never a JavaScript function; `custom` and `map` series cannot be drawn.
- **Layout**: long or many category names go on `yAxis` (horizontal bars). Keep to eight series or fewer, folding the rest into “其他”; use one value axis, never two.
- **Files**: the PNG is the figure in the answer, laid out at 800 × 500 unless `--width` and `--height` say otherwise. Add `--html artifacts/name.html` only when the user wants to hover or zoom, and `--svg artifacts/name.svg` for a vector file.
- **Failures**: a failed render prints one `echarts-render:` line that says what to fix. A line naming characters without a font means they show as boxes: emoji, Korean and Arabic are not in the font.
