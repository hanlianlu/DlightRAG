// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** What `core.prepare` hands ECharts for one chart: the option each slicer state and width produces. */

const assert = require('node:assert/strict');
const {test} = require('node:test');

const Core = require('../report/core.js');
const palette = require('../palette.json');
const structure = require('../theme.json');

const themes = Core.themes(structure, palette.light);

const region = {id: 'region', type: 'filter', field: 'region', mode: 'single', all: true};
const regions = {id: 'region', type: 'filter', field: 'region', mode: 'multi', all: true};
const metric = {
  id: 'metric',
  type: 'metric',
  options: [
    {label: '销售额', y: 'revenue'},
    {label: '利润', y: 'profit'},
  ],
};

const quarters = ['2023Q1', '2023Q2', '2023Q3'];
const sales = ['华东', '华北', '华南'].flatMap((r, i) =>
  quarters.map((q, j) => ({region: r, quarter: q, revenue: 100 * (i + 1) + 10 * j, profit: 10 * (i + 1) + j})),
);

const chart = (option, extra = {}) => ({filters: [], slicers: {}, option, ...extra});
const plan = (spec, state = {}, width = 720) => Core.prepare(spec, state, width, {themes});
const names = (p) => p.option.series.map((s) => s.name);

const bars = (series = {}) =>
  chart({
    dataset: {source: sales},
    xAxis: {type: 'category'},
    yAxis: {type: 'value'},
    series: [{type: 'bar', seriesBy: 'region', encode: {x: 'quarter', y: 'revenue'}, ...series}],
  });

test('seriesBy makes one series per value, in order of first appearance, aligned on the categories', () => {
  const p = plan(bars());
  assert.deepEqual(names(p), ['华东', '华北', '华南']);
  assert.deepEqual(p.option.xAxis.data, quarters);
  assert.deepEqual(p.option.series[0].data, [100, 110, 120]);
  assert.equal(p.option.series[0].encode, undefined);
  assert.equal(p.option.series[0].seriesBy, undefined);
  assert.equal(p.option.dataset, undefined);
});

test('seriesOrder overrides the order of appearance', () => {
  const p = plan(bars({seriesOrder: ['华南', '华东']}));
  assert.deepEqual(names(p), ['华南', '华东']);
});

test('a missing cell is null, and a stack survives the expansion', () => {
  const rows = [
    {q: 'Q1', r: 'A', v: 1},
    {q: 'Q1', r: 'B', v: 2},
    {q: 'Q2', r: 'A', v: 3},
  ];
  const p = plan(
    chart({
      dataset: {source: rows},
      xAxis: {type: 'category'},
      yAxis: {type: 'value'},
      series: [{type: 'bar', stack: 'total', seriesBy: 'r', encode: {x: 'q', y: 'v'}}],
    }),
  );
  assert.deepEqual(p.option.series[0].data, [1, 3]);
  assert.deepEqual(p.option.series[1].data, [2, null]);
  assert.deepEqual(
    p.option.series.map((s) => s.stack),
    ['total', 'total'],
  );
});

test('rows that share a cell are summed, or reduced the way the series asks', () => {
  const rows = [
    {q: 'Q1', v: 1},
    {q: 'Q1', v: 3},
    {q: 'Q2', v: 5},
  ];
  const draw = (extra) =>
    plan(
      chart({
        dataset: {source: rows},
        xAxis: {type: 'category'},
        yAxis: {type: 'value'},
        series: [{type: 'line', name: 'v', encode: {x: 'q', y: 'v'}, ...extra}],
      }),
    ).option.series[0].data;
  assert.deepEqual(draw({}), [4, 5]);
  assert.deepEqual(draw({aggregate: 'avg'}), [2, 5]);
  assert.deepEqual(draw({aggregate: 'max'}), [3, 5]);
});

test('a time axis lines the series up on the union of their times, sorted, with null where one has none', () => {
  const rows = [
    {t: '2024-03', r: 'A', v: 3},
    {t: '2024-01', r: 'A', v: 1},
    {t: '2024-02', r: 'B', v: 2},
  ];
  const p = plan(
    chart({
      dataset: {source: rows},
      xAxis: {type: 'time'},
      yAxis: {type: 'value'},
      series: [{type: 'line', seriesBy: 'r', encode: {x: 't', y: 'v'}}],
    }),
  );
  assert.deepEqual(p.option.series[0].data, [['2024-01', 1], ['2024-02', null], ['2024-03', 3]]);
  assert.deepEqual(p.option.series[1].data, [['2024-01', null], ['2024-02', 2], ['2024-03', null]]);
});

test('a value axis sorts numerically, not as text, and stacked areas line up', () => {
  const rows = [
    {x: 10, r: 'A', v: 1},
    {x: 9, r: 'A', v: 2},
    {x: 100, r: 'B', v: 3},
  ];
  const p = plan(
    chart({
      dataset: {source: rows},
      xAxis: {type: 'value'},
      yAxis: {type: 'value'},
      series: [{type: 'line', stack: 's', areaStyle: {}, seriesBy: 'r', encode: {x: 'x', y: 'v'}}],
    }),
  );
  assert.deepEqual(p.option.series[0].data.map((point) => point[0]), [9, 10, 100]);
  assert.deepEqual(p.option.series[1].data, [[9, null], [10, null], [100, 3]]);
});

test('a scatter series keeps one point per row', () => {
  const p = plan(
    chart({
      dataset: {source: [{a: 1, b: 2, g: 'x'}, {a: 3, b: 4, g: 'y'}]},
      xAxis: {type: 'value'},
      yAxis: {type: 'value'},
      series: [{type: 'scatter', seriesBy: 'g', encode: {x: 'a', y: 'b'}}],
    }),
  );
  assert.deepEqual(names(p), ['x', 'y']);
  assert.deepEqual(p.option.series[1].data, [[3, 4]]);
});

test('a single filter keeps the rows of one value, and All keeps every row', () => {
  const spec = bars({});
  spec.filters = ['region'];
  spec.slicers = {region};
  assert.deepEqual(names(plan(spec, {region: '华北'})), ['华北']);
  assert.deepEqual(names(plan(spec, {region: null})), ['华东', '华北', '华南']);
  assert.deepEqual(names(plan(spec, {})), ['华东', '华北', '华南']);
});

test('a multi filter keeps the rows of any chosen value; none chosen is All', () => {
  const spec = bars({});
  spec.filters = ['region'];
  spec.slicers = {region: regions};
  assert.deepEqual(names(plan(spec, {region: ['华东', '华南']})), ['华东', '华南']);
  assert.deepEqual(names(plan(spec, {region: []})), ['华东', '华北', '华南']);
});

test('a filter that matches no row leaves an empty chart, not an error', () => {
  const spec = bars({});
  spec.filters = ['region'];
  spec.slicers = {region};
  const p = plan(spec, {region: '西南'});
  assert.equal(p.empty, true);
  assert.deepEqual(p.option.series, []);
});

test('a series keeps its colour when a filter removes the ones before it', () => {
  const spec = bars({});
  spec.filters = ['region'];
  spec.slicers = {region};
  const all = plan(spec, {});
  const only = plan(spec, {region: '华南'});
  assert.equal(only.option.series[0].itemStyle.color, all.option.series[2].itemStyle.color);
  assert.equal(only.option.series[0].itemStyle.color, themes.categorical.color[2]);
});

test('the metric slicer picks the field of encode.y and the label of the title and the name', () => {
  const spec = chart(
    {
      title: {text: '季度{metric}', subtext: '单位：万元'},
      dataset: {source: sales},
      xAxis: {type: 'category'},
      yAxis: {type: 'value'},
      series: [{type: 'line', encode: {x: 'quarter', y: '{metric}'}}],
    },
    {filters: ['metric'], slicers: {metric}},
  );
  const first = plan(spec, {});
  assert.equal(first.caption.title, '季度销售额');
  assert.equal(first.option.series[0].name, '销售额');
  assert.deepEqual(first.option.series[0].data, [600, 630, 660]);
  const second = plan(spec, {metric: '利润'});
  assert.equal(second.caption.title, '季度利润');
  assert.deepEqual(second.option.series[0].data, [60, 63, 66]);
});

const spread = {id: 'spread', type: 'slider', min: -1, max: 2, step: 0.25, value: 0.5, unit: 'pp'};
const inflation = {id: 'inflation', type: 'slider', min: 0, max: 4, step: 0.25, value: 2, unit: '%'};
const scenario = {id: 'scenario', type: 'filter', mode: 'single', all: false, values: ['基准', '乐观', '悲观']};
const many = {...regions, id: 'many'};
const slicers = {region, many, metric, spread, inflation, scenario};
const tokens = (option, state = {}, lang = 'zh', extra = {}) =>
  Core.prepare(chart(option, extra), state, 720, {themes, slicers, lang});

test('a slider snaps to its notches, stays inside min..max, and reads with its decimals, a plus and its unit', () => {
  assert.equal(Core.snapSlider(spread, 0.6), 0.5);
  assert.equal(Core.snapSlider(spread, '1.13'), 1.25);
  assert.equal(Core.snapSlider(spread, -7), -1);
  assert.equal(Core.snapSlider(spread, 9), 2);
  assert.equal(Core.snapSlider(spread, ''), null);
  assert.equal(Core.snapSlider(spread, 'x'), null);
  assert.equal(Core.snapSlider({id: 'o', min: 0, max: 1, step: 0.3}, 1), 0.9, 'the last notch at or below max');
  assert.equal(Core.snapSlider({id: 't', min: 0, max: 1, step: 0.1}, 0.30000000000000004), 0.3, 'no floating-point noise');
  assert.equal(Core.formatSlider(spread, 0.5), '+0.50pp');
  assert.equal(Core.formatSlider(spread, -0.25), '-0.25pp');
  assert.equal(Core.formatSlider(spread, 0), '0.00pp');
  assert.equal(Core.formatSlider(inflation, 2), '2.00%', 'no plus when the range never runs below zero');
  assert.equal(Core.formatSlider({id: 'y', min: 2020, max: 2030, step: 1, unit: '年'}, 2026), '2026年');
  assert.equal(Core.formatSlider({id: 'h', min: 0.5, max: 3, step: 1}, 1.5), '1.5', 'the decimals of min count too');
  assert.equal(Core.slicerDefault({...spread, value: 9}), 2);
  assert.equal(Core.slicerDefault({...spread, value: undefined}), -1);
});

test('every slicer is a {id} token in any string of the option, whether a chart lists it or not', () => {
  const p = tokens(
    {
      title: {text: '利差 {spread}', subtext: '通胀 {inflation} · {scenario} · {region}'},
      legend: {data: ['{spread}']},
      dataset: {source: [{k: '{spread}', v: 1}]},
      xAxis: {type: 'category', name: '{inflation}'},
      yAxis: {type: 'value'},
      tooltip: {formatter: '{b}: {c} ({nothing}) {spread}'},
      series: [{type: 'line', name: '{spread}', encode: {x: 'k', y: 'v'}}],
    },
    {spread: 1},
  );
  assert.deepEqual(p.caption, {title: '利差 +1.00pp', subtitle: '通胀 2.00% · 基准 · 全部'});
  assert.deepEqual(p.option.legend.data, ['+1.00pp']);
  assert.equal(p.option.xAxis.name, '2.00%');
  assert.equal(p.option.series[0].name, '+1.00pp');
  assert.equal(p.option.tooltip.formatter, '{b}: {c} ({nothing}) +1.00pp', 'a {name} no slicer owns stays');
  assert.deepEqual(p.option.xAxis.data, ['{spread}'], 'rows are data, not text');
});

test('a filter token is the chosen value, several joined, or All in the report language', () => {
  const text = (state, lang) => tokens({title: {text: '{region}|{many}'}, series: [{type: 'bar', data: [1]}], xAxis: {type: 'category', data: ['a']}, yAxis: {}}, state, lang).caption.title;
  assert.equal(text({}, 'zh'), '全部|全部');
  assert.equal(text({}, 'en'), 'All|All');
  assert.equal(text({region: '华东', many: ['华东', '华北']}, 'zh'), '华东|华东、华北');
  assert.equal(text({region: '华东', many: ['华东', '华北']}, 'en'), '华东|华东, 华北');
});

test('a slider token follows the state, snapped and kept inside its range', () => {
  const title = (state) => tokens({title: {text: '{spread}'}, series: [{type: 'bar', data: [1]}], xAxis: {type: 'category', data: ['a']}, yAxis: {}}, state).caption.title;
  assert.equal(title({}), '+0.50pp');
  assert.equal(title({spread: 9}), '+2.00pp');
  assert.equal(title({spread: -0.1}), '0.00pp');
  assert.equal(title({spread: 'x'}), '+0.50pp');
});

test('a metric slicer of any id fills its token, and in encode the token is the row field', () => {
  const kpi = {...metric, id: 'kpi'};
  const spec = chart(
    {
      title: {text: '季度{kpi}'},
      dataset: {source: sales},
      xAxis: {type: 'category'},
      yAxis: {type: 'value'},
      series: [{type: 'line', encode: {x: 'quarter', y: '{kpi}'}}],
    },
    {filters: ['kpi'], slicers: {kpi}},
  );
  const p = Core.prepare(spec, {kpi: '利润'}, 720, {themes, slicers: {kpi}, lang: 'zh'});
  assert.equal(p.caption.title, '季度利润');
  assert.equal(p.option.series[0].name, '利润');
  assert.deepEqual(p.option.series[0].data, [60, 63, 66]);
});

test('a slider filters no rows, and tokenIds names the slicers whose token a chart writes', () => {
  const spec = chart(
    {dataset: {source: sales}, xAxis: {type: 'category'}, yAxis: {type: 'value'}, title: {text: '{spread}'}, series: [{type: 'bar', encode: {x: 'quarter', y: 'revenue'}}]},
    {filters: ['spread'], slicers: {spread}},
  );
  assert.deepEqual(plan(spec, {spread: 1}).option.series[0].data, [600, 630, 660]);
  assert.deepEqual(Core.tokenIds(spec.option, ['spread', 'inflation', 'region']), ['spread']);
  assert.deepEqual(Core.tokenIds({}, ['spread']), []);
});

test('the title leaves the chart for the caption, and so do the toolbox and the background', () => {
  const p = plan(chart({title: {text: 'T', subtext: 'S'}, toolbox: {feature: {}}, backgroundColor: '#fff', series: [{type: 'bar', data: [1]}], xAxis: {type: 'category', data: ['a']}, yAxis: {}}));
  assert.deepEqual(p.caption, {title: 'T', subtitle: 'S'});
  for (const key of ['title', 'toolbox', 'backgroundColor']) assert.equal(p.option[key], undefined);
  assert.equal(plan(chart({series: [{type: 'bar', data: [1]}], xAxis: {type: 'category', data: ['a']}, yAxis: {}})).caption, null);
});

test('the profile follows the width: narrow below 420, medium below 720, wide above', () => {
  const spec = bars({});
  assert.equal(plan(spec, {}, 360).profile, 'narrow');
  assert.equal(plan(spec, {}, 419).profile, 'narrow');
  assert.equal(plan(spec, {}, 420).profile, 'medium');
  assert.equal(plan(spec, {}, 719).profile, 'medium');
  assert.equal(plan(spec, {}, 720).profile, 'wide');
});

test('text sizes follow the profile and the legend is never larger than the axis labels', () => {
  for (const [width, size] of [[360, 11], [600, 12], [900, 12]]) {
    const p = plan(bars({}), {}, width);
    assert.equal(p.option.xAxis.axisLabel.fontSize, size);
    assert.equal(p.option.yAxis.axisLabel.fontSize, size);
    assert.ok(p.option.legend.textStyle.fontSize <= p.option.xAxis.axisLabel.fontSize);
    assert.ok(p.option.tooltip.textStyle.fontSize <= 12);
  }
});

test('the legend sits top left on a wide figure and along the bottom of a narrow one', () => {
  const wide = plan(bars({}), {}, 900).option;
  assert.equal(wide.legend.top, 0);
  assert.equal(wide.legend.bottom, undefined);
  const narrow = plan(bars({}), {}, 360).option;
  assert.equal(narrow.legend.bottom, 4);
  assert.equal(narrow.legend.top, undefined);
  assert.ok(narrow.grid.bottom > wide.grid.bottom);
});

test('a legend the author wrote keeps its entries but not its position', () => {
  const spec = bars({});
  spec.option.legend = {top: 80, left: 24, data: ['华东']};
  const legend = plan(spec, {}, 900).option.legend;
  assert.deepEqual(legend.data, ['华东']);
  assert.equal(legend.top, 0);
  assert.equal(legend.left, 12);
});

test('the grid margins belong to the report, not to the author', () => {
  const spec = bars({});
  spec.option.grid = {top: 124, left: 24, containLabel: false, show: true};
  const grid = plan(spec, {}, 360).option.grid;
  assert.equal(grid.left, 8);
  assert.equal(grid.outerBoundsMode, 'same');
  assert.equal(grid.show, true);
});

test('a vertical bar chart with long labels lies down on a narrow figure, first category on top', () => {
  const rows = ['新能源汽车及零部件制造', '半导体及集成电路'].map((industry, i) => ({industry, value: 10 - i}));
  const spec = chart({
    dataset: {source: rows},
    xAxis: {type: 'category'},
    yAxis: {type: 'value'},
    series: [{type: 'bar', encode: {x: 'industry', y: 'value'}, label: {show: true, position: 'top'}}],
  });
  const narrow = plan(spec, {}, 360);
  assert.equal(narrow.horizontal, true);
  assert.equal(narrow.option.yAxis.type, 'category');
  assert.equal(narrow.option.yAxis.inverse, true);
  assert.deepEqual(narrow.option.yAxis.data, rows.map((r) => r.industry));
  assert.equal(narrow.option.xAxis.type, 'value');
  assert.equal(narrow.option.series[0].label.position, 'right');
  assert.deepEqual(narrow.option.series[0].data, [10, 9]);
});

test('orient keep stops the flip, and short labels on a wide figure stay vertical', () => {
  const rows = ['新能源汽车及零部件制造', '半导体及集成电路'].map((industry, i) => ({industry, value: 10 - i}));
  const base = {
    dataset: {source: rows},
    xAxis: {type: 'category'},
    yAxis: {type: 'value'},
    series: [{type: 'bar', encode: {x: 'industry', y: 'value'}}],
  };
  assert.equal(plan(chart(base, {orient: 'keep'}), {}, 360).horizontal, false);
  assert.equal(plan(chart(base), {}, 900).horizontal, false);
  const many = {...base, dataset: {source: Array.from({length: 9}, (_, i) => ({industry: `类${i}`, value: i}))}};
  assert.equal(plan(chart(many), {}, 360).horizontal, true);
  assert.equal(plan(chart(many), {}, 900).horizontal, false);
});

test('labels that cannot wrap into two lines flip a bar chart at any width', () => {
  const rows = ['新能源汽车及零部件制造', '半导体及集成电路', '生物医药与医疗器械', '工业机器人与智能装备', '新型显示与光电子', '航空航天与国防装备'].map((industry, i) => ({industry, value: 10 - i}));
  const spec = chart({dataset: {source: rows}, xAxis: {type: 'category'}, yAxis: {type: 'value'}, series: [{type: 'bar', encode: {x: 'industry', y: 'value'}}]});
  assert.equal(plan(spec, {}, 500).horizontal, true);
  assert.equal(plan(spec, {}, 1000).horizontal, false);
});

test('a bar chart written horizontally lists its first category on top', () => {
  const rows = [{k: 'a', v: 1}, {k: 'b', v: 2}];
  const p = plan(chart({dataset: {source: rows}, xAxis: {type: 'value'}, yAxis: {type: 'category'}, series: [{type: 'bar', encode: {x: 'v', y: 'k'}}]}), {}, 600);
  assert.equal(p.option.yAxis.inverse, true);
  assert.deepEqual(p.option.yAxis.data, ['a', 'b']);
  const kept = plan(chart({dataset: {source: rows}, xAxis: {type: 'value'}, yAxis: {type: 'category', inverse: false}, series: [{type: 'bar', encode: {x: 'v', y: 'k'}}]}), {}, 600);
  assert.equal(kept.option.yAxis.inverse, false);
});

test('a long axis on a narrow figure can be zoomed with a gesture, and a wide one is left alone', () => {
  const rows = Array.from({length: 40}, (_, i) => ({d: `d${i}`, v: i}));
  const spec = chart({dataset: {source: rows}, xAxis: {type: 'category'}, yAxis: {type: 'value'}, series: [{type: 'line', name: 'v', encode: {x: 'd', y: 'v'}}]});
  assert.equal(plan(spec, {}, 360).option.dataZoom[0].type, 'inside');
  assert.equal(plan(spec, {}, 900).option.dataZoom, undefined);
});

test('a pie sums its slices by name and keeps their colours when a filter removes one', () => {
  const rows = [
    {r: '华东', p: 'A', v: 1},
    {r: '华东', p: 'B', v: 2},
    {r: '华北', p: 'B', v: 3},
    {r: '华北', p: 'C', v: 4},
  ];
  const spec = chart(
    {dataset: {source: rows}, series: [{type: 'pie', encode: {itemName: 'p', value: 'v'}}]},
    {filters: ['region'], slicers: {region: {...region, field: 'r'}}},
  );
  const all = plan(spec, {}, 720).option.series[0].data;
  assert.deepEqual(all.map((d) => [d.name, d.value]), [['A', 1], ['B', 5], ['C', 4]]);
  const north = plan(spec, {region: '华北'}, 720).option.series[0].data;
  assert.deepEqual(north.map((d) => d.name), ['B', 'C']);
  assert.equal(north[0].itemStyle.color, themes.categorical.color[1]);
});

test('a pie shows its labels outside on a wide figure and leaves them to the legend on a narrow one', () => {
  const spec = chart({dataset: {source: [{p: 'A', v: 1}, {p: 'B', v: 2}]}, series: [{type: 'pie', encode: {itemName: 'p', value: 'v'}}]});
  assert.notEqual(plan(spec, {}, 900).option.series[0].label.show, false);
  assert.equal(plan(spec, {}, 360).option.series[0].label.show, false);
  assert.equal(plan(spec, {}, 360).option.legend.bottom, 4);
});

test('a pie is seated in the box its legend leaves, in pixels, and keeps the radius an author wrote', () => {
  const rows = ['新能源汽车及零部件制造', '半导体及集成电路', '生物医药与医疗器械', '工业机器人与智能装备', '新型显示与光电子', '航空航天与国防装备'].map((p, i) => ({p, v: 10 - i}));
  const spec = (series = {}) => chart({dataset: {source: rows}, series: [{type: 'pie', encode: {itemName: 'p', value: 'v'}, ...series}]});
  const wide = plan(spec(), {}, 560);
  const [x, y] = wide.option.series[0].center;
  const [inner, outer] = wide.option.series[0].radius;
  assert.equal(x, 280);
  assert.ok(wide.option.legend.type === 'plain' && y > 70, `centre ${y}`);
  assert.ok(inner < outer, `radius ${inner}, ${outer}`);
  assert.ok(y - outer >= 48 + 14 && y + outer <= wide.height - 14, `pie ${y - outer}..${y + outer} in ${wide.height}`);
  assert.deepEqual(plan(spec({radius: ['10%', '50%']}), {}, 560).option.series[0].radius, ['10%', '50%']);
  // Without outside labels a narrow figure gives the pie more than the labelled one would get, and keeps it inside the figure.
  const narrow = plan(spec(), {}, 360);
  const [, narrowOuter] = narrow.option.series[0].radius;
  assert.ok(narrowOuter > (360 - 112) / 2 && narrowOuter * 2 <= 360 - 16, `narrow radius ${narrowOuter}`);
});

test('the height follows the aspect, a narrow figure gets a taller aspect, and minHeight is a floor', () => {
  const spec = (extra) => ({...bars({}), ...extra});
  assert.equal(plan(spec({aspect: '2:1'}), {}, 800).height, 400);
  const narrow = plan(spec({}), {}, 360).height;
  assert.ok(narrow >= 360 / 1.15 - 1, `narrow height ${narrow}`);
  assert.equal(plan(spec({minHeight: 500}), {}, 800).height, 500);
  assert.ok(plan(spec({aspect: '16:9'}), {}, 300).height >= 240);
});

test('a horizontal bar chart is as tall as its categories need', () => {
  const rows = Array.from({length: 12}, (_, i) => ({k: `类${i}`, v: i}));
  const p = plan(chart({dataset: {source: rows}, xAxis: {type: 'value'}, yAxis: {type: 'category'}, series: [{type: 'bar', encode: {x: 'v', y: 'k'}}]}), {}, 600);
  assert.ok(p.height >= 12 * 30, `height ${p.height}`);
});

test('from four lines up, each line has its own dash and marker; three keep the plain look', () => {
  const lines = (n) =>
    chart({
      dataset: {source: Array.from({length: n}, (_, i) => ({q: 'Q1', r: `s${i}`, v: i}))},
      xAxis: {type: 'category'},
      yAxis: {type: 'value'},
      series: [{type: 'line', seriesBy: 'r', encode: {x: 'q', y: 'v'}}],
    });
  const four = plan(lines(4)).option.series;
  assert.equal(new Set(four.map((s) => `${JSON.stringify(s.lineStyle.type)}${s.symbol}`)).size, 4);
  const three = plan(lines(3)).option.series;
  assert.ok(three.every((s) => s.lineStyle.type === undefined && s.symbol === undefined));
});

test('the palette an option asks for chooses the theme, and an author colour wins', () => {
  const spec = bars({});
  spec.option.palette = 'highlight';
  const p = plan(spec);
  assert.equal(p.theme, 'highlight');
  assert.equal(p.option.palette, undefined);
  assert.equal(p.option.series[0].itemStyle.color, palette.light.highlight[0]);
  const colored = bars({});
  colored.option.color = ['#111111', '#222222', '#333333'];
  assert.equal(plan(colored).option.series[1].itemStyle.color, '#222222');
  const bad = bars({});
  bad.option.palette = 'neon';
  assert.throws(() => plan(bad), /palette "neon"/);
});

test('a chart prepared twice gives the same answer and never changes its spec', () => {
  const spec = bars({});
  spec.filters = ['region'];
  spec.slicers = {region};
  const before = JSON.stringify(spec);
  const a = plan(spec, {region: '华东'}, 500);
  const b = plan(spec, {region: '华东'}, 500);
  assert.deepEqual(a, b);
  assert.equal(JSON.stringify(spec), before);
});

test('a series without rows to read passes through untouched, so any ECharts option still draws', () => {
  const p = plan(chart({series: [{type: 'gauge', data: [{value: 70, name: '完成率'}]}]}));
  assert.equal(p.empty, false);
  assert.deepEqual(p.option.series[0].data, [{value: 70, name: '完成率'}]);
});

test('no undefined survives into the option', () => {
  const p = plan(bars({}));
  assert.equal(JSON.stringify(p.option), JSON.stringify(JSON.parse(JSON.stringify(p.option))));
  const walk = (value) => {
    if (value && typeof value === 'object') Object.values(value).forEach(walk);
    else assert.notEqual(value, undefined);
  };
  walk(p.option);
});

test('the report themes drop the title and the grid margins and paint the card, not a page', () => {
  for (const name of Object.keys(themes)) {
    assert.equal(themes[name].title, undefined);
    assert.equal(themes[name].grid, undefined);
    assert.equal(themes[name].backgroundColor, 'transparent');
    assert.equal(themes[name].legend.top, undefined);
  }
  const dark = Core.themes(structure, palette.dark);
  assert.equal(dark.categorical.color[0], palette.dark.categorical[0]);
  assert.equal(dark.categorical.pie.itemStyle.borderColor, palette.dark.roles.surface);
});

test('filterValues lists the distinct values of the charts that name the slicer, or the fixed list', () => {
  const a = {...bars({}), filters: ['region']};
  const b = chart({dataset: {source: [{region: '西南'}]}, series: []}, {filters: ['region']});
  const c = chart({dataset: {source: [{region: '不相干'}]}, series: []});
  assert.deepEqual(Core.filterValues(region, [a, b, c]), ['华东', '华北', '华南', '西南']);
  assert.deepEqual(Core.filterValues({...region, values: ['z', 'y']}, [a]), ['z', 'y']);
});
