// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/**
 * The pure half of `html-report`: everything that turns one chart's spec, the slicers' state and the
 * figure's width into the final ECharts option. No DOM, so the browser runtime and the node preview
 * call the same `prepare` and cannot disagree about what a chart looks like.
 */
'use strict';

const Theme = require('../theme.js');

const {asList, isObject} = Theme;

// A figure's profile follows the width of its own container, never the viewport.
const PROFILES = [
  {name: 'narrow', below: 420, axis: 11, legend: 11, tooltip: 12, line: 2, symbol: 5, barMax: 24, gutter: 8},
  {name: 'medium', below: 720, axis: 12, legend: 12, tooltip: 12, line: 2.5, symbol: 6, barMax: 32, gutter: 12},
  {name: 'wide', below: Infinity, axis: 12, legend: 12, tooltip: 12, line: 2.5, symbol: 6, barMax: 36, gutter: 16},
];

// Series that read rows from a dataset, and series drawn on a cartesian grid.
const CARTESIAN = new Set(['bar', 'line', 'scatter', 'effectScatter']);
const GRIDDED = new Set([...CARTESIAN, 'heatmap']);
const AGGREGATES = new Set(['sum', 'avg', 'min', 'max', 'count', 'first', 'last']);
// A vertical bar chart turns horizontal on a narrow figure when its labels are long or many.
const LONG_LABEL = 10;
const MANY_CATEGORIES = 8;
// An axis with more points than this gets an inside zoom on a narrow figure.
const LONG_AXIS = 24;
const LEGEND_KEYS = ['top', 'left', 'right', 'bottom', 'x', 'y', 'width', 'height', 'orient', 'align'];
const GRID_KEYS = ['top', 'left', 'right', 'bottom', 'width', 'height', 'containLabel'];

function profileFor(width) {
  return PROFILES.find((p) => width < p.below);
}

const clone = (value) => (value === undefined ? undefined : JSON.parse(JSON.stringify(value)));
const isCjk = (ch) => /[\u1100-\u11ff\u2e80-\ua4cf\uac00-\ud7af\uf900-\ufaff\ufe30-\ufe4f\uff00-\uffef]/.test(ch);

/** Estimate a label's width: a CJK character is one em, a Latin one about 0.56 em. */
function textWidth(text, size) {
  let width = 0;
  for (const ch of String(text)) width += isCjk(ch) ? size : /[ ,.:;'|!i`l]/.test(ch) ? size * 0.32 : size * 0.58;
  return width;
}

function parseAspect(aspect) {
  const [w, h] = String(aspect ?? '16:9').split(':').map(Number);
  return w > 0 && h > 0 ? w / h : 16 / 9;
}

// ---- slicers ----------------------------------------------------------------------------------

/** The value a slicer holds before anyone touches it. */
function slicerDefault(def) {
  if (def.type === 'metric') return def.options[0].label;
  if (def.mode === 'multi') return [];
  return def.all === false && def.values?.length ? def.values[0] : null;
}

/** The slicer's value in `state`, or its default; a value `def` does not offer counts as unset. */
function slicerValue(def, state) {
  const value = state?.[def.id];
  if (value === undefined) return slicerDefault(def);
  if (def.type === 'metric') return def.options.some((o) => o.label === value) ? value : slicerDefault(def);
  return value;
}

/** The metric option the metric slicer in `spec.filters` has selected, or null. */
function selectedMetric(spec, state) {
  for (const id of spec.filters ?? []) {
    const def = spec.slicers?.[id];
    if (def?.type === 'metric') return def.options.find((o) => o.label === slicerValue(def, state));
  }
  return null;
}

/** Return the rows of a chart's dataset; the first dataset when there are several. */
function sourceRows(option) {
  const dataset = asList(option.dataset)[0];
  return Array.isArray(dataset?.source) ? dataset.source.filter(isObject) : [];
}

function filterRows(rows, spec, state) {
  const checks = [];
  for (const id of spec.filters ?? []) {
    const def = spec.slicers?.[id];
    if (def?.type === 'metric' || !def) continue;
    const value = slicerValue(def, state);
    if (def.mode === 'multi') {
      if (Array.isArray(value) && value.length) checks.push((row) => value.includes(String(row[def.field])));
    } else if (value !== null && value !== '') {
      checks.push((row) => String(row[def.field]) === String(value));
    }
  }
  return checks.length ? rows.filter((row) => checks.every((check) => check(row))) : rows;
}

/** The distinct values of `field` over `rows`, in order of first appearance. */
function distinct(rows, field) {
  const seen = new Set();
  const out = [];
  for (const row of rows) {
    const value = row[field];
    if (value === undefined || value === null || seen.has(String(value))) continue;
    seen.add(String(value));
    out.push(value);
  }
  return out;
}

// ---- tokens -----------------------------------------------------------------------------------

/** Replace `{metric}` in the option: by the metric's field inside `encode`, by its label elsewhere. */
function applyMetric(option, metric) {
  if (!metric) return;
  const label = (text) => (typeof text === 'string' ? text.replaceAll('{metric}', metric.label) : text);
  for (const title of asList(option.title)) {
    title.text = label(title.text);
    title.subtext = label(title.subtext);
  }
  for (const axis of [...asList(option.xAxis), ...asList(option.yAxis)]) {
    if (isObject(axis) && axis.name !== undefined) axis.name = label(axis.name);
  }
  for (const series of asList(option.series)) {
    if (!isObject(series)) continue;
    if (series.name !== undefined) series.name = label(series.name);
    for (const [key, value] of Object.entries(series.encode ?? {})) {
      if (value !== '{metric}') continue;
      series.encode[key] = metric.y;
      if (series.name === undefined && !series.seriesBy) series.name = metric.label;
    }
  }
}

// ---- tidy rows to series ----------------------------------------------------------------------

function axisOf(axes, series, key) {
  const list = asList(axes);
  return list[series[key] ?? 0] ?? {};
}

/** The axis type ECharts gives an axis that does not name one: x is a category axis, y a value axis. */
const axisType = (axis, dflt) => axis.type ?? dflt;

/**
 * How a series reads its rows. `key` is the dimension whose values place a mark (categories, times,
 * x), `measure` the one that sizes it; a horizontal bar chart keys on y.
 */
function classify(series, option) {
  if (!isObject(series) || !isObject(series.encode)) return null;
  if (series.type === 'pie' || series.type === 'funnel') {
    const {itemName, value} = series.encode;
    return itemName && value ? {kind: 'slice', nameField: itemName, valueField: value} : null;
  }
  if (!CARTESIAN.has(series.type)) return null;
  const x = axisType(axisOf(option.xAxis, series, 'xAxisIndex'), 'category');
  const y = axisType(axisOf(option.yAxis, series, 'yAxisIndex'), 'value');
  const {x: xField, y: yField} = series.encode;
  if (typeof xField !== 'string' || typeof yField !== 'string') return null;
  if (series.type === 'scatter' || series.type === 'effectScatter') {
    return {kind: 'points', xField, yField, nameField: series.encode.itemName};
  }
  if (x === 'category') return {kind: 'cartesian', axis: 'x', keyField: xField, measureField: yField, keyType: 'category'};
  if (y === 'category') return {kind: 'cartesian', axis: 'y', keyField: yField, measureField: xField, keyType: 'category'};
  return {kind: 'cartesian', axis: 'x', keyField: xField, measureField: yField, keyType: x};
}

function reduceValues(values, how) {
  const numbers = values.filter((v) => v !== null && v !== '' && Number.isFinite(Number(v))).map(Number);
  if (how === 'count') return values.length;
  if (!numbers.length) return null;
  switch (how) {
    case 'avg': return numbers.reduce((a, b) => a + b, 0) / numbers.length;
    case 'min': return Math.min(...numbers);
    case 'max': return Math.max(...numbers);
    case 'first': return numbers[0];
    case 'last': return numbers[numbers.length - 1];
    default: return numbers.reduce((a, b) => a + b, 0);
  }
}

const sortKey = (value, type) => (type === 'time' ? new Date(value).getTime() || Number(value) : Number(value));

/** Group rows by the string form of `field`; the insertion order of the Map is first appearance. */
function groupBy(rows, field) {
  const groups = new Map();
  for (const row of rows) {
    const key = String(row[field]);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return groups;
}

/**
 * Turn every series that names row fields in `encode` into series that carry their own data, and
 * drop the series `seriesBy` and the filters leave empty. Returns the kinds of axis it laid out.
 */
function expandSeries(option, rows, allRows, colors) {
  const list = asList(option.series).filter(isObject);
  const jobs = list.map((series) => ({series, job: classify(series, option)}));
  // Categories are shared by every series on the same axis, in order of first appearance.
  const categories = {x: null, y: null};
  for (const axis of ['x', 'y']) {
    const declared = asList(axis === 'x' ? option.xAxis : option.yAxis)[0];
    const users = jobs.filter(({job}) => job?.kind === 'cartesian' && job.axis === axis && job.keyType === 'category');
    if (!users.length) continue;
    if (Array.isArray(declared?.data)) {
      categories[axis] = declared.data.map((item) => (isObject(item) ? item.value ?? item.name : item));
    } else {
      const seen = [];
      for (const {job} of users) for (const v of distinct(rows, job.keyField)) if (!seen.some((s) => String(s) === String(v))) seen.push(v);
      categories[axis] = seen;
    }
  }
  // Time and value axes line the series up on the union of their x values, as category axes do.
  const sweeps = {x: null, y: null};
  for (const axis of ['x', 'y']) {
    const users = jobs.filter(({job}) => job?.kind === 'cartesian' && job.axis === axis && job.keyType !== 'category');
    if (!users.length) continue;
    const type = users[0].job.keyType;
    const seen = new Map();
    for (const {job} of users) for (const v of distinct(rows, job.keyField)) seen.set(String(v), v);
    sweeps[axis] = [...seen.values()]
      .map((v) => (type === 'value' || type === 'log' ? Number(v) : v))
      .sort((a, b) => sortKey(a, type) - sortKey(b, type));
  }
  const out = [];
  let slot = 0;
  for (const {series, job} of jobs) {
    if (!job) {
      out.push(series);
      slot += 1;
      continue;
    }
    const how = AGGREGATES.has(series.aggregate) ? series.aggregate : 'sum';
    const base = {...series};
    for (const key of ['encode', 'seriesBy', 'seriesOrder', 'aggregate']) delete base[key];
    if (job.kind === 'slice') {
      const names = distinct(allRows, job.nameField);
      const data = [];
      for (const [name, group] of groupBy(rows, job.nameField)) {
        const item = {name, value: reduceValues(group.map((r) => r[job.valueField]), how)};
        const at = names.findIndex((n) => String(n) === name);
        if (colors?.length && at >= 0) item.itemStyle = {color: colors[(slot + at) % colors.length]};
        data.push(item);
      }
      out.push({...base, name: base.name ?? job.valueField, data});
      slot += Math.max(names.length, 1);
      continue;
    }
    // One partition per seriesBy value, ordered as given or as the whole data first shows them.
    let parts;
    let globalOrder;
    if (series.seriesBy) {
      globalOrder = Array.isArray(series.seriesOrder) ? series.seriesOrder : distinct(allRows, series.seriesBy);
      const groups = groupBy(rows, series.seriesBy);
      parts = globalOrder
        .filter((value) => groups.has(String(value)))
        .map((value) => ({name: String(value), rows: groups.get(String(value)), at: globalOrder.indexOf(value)}));
    } else {
      globalOrder = [null];
      parts = [{name: base.name ?? job.measureField ?? job.yField, rows, at: 0}];
    }
    for (const part of parts) {
      const next = {...base, name: part.name};
      if (colors?.length && next.itemStyle?.color === undefined) {
        const color = colors[(slot + part.at) % colors.length];
        next.itemStyle = {...next.itemStyle, color};
        if (series.type === 'line') next.lineStyle = {...next.lineStyle, color};
      }
      if (job.kind === 'points') {
        next.data = part.rows.map((row) => {
          const value = [Number(row[job.xField]), Number(row[job.yField])];
          return job.nameField ? {name: String(row[job.nameField]), value} : value;
        });
      } else if (job.keyType === 'category') {
        const cells = groupBy(part.rows, job.keyField);
        next.data = categories[job.axis].map((c) => {
          const cell = cells.get(String(c));
          return cell ? reduceValues(cell.map((r) => r[job.measureField]), how) : null;
        });
      } else {
        const cells = groupBy(part.rows, job.keyField);
        next.data = sweeps[job.axis].map((k) => {
          const cell = cells.get(String(k));
          return [k, cell ? reduceValues(cell.map((r) => r[job.measureField]), how) : null];
        });
      }
      out.push(next);
    }
    slot += globalOrder.length;
  }
  for (const axis of ['x', 'y']) {
    if (!categories[axis]) continue;
    const key = axis === 'x' ? 'xAxis' : 'yAxis';
    const axes = asList(option[key]).map((a, i) => (i === 0 ? {...a, data: categories[axis]} : a));
    option[key] = Array.isArray(option[key]) ? axes : axes[0];
  }
  option.series = out;
  return categories;
}

// ---- layout -----------------------------------------------------------------------------------

const hasData = (series) =>
  asList(series.data).some((item) => {
    const value = isObject(item) && 'value' in item ? item.value : item;
    return Array.isArray(value) ? value.some((v) => v !== null && v !== undefined) : value !== null && value !== undefined;
  });

/** How many rows a legend of `names` wraps onto in `width` pixels. */
function legendRows(names, width, size) {
  let rows = 1;
  let x = 0;
  for (const name of names) {
    const item = 12 + 6 + textWidth(name, size) + 14;
    if (x > 0 && x + item > width) {
      rows += 1;
      x = 0;
    }
    x += item;
  }
  return rows;
}

function seriesNames(series) {
  return series.flatMap((s) => (s.type === 'pie' || s.type === 'funnel' ? asList(s.data).map((d) => d.name) : [s.name])).filter((n) => n !== undefined);
}

/** Flip a vertical bar chart to horizontal, first category on top, and return true when it did. */
function flipHorizontal(option) {
  const x = asList(option.xAxis);
  const y = asList(option.yAxis);
  if (x.length !== 1 || y.length !== 1 || axisType(x[0], 'category') !== 'category' || axisType(y[0], 'value') !== 'value') return false;
  const series = asList(option.series);
  if (!series.length || !series.every((s) => isObject(s) && s.type === 'bar')) return false;
  option.xAxis = {...y[0], type: 'value'};
  option.yAxis = {...x[0], type: 'category', inverse: true};
  const edge = {top: 'right', insideTop: 'insideRight', bottom: 'left', insideBottom: 'insideLeft'};
  option.series = series.map((s) => {
    const flipped = {...s, itemStyle: {borderRadius: [0, 4, 4, 0], ...s.itemStyle}};
    if (s.label?.position in edge) flipped.label = {...s.label, position: edge[s.label.position]};
    return flipped;
  });
  return true;
}

function describe(option) {
  const series = asList(option.series).filter(isObject);
  const types = new Set(series.map((s) => s.type));
  const cartesian = series.some((s) => GRIDDED.has(s.type)) && option.xAxis !== undefined;
  const circular = types.has('pie') || types.has('funnel');
  return {series, types, cartesian, circular, onlyBars: series.length > 0 && series.every((s) => s.type === 'bar')};
}

/** The tooltip an author who wrote none would have wanted; the report is read by hovering and tapping. */
function defaultTooltip(info) {
  const points = [...info.types].every((t) => t === 'scatter' || t === 'effectScatter' || t === 'heatmap');
  if (info.circular || !info.cartesian || points) return {trigger: 'item'};
  return {trigger: 'axis', axisPointer: {type: info.onlyBars ? 'shadow' : 'line'}};
}

/**
 * Lay the option out for the figure's profile: the report owns margins, legend placement, label
 * strategy and text sizes, because the author wrote an option for an 800 x 500 picture.
 */
function layout(option, ctx) {
  const {profile, width, categories} = ctx;
  const info = describe(option);
  const gutter = profile.gutter;
  const narrow = profile.name === 'narrow';
  const horizontal = ctx.horizontal;

  // Legend: top left on a wide figure, along the bottom of a narrow one; it wraps, and scrolls past a few rows.
  const names = seriesNames(info.series);
  let legend = option.legend === false ? {show: false} : isObject(option.legend) ? {...option.legend} : null;
  if (legend === null && (info.series.length > 1 || info.circular)) legend = {};
  let legendHeight = 0;
  let legendBottom = false;
  if (legend && legend.show !== false) {
    for (const key of LEGEND_KEYS) delete legend[key];
    const wanted = legendRows(names, width - 2 * gutter, profile.legend);
    const cap = narrow ? 6 : 3;
    const rows = Math.min(wanted, cap);
    legendHeight = rows * 20 + 4;
    legendBottom = narrow;
    option.legend = {
      ...legend,
      type: wanted > cap ? 'scroll' : 'plain',
      [legendBottom ? 'bottom' : 'top']: legendBottom ? 4 : 0,
      left: gutter - 4,
      right: gutter,
      itemWidth: 12,
      itemHeight: 8,
      itemGap: 14,
      icon: 'roundRect',
      textStyle: {fontSize: profile.legend},
      pageIconSize: 10,
      pageTextStyle: {fontSize: profile.legend},
    };
  } else {
    option.legend = {show: false};
  }
  // A visual map sits along the bottom, small, instead of down the side where the plot needs the room.
  const mapped = asList(option.visualMap).some((v) => isObject(v) && v.show !== false);
  const topReserve = legendBottom || !legendHeight ? 0 : legendHeight + 4;
  const bottomReserve = (legendBottom ? legendHeight + 4 : 0) + (mapped ? 40 : 0);
  if (mapped) {
    const place = (v) => {
      if (!isObject(v) || v.show === false) return v;
      const {top, left, right, bottom, orient, itemHeight, ...rest} = v;
      return {
        ...rest,
        orient: 'horizontal',
        left: 'center',
        bottom: legendBottom ? legendHeight + 8 : 4,
        itemWidth: 12,
        itemHeight: Math.min(160, Math.round(width * 0.4)),
        textStyle: {fontSize: profile.legend, ...v.textStyle},
      };
    };
    option.visualMap = Array.isArray(option.visualMap) ? option.visualMap.map(place) : place(option.visualMap);
  }

  if (info.circular) {
    const pieLabels = !narrow && option.series.every((s) => s.type !== 'pie' || s.label?.show !== false);
    option.series = option.series.map((s) => {
      if (s.type !== 'pie') return s;
      const pie = {
        ...s,
        label: pieLabels ? {fontSize: profile.axis, formatter: '{d}%', ...s.label} : {show: false},
      };
      if (!pieLabels) pie.labelLine = {show: false};
      return pie;
    });
    delete option.grid;
    return {legendHeight, topReserve, bottomReserve, pieLabels};
  }
  if (!info.cartesian) return {legendHeight, topReserve, bottomReserve};

  // Grid: the margins hold the legend, and ECharts keeps the axis labels inside the rest.
  const grid = isObject(asList(option.grid)[0]) ? {...asList(option.grid)[0]} : {};
  for (const key of GRID_KEYS) delete grid[key];
  const namesOnAxes = [...asList(option.xAxis), ...asList(option.yAxis)].some((a) => isObject(a) && a.name);
  option.grid = {
    ...grid,
    left: gutter,
    right: gutter + 4,
    top: gutter + topReserve + (namesOnAxes ? 6 : 0),
    bottom: gutter + bottomReserve,
    outerBoundsMode: 'same',
    outerBoundsContain: 'all',
    outerBoundsClampWidth: '60%',
  };

  // Axes: label size, a width that wrapping can honour, and no overlap ever.
  const labelBudget = Math.max(64, Math.round(width * (horizontal ? (narrow ? 0.3 : 0.28) : 1)));
  const style = {fontSize: profile.axis};
  const bandWidth = categories?.x?.length ? Math.max(24, ((width - 2 * gutter - 48) / categories.x.length) * 0.84) : null;
  const fixAxis = (axis, dim) => {
    if (!isObject(axis)) return axis;
    const type = axisType(axis, dim === 'x' ? 'category' : 'value');
    const out = {...axis, nameTextStyle: {fontSize: profile.axis, ...axis.nameTextStyle}};
    const label = {...axis.axisLabel, ...style, hideOverlap: true};
    if (type === 'category') {
      if (dim === 'y') {
        Object.assign(label, {width: labelBudget, overflow: 'break', lineHeight: profile.axis + 3, interval: 0});
      } else if (bandWidth !== null && bandWidth >= profile.axis * 3 && categories.x.every((c) => textWidth(c, profile.axis) <= bandWidth * 2)) {
        Object.assign(label, {width: bandWidth, overflow: 'break', lineHeight: profile.axis + 3, interval: 0});
      } else {
        label.interval ??= 'auto';
      }
      out.axisTick = {show: false};
    }
    out.axisLabel = label;
    return out;
  };
  option.xAxis = Array.isArray(option.xAxis) ? option.xAxis.map((a) => fixAxis(a, 'x')) : fixAxis(option.xAxis, 'x');
  option.yAxis = Array.isArray(option.yAxis) ? option.yAxis.map((a) => fixAxis(a, 'y')) : fixAxis(option.yAxis, 'y');
  // Horizontal bars read from the top, in the order the data gives.
  if (horizontal && isObject(option.yAxis) && option.yAxis.inverse === undefined) option.yAxis.inverse = true;

  // Marks: lines and symbols follow the profile, bars never fill the slot.
  option.series = option.series.map((s) => {
    if (s.type === 'line') {
      return {...s, lineStyle: {width: profile.line, ...s.lineStyle}, symbolSize: s.symbolSize ?? profile.symbol};
    }
    if (s.type === 'bar') return {...s, barMaxWidth: s.barMaxWidth ?? profile.barMax};
    return s;
  });

  // A long axis on a narrow figure can be pinched open.
  const points = Math.max(0, ...option.series.filter((s) => CARTESIAN.has(s.type)).map((s) => asList(s.data).length));
  if (narrow && points > LONG_AXIS && !option.dataZoom && !horizontal) {
    option.dataZoom = [{type: 'inside', xAxisIndex: 0, filterMode: 'none', zoomLock: false}];
  }
  return {legendHeight, topReserve, bottomReserve};
}

/**
 * Seat each pie in the box the legend leaves: centred there, as large as its outside labels allow.
 * Pixels, not percentages, so a legend that takes two rows moves the pie, not the labels over it.
 */
function placePies(option, {width, height, top, bottom, labels}) {
  const free = height - top - bottom;
  const outer = Math.max(40, Math.min((width - (labels ? 112 : 16)) / 2, (free - (labels ? 44 : 16)) / 2));
  option.series = option.series.map((s) => {
    if (s.type !== 'pie') return s;
    return {
      ...s,
      radius: s.radius ?? [Math.round(outer * 0.6), Math.round(outer)],
      center: s.center ?? [Math.round(width / 2), Math.round(top + free / 2)],
    };
  });
}

/** The height of the chart body: from the aspect on the figure's width, never below its minimum. */
function chartHeight(spec, option, ctx) {
  const {profile, width, horizontal, categories, reserve} = ctx;
  const info = describe(option);
  const minHeight = Number(spec.minHeight) > 0 ? Number(spec.minHeight) : 240;
  const ratio = parseAspect(spec.aspect);
  const narrowRatio = info.circular ? 1.05 : Math.min(ratio, 1.15);
  const mediumRatio = Math.min(ratio, 1.55);
  let height = width / (profile.name === 'narrow' ? narrowRatio : profile.name === 'medium' ? mediumRatio : ratio);
  height = Math.min(height, profile.name === 'wide' ? 440 : 460);
  if (horizontal) {
    const count = categories?.y?.length ?? 0;
    const series = info.series.filter((s) => s.type === 'bar').length;
    const stacked = info.series.some((s) => s.stack);
    const band = (stacked ? 1 : Math.max(series, 1)) * 14 + 22;
    height = Math.max(height, count * band + 2 * profile.gutter + reserve + 36);
  }
  return Math.round(Math.max(height, minHeight));
}

/**
 * Build everything the runtime needs to draw one chart at one width:
 * `{option, height, theme, caption, empty, profile, horizontal}`.
 *
 * `spec` is the chart block plus the definitions of the slicers it lists (`spec.slicers`), `state`
 * maps slicer ids to values, and `env.themes` is what `themes()` built: its colour list gives each
 * `seriesBy` series a colour that does not change when a filter removes another series.
 */
function prepare(spec, state = {}, width = 720, env = {}) {
  const profile = profileFor(width);
  const {name: paletteName, option} = Theme.pick(clone(spec.option ?? {}));

  applyMetric(option, selectedMetric(spec, state));
  const allRows = sourceRows(option);
  const rows = filterRows(allRows, spec, state);
  const colors = option.color ?? env.themes?.[paletteName]?.color;
  const categories = expandSeries(option, rows, allRows, colors);
  if (option.dataset && option.series.every((s) => s.data !== undefined)) delete option.dataset;
  else if (option.dataset) asList(option.dataset).forEach((d, i) => i === 0 && (d.source = rows));

  // Caption: the title leaves the chart and becomes HTML, so it wraps like text.
  const title = asList(option.title)[0] ?? {};
  const caption = title.text || title.subtext ? {title: title.text ?? '', subtitle: title.subtext ?? ''} : null;
  delete option.title;
  delete option.toolbox;
  delete option.backgroundColor;
  delete option.palette;

  const series = option.series.filter(isObject);
  const empty = series.length === 0 || (rows.length === 0 && allRows.length > 0) || !series.some(hasData);

  // Vertical bars lie down when their labels are long or many on a narrow figure, or cannot wrap into two lines.
  const info = describe(option);
  let horizontal = axisType(asList(option.yAxis)[0] ?? {}, 'value') === 'category' && info.cartesian && info.onlyBars;
  let finalCategories = categories;
  if (!horizontal && info.onlyBars && spec.orient !== 'keep') {
    const labels = categories.x ?? asList(asList(option.xAxis)[0]?.data);
    const longest = Math.max(0, ...labels.map((c) => [...String(c)].length));
    const band = Math.max(1, (width - 2 * profile.gutter - 48) / Math.max(labels.length, 1) - 6);
    const lines = Math.max(0, ...labels.map((c) => Math.ceil(textWidth(c, profile.axis) / band)));
    if ((profile.name === 'narrow' && (labels.length > MANY_CATEGORIES || longest > LONG_LABEL)) || lines > 2) {
      horizontal = flipHorizontal(option);
      finalCategories = {x: null, y: labels};
    }
  }
  const reserve = layout(option, {profile, width, categories: finalCategories, horizontal});
  if (option.tooltip === undefined) option.tooltip = defaultTooltip(describe(option));
  option.tooltip = {...option.tooltip, confine: true, textStyle: {fontSize: profile.tooltip, ...option.tooltip.textStyle}};

  const height = chartHeight(spec, option, {
    profile,
    width,
    horizontal,
    categories: finalCategories,
    reserve: reserve.topReserve + reserve.bottomReserve,
  });
  if (describe(option).circular) {
    placePies(option, {width, height, top: reserve.topReserve, bottom: reserve.bottomReserve, labels: reserve.pieLabels});
  }
  return {
    option: Theme.decorate(clone(option), paletteName),
    height,
    theme: paletteName,
    caption,
    empty,
    profile: profile.name,
    horizontal,
  };
}

/** Every distinct value of `field` over the rows of the charts that list a filter slicer. */
function filterValues(def, charts) {
  if (Array.isArray(def.values)) return def.values.map(String);
  const values = [];
  for (const chart of charts) {
    if (!(chart.filters ?? []).includes(def.id)) continue;
    for (const value of distinct(sourceRows(chart.option ?? {}), def.field)) {
      if (!values.includes(String(value))) values.push(String(value));
    }
  }
  return values;
}

/** The report's chart themes for one palette mode: the house theme, minus what an embedded chart does not need. */
function themes(structure, colors) {
  const roles = {...colors.roles, background: colors.roles.surface, gap: colors.roles.surface};
  const built = Theme.build(structure, {...colors, roles});
  const out = {};
  for (const [name, theme] of Object.entries(built)) {
    const {title, grid, ...rest} = theme;
    const {top, left, ...legend} = rest.legend;
    out[name] = Theme.merge({...rest, legend}, {
      backgroundColor: 'transparent',
      legend: {itemWidth: 12, itemHeight: 8, itemGap: 14, textStyle: {fontSize: 12}},
      categoryAxis: {axisLabel: {fontSize: 12, margin: 8}, nameTextStyle: {fontSize: 12}},
      valueAxis: {axisLabel: {fontSize: 12, margin: 8}, nameTextStyle: {fontSize: 12}, nameGap: 10},
      timeAxis: {axisLabel: {fontSize: 12, margin: 8}},
      logAxis: {axisLabel: {fontSize: 12}},
      line: {symbolSize: 6, lineStyle: {width: 2.5}},
      pie: {label: {fontSize: 12}},
      tooltip: {
        padding: [8, 10],
        extraCssText: `border-radius:10px;box-shadow:0 12px 34px ${roles.tooltipShadow};`,
        textStyle: {fontSize: 12},
      },
    });
  }
  return out;
}

module.exports = {
  AGGREGATES,
  PROFILES,
  distinct,
  filterRows,
  filterValues,
  prepare,
  profileFor,
  slicerDefault,
  slicerValue,
  sourceRows,
  textWidth,
  themes,
};
