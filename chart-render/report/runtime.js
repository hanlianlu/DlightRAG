// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/**
 * The `html-report` browser runtime. `html_report.py` inlines it as the body of one function, after
 * theme.js and core.js, with `require` serving those two and `DATA` carrying theme.json and
 * palette.json; the only global it adds is `window.Report`. It hydrates the author's fragment
 * (slicers, pages, charts, tables) once the document is parsed and keeps every chart laid out for
 * the width its own figure has.
 *
 * It runs inside the product's sandbox: no storage, no dialogs, no network, no popups.
 */
const Core = require('core.js');

const doc = document;
const root = doc.documentElement;
const win = window;

// ---- language ---------------------------------------------------------------------------------

// The build sets `lang` from the fragment's own text, so the controls speak the report's language.
const isChinese = () => (root.getAttribute('lang') || '').toLowerCase().startsWith('zh');
const STRINGS = {
  zh: {all: '全部', reset: '重置', filters: '筛选', empty: '没有符合当前筛选的数据', insight: '洞察', caution: '注意', risk: '风险', note: '说明'},
  en: {all: 'All', reset: 'Reset', filters: 'Filters', empty: 'No data matches the current filters', insight: 'Insight', caution: 'Caution', risk: 'Risk', note: 'Note'},
};
let text = STRINGS.en;

// ---- state ------------------------------------------------------------------------------------

const state = {}; // slicer id -> value
const slicers = new Map(); // id -> {def, el, sync()}
const charts = new Map(); // id -> record
const pages = []; // {id, title, el, tab}
const waiting = [];
let hydrated = false;
let themes = null;
let mode = 'light';

const colorQuery = win.matchMedia ? win.matchMedia('(prefers-color-scheme: dark)') : null;
const motionQuery = win.matchMedia ? win.matchMedia('(prefers-reduced-motion: reduce)') : null;
const printQuery = win.matchMedia ? win.matchMedia('print') : null;

function buildThemes() {
  mode = colorQuery && colorQuery.matches ? 'dark' : 'light';
  themes = Core.themes(DATA.structure, DATA.palette[mode]);
  for (const [name, theme] of Object.entries(themes)) echarts.registerTheme(`report-${mode}-${name}`, theme);
}

const el = (tag, attrs = {}, children = []) => {
  const node = doc.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'text') node.textContent = value;
    else if (value !== undefined && value !== null) node.setAttribute(key, value);
  }
  for (const child of children) node.append(child);
  return node;
};

// ---- slicers ----------------------------------------------------------------------------------

function parseJson(raw, fallback) {
  try {
    return JSON.parse(raw);
  } catch (error) {
    return fallback;
  }
}

const attributeNumber = (node, name) => {
  const raw = node.getAttribute(name);
  return raw === null || raw.trim() === '' ? NaN : Number(raw);
};

function readSlicerDef(node) {
  const id = node.getAttribute('data-slicer');
  if (!id) return null;
  const kind = node.getAttribute('data-type');
  const type = kind === 'metric' || kind === 'slider' ? kind : 'filter';
  const def = {id, type, label: node.getAttribute('data-label') || id, ui: node.getAttribute('data-ui') || 'auto'};
  if (type === 'slider') {
    const [min, max, step] = ['data-min', 'data-max', 'data-step'].map((name) => attributeNumber(node, name));
    if (![min, max, step].every(Number.isFinite) || !(max > min) || !(step > 0)) return null;
    const first = attributeNumber(node, 'data-value');
    return {...def, mode: 'single', min, max, step, unit: node.getAttribute('data-unit') || '', value: Number.isFinite(first) ? first : min};
  }
  if (type === 'metric') {
    def.options = parseJson(node.getAttribute('data-options'), []).filter((o) => o && o.label && o.y);
    if (!def.options.length) return null;
    def.mode = 'single';
    return def;
  }
  def.field = node.getAttribute('data-field');
  def.mode = node.getAttribute('data-mode') === 'multi' ? 'multi' : 'single';
  def.all = node.getAttribute('data-all') !== 'false';
  const values = node.getAttribute('data-values');
  if (values) def.values = parseJson(values, []).map(String);
  return def.field || def.values ? def : null;
}

const copy = (value) => (Array.isArray(value) ? [...value] : value);

function emitSlicer(id) {
  doc.dispatchEvent(new CustomEvent('report:slicer', {detail: {id, value: copy(state[id])}}));
}

// A slider moves many times a frame; its event goes out once a frame, with the latest value.
const pendingEvents = new Set();
function emitLater(id) {
  if (!pendingEvents.size) {
    win.requestAnimationFrame(() => {
      const ids = [...pendingEvents];
      pendingEvents.clear();
      for (const pending of ids) emitSlicer(pending);
    });
  }
  pendingEvents.add(id);
}

/** The charts a slicer decides: the ones that list it in "filters" and the ones whose text holds its {id}. */
const watchers = (id) => [...charts.values()].filter((record) => record.watching.has(id));

function setSlicer(id, value) {
  const entry = slicers.get(id);
  if (!entry) return;
  const {def} = entry;
  let next;
  if (def.type === 'metric') {
    const match = def.options.find((o) => o.label === value || o.y === value);
    if (!match) return;
    next = match.label;
  } else if (def.type === 'slider') {
    next = Core.snapSlider(def, value);
    if (next === null) return;
  } else if (def.mode === 'multi') {
    next = (Array.isArray(value) ? value : value === null || value === '' ? [] : [value]).map(String);
    if (def.values) next = next.filter((v) => def.values.includes(v));
  } else {
    next = value === undefined || value === null || value === '' ? null : String(value);
    if (next !== null && def.values && !def.values.includes(next)) return;
    if (next === null && def.all === false) next = Core.slicerDefault(def);
  }
  const changed = JSON.stringify(state[id]) !== JSON.stringify(next);
  state[id] = next;
  entry.sync();
  if (!changed) return;
  const live = def.type === 'slider';
  for (const record of watchers(id)) {
    // A slider is dragged: each frame replaces the last at once, with nothing animating in between.
    record.instant ||= live;
    scheduleRender(record);
  }
  if (live) emitLater(id);
  else emitSlicer(id);
}

function differsFromDefault(def, value) {
  const dflt = Core.slicerDefault(def);
  return Array.isArray(value) ? value.length > 0 : value !== dflt;
}

/** Mark which edges of a scrolling row have more behind them, so the stylesheet can fade them. */
function watchOverflow(row) {
  const mark = () => {
    const room = row.scrollWidth - row.clientWidth;
    const flags = [];
    if (room > 1 && row.scrollLeft > 1) flags.push('start');
    if (room > 1 && row.scrollLeft < room - 1) flags.push('end');
    row.setAttribute('data-overflow', flags.join(' '));
  };
  row.addEventListener('scroll', mark, {passive: true});
  if (win.ResizeObserver) new win.ResizeObserver(mark).observe(row);
  win.requestAnimationFrame(mark);
}

/** A real range input with its label, its current value and its two ends; the fill follows the thumb. */
function buildSlider(node, def) {
  const labelId = `slicer-${def.id}-label`;
  node.textContent = '';
  node.classList.add('slicer-ready');
  const label = el('span', {class: 'slicer-label', id: labelId, text: def.label});
  const reset = el('button', {type: 'button', class: 'slicer-reset', text: text.reset});
  const shown = el('span', {class: 'slicer-value', 'aria-hidden': 'true'});
  const input = el('input', {
    type: 'range',
    class: 'slicer-range',
    'aria-labelledby': labelId,
    min: String(def.min),
    max: String(def.max),
    step: String(def.step),
  });
  const ends = el('span', {class: 'slicer-ends', 'aria-hidden': 'true'}, [
    el('span', {text: Core.formatSlider(def, def.min)}),
    el('span', {text: Core.formatSlider(def, Core.snapSlider(def, def.max))}),
  ]);
  input.addEventListener('input', () => setSlicer(def.id, input.valueAsNumber));
  reset.addEventListener('click', () => setSlicer(def.id, Core.slicerDefault(def)));
  node.append(label, reset, shown, input, ends);
  const entry = {
    def,
    el: node,
    sync() {
      const value = state[def.id];
      const words = Core.formatSlider(def, value);
      input.value = String(value);
      input.setAttribute('aria-valuetext', words);
      input.style.setProperty('--ratio', String((value - def.min) / (def.max - def.min)));
      shown.textContent = words;
      reset.classList.toggle('on', differsFromDefault(def, value));
    },
  };
  slicers.set(def.id, entry);
  entry.sync();
}

function buildSlicer(node, def) {
  if (def.type === 'slider') return buildSlider(node, def);
  const choices = def.type === 'metric' ? def.options.map((o) => o.label) : Core.filterValues(def, [...charts.values()].map((c) => c.spec));
  const useSelect = def.ui === 'select' ? def.mode !== 'multi' : def.ui === 'chips' ? false : def.mode === 'single' && choices.length > 6;
  const labelId = `slicer-${def.id}-label`;
  node.textContent = '';
  node.classList.add('slicer-ready');
  const label = el('span', {class: 'slicer-label', id: labelId, text: def.label});
  const reset = el('button', {type: 'button', class: 'slicer-reset', text: text.reset});
  reset.addEventListener('click', () => setSlicer(def.id, Core.slicerDefault(def)));
  let control;
  let sync;
  if (useSelect) {
    const select = el('select', {class: 'slicer-select', 'aria-labelledby': labelId});
    const options = [];
    if (def.type === 'filter' && def.all) options.push({value: '', label: text.all});
    for (const choice of choices) options.push({value: choice, label: choice});
    for (const option of options) select.append(el('option', {value: option.value, text: option.label}));
    select.addEventListener('change', () => setSlicer(def.id, select.value === '' ? null : select.value));
    control = el('div', {class: 'slicer-control'}, [select]);
    sync = () => {
      const value = state[def.id];
      select.value = value === null || value === undefined ? '' : String(value);
    };
  } else {
    const group = el('div', {class: 'chips', role: 'group', 'aria-labelledby': labelId});
    const chips = [];
    const add = (value, labelText, isAll = false) => {
      const chip = el('button', {type: 'button', class: 'chip', text: labelText, 'data-label': labelText, 'aria-pressed': 'false'});
      chip.dataset.value = isAll ? '' : String(value);
      chip.addEventListener('click', () => {
        if (isAll) return setSlicer(def.id, def.mode === 'multi' ? [] : null);
        if (def.mode === 'multi') {
          const current = state[def.id] || [];
          const next = current.includes(String(value)) ? current.filter((v) => v !== String(value)) : [...current, String(value)];
          return setSlicer(def.id, next);
        }
        return setSlicer(def.id, value);
      });
      group.append(chip);
      chips.push({chip, value, isAll});
    };
    if (def.type === 'filter' && def.all) add(null, text.all, true);
    for (const choice of choices) add(choice, choice);
    control = el('div', {class: 'slicer-control'}, [group]);
    watchOverflow(group);
    sync = () => {
      const value = state[def.id];
      for (const {chip, value: v, isAll} of chips) {
        const pressed = isAll
          ? Array.isArray(value) ? value.length === 0 : value === null
          : Array.isArray(value) ? value.includes(String(v)) : String(value) === String(v);
        chip.setAttribute('aria-pressed', pressed ? 'true' : 'false');
      }
    };
  }
  node.append(label, control, reset);
  const entry = {
    def,
    el: node,
    sync() {
      sync();
      reset.classList.toggle('on', differsFromDefault(def, state[def.id]));
    },
  };
  slicers.set(def.id, entry);
  entry.sync();
}

function readSlicers() {
  const specs = [...charts.values()].map((c) => c.spec);
  for (const node of doc.querySelectorAll('.slicer[data-slicer]')) {
    const def = readSlicerDef(node);
    if (!def) continue;
    state[def.id] = Core.slicerDefault(def);
    node.dataset.reportSlicer = def.id;
    // The chart specs name slicers by id; give each its definition.
    for (const spec of specs) if (spec.filters.includes(def.id)) spec.slicers[def.id] = def;
    // Choices are read from the charts' rows, so build the controls after the specs know the defs.
    slicers.set(def.id, {def, el: node, sync() {}});
  }
  for (const {el: node, def} of [...slicers.values()]) buildSlicer(node, def);
  // A chart redraws for the slicers it lists and for the ones its text names with {id}.
  const known = [...slicers.keys()];
  for (const record of charts.values()) {
    record.watching = new Set([...record.spec.filters, ...Core.tokenIds(record.spec.option, known)]);
  }
  // A page-level slicer stays in view: wrap runs of them in a sticky bar. A slider is too tall for
  // that and stays where the author put it.
  const pageLevel = [...doc.querySelectorAll('.slicer.slicer-ready')].filter(
    (n) => n.getAttribute('data-type') !== 'slider' && !n.closest('.card, figure, .callout, .kpis, .timeline, table'),
  );
  for (const node of pageLevel) {
    if (node.parentElement && node.parentElement.classList.contains('slicer-bar')) continue;
    const bar = el('div', {class: 'slicer-bar', role: 'group', 'aria-label': text.filters});
    node.before(bar);
    bar.append(node);
    let next = bar.nextElementSibling;
    while (next && next.classList.contains('slicer') && next.classList.contains('slicer-ready')) {
      const following = next.nextElementSibling;
      bar.append(next);
      next = following;
    }
  }
}

// ---- charts -----------------------------------------------------------------------------------

/** Replace each `dataset.from` of an option by the rows of the shared `data-*` block it names. */
function resolveDatasets(option, datasets) {
  const entries = Array.isArray(option.dataset) ? option.dataset : option.dataset ? [option.dataset] : [];
  for (const entry of entries) {
    if (!entry || typeof entry !== 'object' || !('from' in entry)) continue;
    const rows = datasets[entry.from];
    if (!Array.isArray(rows)) {
      throw new Error(`the data block data-${entry.from} is missing or is not a list of rows`);
    }
    delete entry.from;
    entry.source = rows;
  }
}

function readCharts() {
  const datasets = {};
  for (const node of doc.querySelectorAll('script[type="application/json"][id^="data-"]')) {
    datasets[node.id.slice(5)] = parseJson(node.textContent, null);
  }
  for (const figure of doc.querySelectorAll('figure[data-chart]')) {
    const id = figure.getAttribute('data-chart');
    const script = doc.getElementById(`chart-${id}`);
    const block = script ? parseJson(script.textContent, null) : null;
    const record = {
      id,
      figure,
      body: null,
      caption: null,
      emptyNode: null,
      errorNode: null,
      instance: null,
      themeKey: null,
      width: 0,
      dirty: true,
      fatal: false,
      instant: false,
      watching: new Set(),
      frame: 0,
      spec: {
        id,
        filters: block && Array.isArray(block.filters) ? block.filters : [],
        aspect: block && block.aspect,
        minHeight: block && block.minHeight,
        orient: block && block.orient,
        option: block && block.option ? block.option : {},
        slicers: {},
      },
    };
    record.body = el('div', {class: 'chart-body', role: 'img'});
    record.emptyNode = el('div', {class: 'chart-empty', hidden: ''}, [el('span', {text: text.empty})]);
    record.errorNode = el('div', {class: 'chart-error', hidden: '', role: 'alert'});
    record.body.append(record.emptyNode);
    figure.prepend(record.body);
    figure.append(record.errorNode);
    charts.set(id, record);
    if (!block) {
      record.fatal = true;
      showError(record, new Error(`the JSON block chart-${id} is missing or is not valid JSON`));
    } else {
      try {
        resolveDatasets(record.spec.option, datasets);
      } catch (error) {
        record.fatal = true;
        showError(record, error);
      }
    }
  }
}

function showError(record, error) {
  record.errorNode.textContent = `This chart could not be drawn: ${error && error.message ? error.message : error}`;
  record.errorNode.hidden = false;
  record.body.classList.add('failed');
}

const isVisible = (record) => record.figure.offsetParent !== null && record.body.clientWidth > 0;

function scheduleRender(record) {
  record.dirty = true;
  if (record.frame) return;
  record.frame = win.requestAnimationFrame(() => {
    record.frame = 0;
    renderChart(record);
  });
}

function setCaption(record, caption) {
  let node = record.caption;
  if (!caption) {
    if (node) node.remove();
    record.caption = null;
    return;
  }
  if (!node) {
    node = el('figcaption', {class: 'chart-caption'}, [el('span', {class: 'chart-title'}), el('span', {class: 'chart-subtitle'})]);
    record.figure.prepend(node);
    record.caption = node;
  }
  node.firstChild.textContent = caption.title;
  node.lastChild.textContent = caption.subtitle;
  node.lastChild.hidden = !caption.subtitle;
  node.firstChild.hidden = !caption.title;
  record.body.setAttribute('aria-label', [caption.title, caption.subtitle].filter(Boolean).join(' · '));
}

function ensureInstance(record, paletteName) {
  const key = `report-${mode}-${paletteName}`;
  if (record.instance && record.themeKey === key) return;
  if (record.instance) record.instance.dispose();
  record.instance = echarts.init(record.body, key, {renderer: 'svg'});
  record.themeKey = key;
  // The empty-state node lives in the body; ECharts only appends its own layers.
  if (!record.emptyNode.isConnected) record.body.append(record.emptyNode);
}

function renderChart(record) {
  if (record.fatal || !record.dirty || !isVisible(record)) return;
  const width = Math.floor(record.body.clientWidth);
  try {
    const slicerDefs = Object.fromEntries([...slicers].map(([slicerId, entry]) => [slicerId, entry.def]));
    const plan = Core.prepare(record.spec, state, width, {themes, slicers: slicerDefs, lang: isChinese() ? 'zh' : 'en'});
    record.dirty = false;
    record.width = width;
    record.errorNode.hidden = true;
    record.body.classList.remove('failed');
    record.figure.dataset.profile = plan.profile;
    setCaption(record, plan.caption);
    record.body.style.height = `${plan.height}px`;
    ensureInstance(record, plan.theme);
    record.emptyNode.hidden = !plan.empty;
    record.body.classList.toggle('empty', plan.empty);
    if (plan.empty) {
      record.instance.clear();
    } else {
      const reduced = motionQuery && motionQuery.matches;
      record.instance.setOption(
        {...plan.option, animation: !reduced && !record.instant, animationDuration: 320, animationDurationUpdate: 240},
        {notMerge: true},
      );
      record.instant = false;
    }
    record.instance.resize({width, height: plan.height});
  } catch (error) {
    record.dirty = false;
    showError(record, error);
  }
}

function renderAll() {
  for (const record of charts.values()) scheduleRender(record);
}

function observeSizes() {
  if (!win.ResizeObserver) return;
  const observer = new win.ResizeObserver((entries) => {
    for (const entry of entries) {
      const record = [...charts.values()].find((r) => r.figure === entry.target);
      if (!record) continue;
      const width = Math.floor(record.body.clientWidth);
      if (width > 0 && Math.abs(width - record.width) >= 1) scheduleRender(record);
    }
  });
  for (const record of charts.values()) observer.observe(record.figure);
}

// ---- pages ------------------------------------------------------------------------------------

function showPage(id, options = {}) {
  const page = pages.find((p) => p.id === id) || pages[0];
  if (!page) return;
  for (const p of pages) {
    const active = p === page;
    p.el.hidden = !active;
    p.tab.setAttribute('aria-selected', active ? 'true' : 'false');
    p.tab.tabIndex = active ? 0 : -1;
  }
  if (options.focus) page.tab.focus();
  if (options.hash !== false) {
    try {
      win.history.replaceState(null, '', `#${page.id}`);
    } catch (error) {
      try {
        win.location.hash = page.id;
      } catch (inner) {
        // A frame that refuses to change its URL still shows the page.
      }
    }
  }
  updateRelevance();
  for (const record of charts.values()) if (page.el.contains(record.figure)) scheduleRender(record);
  if (options.scroll && page.el.getBoundingClientRect().top < 0) page.el.scrollIntoView({block: 'start'});
}

function setupPages() {
  const sections = [...doc.querySelectorAll('[data-page]')];
  if (sections.length < 2) return;
  const list = el('div', {class: 'tabs', role: 'tablist'});
  sections.forEach((section, index) => {
    const id = section.getAttribute('data-page') || `page-${index + 1}`;
    const title = section.getAttribute('data-title') || id;
    if (!section.id) section.id = `page-${id}`;
    const tab = el('button', {type: 'button', role: 'tab', id: `tab-${id}`, class: 'tab', 'aria-selected': 'false', 'aria-controls': section.id, tabindex: '-1', text: title, 'data-label': title});
    tab.dataset.tab = id;
    section.setAttribute('role', 'tabpanel');
    section.setAttribute('aria-labelledby', tab.id);
    section.classList.add('page');
    tab.addEventListener('click', () => showPage(id, {scroll: true}));
    list.append(tab);
    pages.push({id, title, el: section, tab});
  });
  list.addEventListener('keydown', (event) => {
    const at = pages.findIndex((p) => p.tab === doc.activeElement);
    if (at < 0) return;
    const last = pages.length - 1;
    const target = {ArrowRight: at === last ? 0 : at + 1, ArrowLeft: at === 0 ? last : at - 1, Home: 0, End: last}[event.key];
    if (target === undefined) return;
    event.preventDefault();
    showPage(pages[target].id, {focus: true});
  });
  sections[0].before(list);
  const wanted = (win.location.hash || '').slice(1);
  showPage(pages.some((p) => p.id === wanted) ? wanted : pages[0].id, {hash: false});
  win.addEventListener('hashchange', () => {
    const next = (win.location.hash || '').slice(1);
    if (pages.some((p) => p.id === next)) showPage(next, {hash: false});
  });
}

/** A slicer shows while a chart it filters is on screen; a shared slicer leaves the pages it does not touch. */
function updateRelevance() {
  for (const {def, el: node} of slicers.values()) {
    const used = [...charts.values()].filter((c) => c.spec.filters.includes(def.id));
    node.hidden = used.length > 0 && !used.some((c) => c.figure.offsetParent !== null);
  }
  for (const bar of doc.querySelectorAll('.slicer-bar')) bar.hidden = [...bar.children].every((n) => n.hidden);
}

/** Sticky bars stack: each sits under the ones before it. */
function stackStickyBars() {
  const bars = [...doc.querySelectorAll('.slicer-bar, .tabs')];
  const place = () => {
    let offset = 0;
    bars.forEach((bar, index) => {
      bar.style.top = `${offset}px`;
      bar.style.zIndex = String(30 - index);
      offset += bar.offsetHeight;
    });
  };
  place();
  if (win.ResizeObserver) {
    const observer = new win.ResizeObserver(place);
    for (const bar of bars) observer.observe(bar);
  } else {
    win.addEventListener('resize', place);
  }
}

// ---- the rest of the document ------------------------------------------------------------------

function wrapTables() {
  for (const table of doc.querySelectorAll('table')) {
    if (table.parentElement && table.parentElement.classList.contains('table-wrap')) continue;
    const wrap = el('div', {class: 'table-wrap'});
    table.before(wrap);
    wrap.append(table);
  }
}

function labelCallouts() {
  for (const callout of doc.querySelectorAll('.callout[data-kind]')) {
    const kind = callout.getAttribute('data-kind');
    if (!text[kind] || callout.querySelector('.callout-label')) continue;
    // The glyph is a shape the stylesheet draws, so it holds no text a reader or a checker could meet.
    callout.prepend(el('span', {class: 'callout-label'}, [el('i', {'aria-hidden': 'true'}), doc.createTextNode(text[kind])]));
  }
}

// ---- public API -------------------------------------------------------------------------------

/** Run `fn` after the first charts have drawn: two frames, or a timer where frames are paused. */
function afterFrames(fn) {
  let done = false;
  const once = () => {
    if (done) return;
    done = true;
    fn();
  };
  win.requestAnimationFrame(() => win.requestAnimationFrame(once));
  win.setTimeout(once, 400);
}

function flushReady() {
  hydrated = true;
  root.setAttribute('data-report-ready', 'true');
  for (const fn of waiting.splice(0)) fn();
}

const Report = Object.freeze({
  /** Run `fn` once the report is hydrated; an author script that touches charts or slicers waits here. */
  ready(fn) {
    if (hydrated) fn();
    else waiting.push(fn);
  },
  /** One chart: its ECharts instance, its rows, and ways to change them. */
  chart(id) {
    const record = charts.get(id);
    if (!record) return null;
    return {
      get instance() {
        return record.instance;
      },
      rows: () => Core.sourceRows(record.spec.option).slice(),
      setRows(rows) {
        const dataset = Array.isArray(record.spec.option.dataset) ? record.spec.option.dataset[0] : record.spec.option.dataset;
        if (dataset) dataset.source = rows;
        else record.spec.option.dataset = {source: rows};
        for (const entry of slicers.values()) if (entry.def.type === 'filter' && record.spec.filters.includes(entry.def.id)) buildSlicer(entry.el, entry.def);
        record.instant = true;
        scheduleRender(record);
      },
      refresh() {
        scheduleRender(record);
      },
    };
  },
  /** One slicer: its value, and a way to set it. */
  slicer(id) {
    if (!slicers.has(id)) return null;
    return {
      value: () => copy(state[id]),
      set: (value) => setSlicer(id, value),
    };
  },
});
win.Report = Report;

// ---- start ------------------------------------------------------------------------------------

function hydrate() {
  text = STRINGS[isChinese() ? 'zh' : 'en'];
  buildThemes();
  wrapTables();
  readCharts();
  readSlicers();
  labelCallouts();
  setupPages();
  updateRelevance();
  stackStickyBars();
  observeSizes();
  renderAll();
  if (colorQuery) {
    const onScheme = () => {
      buildThemes();
      for (const record of charts.values()) {
        record.themeKey = null;
        scheduleRender(record);
      }
    };
    if (colorQuery.addEventListener) colorQuery.addEventListener('change', onScheme);
  }
  if (motionQuery && motionQuery.addEventListener) motionQuery.addEventListener('change', renderAll);
  if (printQuery && printQuery.addEventListener) printQuery.addEventListener('change', renderAll);
  afterFrames(flushReady);
}

if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', hydrate);
else hydrate();
