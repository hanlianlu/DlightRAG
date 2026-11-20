// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/**
 * Draw report charts with the server-side SVG renderer, through the same `core.prepare` the browser
 * runtime calls: JSON jobs on stdin, results on stdout.
 *
 * A job is `{id, spec, state, width | widths, mode, svg, sweep, render, slicers, lang}`. `svg: false`
 * only checks that ECharts can draw the chart; `sweep: true` draws it once more for every slicer it
 * lists, at a choice that is not that slicer's default, after the default; `render: false` runs no
 * ECharts at all and returns `digest` instead, which is how a caller asks whether the drawing follows
 * the state rather than what it looks like; `slicers` and `lang` are what the `{id}` tokens read. A
 * result is `{id, width, state, height, empty, svg | digest}` or `{id, width, state, error}`.
 */

const {readFileSync} = require('node:fs');
const Core = require('./core.js');
const palette = require('../palette.json');
const structure = require('../theme.json');

/** The image ships echarts.min.js beside the renderer; a checkout has it in node_modules. */
function loadEcharts() {
  for (const path of ['../echarts.min.js', '../node_modules/echarts/dist/echarts.min.js']) {
    try {
      return require(path);
    } catch (error) {
      if (error.code !== 'MODULE_NOT_FOUND') throw error;
    }
  }
  throw new Error('echarts.min.js is missing: run `npm ci` in chart-render');
}

const escapeXml = (text) => String(text).replace(/[<>&]/g, (c) => ({'<': '&lt;', '>': '&gt;', '&': '&amp;'})[c]);

function emptyFrame(width, height, mode) {
  const roles = palette[mode].roles;
  return (
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">` +
    `<rect width="${width}" height="${height}" fill="${roles.surface}"/>` +
    `<text x="${width / 2}" y="${height / 2}" text-anchor="middle" font-size="13" fill="${roles.textMuted}">${escapeXml('No data')}</text></svg>`
  );
}

/** Whether a choice is the one the slicer already holds, so sweeping to it would change nothing. */
function isDefaultChoice(def, value) {
  const held = Core.slicerDefault(def);
  return !Array.isArray(held) && held !== null && String(held) === String(value);
}

/** The states a sweep draws: for every slicer the chart lists, one choice that is not its default. */
function sweepStates(spec) {
  const out = [];
  for (const id of spec.filters || []) {
    const def = spec.slicers ? spec.slicers[id] : null;
    if (!def) continue;
    if (def.type === 'metric') {
      const other = def.options.find((option) => option.label !== Core.slicerDefault(def));
      if (other) out.push({[id]: other.label});
    } else if (def.type === 'filter') {
      const value = Core.filterValues(def, [spec]).find((v) => !isDefaultChoice(def, v));
      if (value !== undefined) out.push({[id]: def.mode === 'multi' ? [value] : value});
    }
  }
  return out;
}

/** Everything a prepared chart says, as one string: the rows it kept, the tokens, the caption, the box. */
function planDigest(plan) {
  return JSON.stringify([plan.option, plan.caption, plan.height, plan.empty]);
}

function main() {
  const echarts = loadEcharts();
  const {jobs} = JSON.parse(readFileSync(0, 'utf8'));
  const themes = {};
  const results = [];
  for (const job of jobs) {
    const {id, spec, mode = 'light', svg = true, render = true} = job;
    const states = [job.state || {}, ...(job.sweep ? sweepStates(spec) : [])];
    for (const width of job.widths || [job.width]) {
      for (const state of states) {
        const label = {id, width, state: Object.keys(state).length ? state : undefined};
        try {
          const colors = palette[mode];
          themes[mode] ??= Core.themes(structure, colors);
          const plan = Core.prepare(spec, state, width, {themes: themes[mode], slicers: job.slicers, lang: job.lang});
          if (!render) {
            results.push({...label, height: plan.height, empty: plan.empty, digest: planDigest(plan)});
            continue;
          }
          if (plan.empty) {
            results.push({...label, height: plan.height, empty: true, svg: svg ? emptyFrame(width, plan.height, mode) : undefined});
            continue;
          }
          const name = `report-${mode}-${plan.theme}`;
          echarts.registerTheme(name, themes[mode][plan.theme]);
          const chart = echarts.init(null, name, {renderer: 'svg', ssr: true, width, height: plan.height});
          try {
            chart.setOption({...plan.option, backgroundColor: colors.roles.surface, animation: false});
            results.push({...label, height: plan.height, empty: false, svg: svg ? chart.renderToSVGString() : undefined});
          } finally {
            chart.dispose();
          }
        } catch (error) {
          results.push({...label, error: String(error?.message ?? error)});
        }
      }
    }
  }
  process.stdout.write(JSON.stringify(results));
}

main();
