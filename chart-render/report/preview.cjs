// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/**
 * Draw report charts with the server-side SVG renderer, through the same `core.prepare` the browser
 * runtime calls: JSON jobs on stdin, results on stdout.
 *
 * A job is `{id, spec, state, width | widths, mode, svg, sweep, slicers, lang}`. `svg: false` only
 * checks that ECharts can draw the chart; `sweep: true` draws it once for every state of its first
 * filter, after the defaults; `slicers` and `lang` are what the `{id}` tokens read. A result is
 * `{id, width, state, height, empty, svg}` or `{id, width, state, error}`.
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

/** The states a slicer sweep draws: the first filter at each of its values. */
function sweepStates(spec) {
  const id = (spec.filters || [])[0];
  const def = id && spec.slicers ? spec.slicers[id] : null;
  if (!def) return [];
  if (def.type === 'metric') return def.options.map((o) => ({[id]: o.label}));
  const values = Core.filterValues(def, [spec]);
  return [def.mode === 'multi' ? {[id]: []} : {[id]: null}, ...values.map((v) => ({[id]: def.mode === 'multi' ? [v] : v}))];
}

function main() {
  const echarts = loadEcharts();
  const {jobs} = JSON.parse(readFileSync(0, 'utf8'));
  const themes = {};
  const results = [];
  for (const job of jobs) {
    const {id, spec, mode = 'light', svg = true} = job;
    const states = [job.state || {}, ...(job.sweep ? sweepStates(spec) : [])];
    for (const width of job.widths || [job.width]) {
      for (const state of states) {
        const label = {id, width, state: Object.keys(state).length ? state : undefined};
        try {
          const colors = palette[mode];
          themes[mode] ??= Core.themes(structure, colors);
          const plan = Core.prepare(spec, state, width, {themes: themes[mode], slicers: job.slicers, lang: job.lang});
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
