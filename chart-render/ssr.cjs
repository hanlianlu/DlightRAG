// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Draw an ECharts option with the server-side SVG renderer: JSON on stdin, the SVG on stdout. */

const {readFileSync} = require('node:fs');
const Theme = require('./theme.js');
const palette = require('./palette.json');
const structure = require('./theme.json');

/** The image ships echarts.min.js beside this file; a checkout has it in node_modules. */
function loadEcharts() {
  for (const path of ['./echarts.min.js', './node_modules/echarts/dist/echarts.min.js']) {
    try {
      return require(path);
    } catch (error) {
      if (error.code !== 'MODULE_NOT_FOUND') throw error;
    }
  }
  throw new Error('echarts.min.js is missing: run `npm ci` in chart-render');
}

try {
  const echarts = loadEcharts();
  const {option, width, height, mode = 'light'} = JSON.parse(readFileSync(0, 'utf8'));
  const {name, option: picked} = Theme.pick(option);
  echarts.registerTheme('dlight', Theme.build(structure, palette[mode])[name]);
  const chart = echarts.init(null, 'dlight', {renderer: 'svg', ssr: true, width, height});
  chart.setOption(Theme.decorate(picked, name));
  process.stdout.write(chart.renderToSVGString());
  // Without dispose, zrender's timer keeps node alive.
  chart.dispose();
} catch (error) {
  process.stderr.write(String(error?.message ?? error));
  process.exitCode = 1;
}
