// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** Draw an ECharts option with the server-side SVG renderer: JSON on stdin, the SVG on stdout. */

const echarts = require('./echarts.min.js');
const {readFileSync} = require('node:fs');

try {
  const {option, theme, width, height} = JSON.parse(readFileSync(0, 'utf8'));
  echarts.registerTheme('dlight', theme);
  const chart = echarts.init(null, 'dlight', {renderer: 'svg', ssr: true, width, height});
  chart.setOption(option);
  process.stdout.write(chart.renderToSVGString());
  // Without dispose, zrender's timer keeps node alive.
  chart.dispose();
} catch (error) {
  process.stderr.write(String(error?.message ?? error));
  process.exitCode = 1;
}
