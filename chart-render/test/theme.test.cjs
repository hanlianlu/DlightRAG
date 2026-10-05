// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/** The theme layer both `echarts-render` and `html-report` build their charts from. */

const assert = require('node:assert/strict');
const {test} = require('node:test');

const Theme = require('../theme.js');
const palette = require('../palette.json');
const structure = require('../theme.json');

const COLOUR = /^(#[0-9a-f]{3,8}|rgba?\(.*|hsla?\(.*)$/i;

const strings = (value, found = []) => {
  if (typeof value === 'string') found.push(value);
  else if (value && typeof value === 'object') Object.values(value).forEach((v) => strings(v, found));
  return found;
};

test('theme.json writes no colour of its own: every colour is a role palette.json fills in', () => {
  const literals = strings(structure).filter((text) => COLOUR.test(text));
  assert.deepEqual(literals, []);
  const roles = strings(structure).filter((text) => text.startsWith('@'));
  assert.ok(roles.length > 10);
  for (const mode of ['light', 'dark']) {
    for (const role of roles) {
      const name = role.slice(1);
      assert.ok(name in palette[mode].roles || name in palette[mode], `${mode} lacks ${name}`);
    }
  }
});

test('both modes define the same roles and the same palettes', () => {
  assert.deepEqual(Object.keys(palette.light.roles).sort(), Object.keys(palette.dark.roles).sort());
  assert.deepEqual(Object.keys(palette.light).sort(), Object.keys(palette.dark).sort());
  for (const mode of ['light', 'dark']) {
    assert.equal(palette[mode].categorical.length, 8);
    assert.equal(palette[mode].diverging.length, 7);
    assert.equal(palette[mode].divergingEnds.length, 2);
  }
});

test('resolve fills every reference and refuses one the palette lacks', () => {
  const colors = {roles: {ink: '#111111'}, ramp: ['#aaaaaa', '#bbbbbb']};
  assert.deepEqual(Theme.resolve({a: '@ink', b: {c: ['@ramp', '@ink']}, d: 3, e: 'plain'}, colors), {
    a: '#111111',
    b: {c: [['#aaaaaa', '#bbbbbb'], '#111111']},
    d: 3,
    e: 'plain',
  });
  assert.throws(() => Theme.resolve({a: '@missing'}, colors), /@missing/);
});

test('build makes one theme per palette from the structure, and the structure is not a theme key', () => {
  const themes = Theme.build(structure, palette.light);
  assert.deepEqual(Object.keys(themes).sort(), ['categorical', 'diverging', 'highlight', 'sequential']);
  for (const theme of Object.values(themes)) assert.equal(theme.palettes, undefined);
  assert.deepEqual(themes.categorical.color, palette.light.categorical);
  assert.deepEqual(themes.highlight.color, palette.light.highlight);
  assert.deepEqual(themes.sequential.color, palette.light.sequential);
  assert.deepEqual(themes.diverging.color, palette.light.divergingEnds);
  assert.deepEqual(themes.diverging.visualMap.inRange.color, palette.light.diverging);
  assert.deepEqual(themes.categorical.visualMap.inRange.color, palette.light.sequential);
  assert.equal(themes.categorical.backgroundColor, palette.light.roles.background);
  assert.equal(themes.categorical.textStyle.color, palette.light.roles.textMuted);
});

test('the structure of the theme is the same in every palette and every mode', () => {
  const shape = (value) => JSON.stringify(value, (key, v) => (typeof v === 'string' && key !== 'fontFamily' ? '' : v));
  const light = Theme.build(structure, palette.light);
  const dark = Theme.build(structure, palette.dark);
  assert.equal(shape(light.categorical.title), shape(dark.categorical.title));
  assert.equal(light.categorical.legend.textStyle.fontSize, dark.categorical.legend.textStyle.fontSize);
  assert.notEqual(light.categorical.textStyle.color, dark.categorical.textStyle.color);
});

test('pick splits the palette from the option, keeps categorical as the default, and refuses an unknown name', () => {
  const asked = Theme.pick({palette: 'diverging', series: []});
  assert.equal(asked.name, 'diverging');
  assert.deepEqual(asked.option, {series: []});
  assert.equal(Theme.pick({series: []}).name, 'categorical');
  assert.throws(() => Theme.pick({palette: 'neon'}), /palette "neon" is not one of categorical, highlight/);
});

const lines = (n, extra = {}) => ({series: Array.from({length: n}, (_, i) => ({type: 'line', name: `s${i}`, data: [i], ...extra}))});

test('decorate gives four or more lines their own dash and marker, and leaves fewer alone', () => {
  const three = lines(3);
  assert.equal(Theme.decorate(three, 'categorical'), three);
  const four = Theme.decorate(lines(4)).series;
  const looks = four.map((s) => `${JSON.stringify(s.lineStyle.type)}/${s.symbol}`);
  assert.equal(new Set(looks).size, 4);
  assert.ok(four.every((s) => s.showSymbol === true));
});

test('decorate keeps what the author wrote and does not touch the original', () => {
  const option = lines(4, {symbol: 'pin'});
  option.series[0].lineStyle = {type: 'solid', width: 5};
  const copy = JSON.stringify(option);
  const out = Theme.decorate(option);
  assert.equal(JSON.stringify(option), copy);
  assert.equal(out.series[0].lineStyle.type, 'solid');
  assert.equal(out.series[0].lineStyle.width, 5);
  assert.ok(out.series.every((s) => s.symbol === 'pin'));
});

test('the highlight palette draws its first line heavier than the rest, whatever the count', () => {
  const out = Theme.decorate(lines(2), 'highlight').series;
  assert.ok(out[0].lineStyle.width > out[1].lineStyle.width);
  assert.equal(Theme.decorate(lines(1), 'highlight').series[0].lineStyle?.width, undefined);
});

test('decorate leaves bars, pies and an option without series as they are', () => {
  const bars = {series: [{type: 'bar', data: [1]}]};
  assert.equal(Theme.decorate(bars, 'highlight'), bars);
  const bare = {title: {text: 'x'}};
  assert.equal(Theme.decorate(bare, 'highlight'), bare);
});

test('merge replaces arrays and merges objects', () => {
  assert.deepEqual(Theme.merge({a: {b: 1, c: [1, 2]}, d: 1}, {a: {c: [3]}, e: 2}), {a: {b: 1, c: [3]}, d: 1, e: 2});
});
