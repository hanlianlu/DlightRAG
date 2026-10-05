// Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
/**
 * The chart theme layer shared by `echarts-render` (the PNG) and the `html-report` runtime.
 *
 * `theme.json` is the structure of the house theme, with every colour written as an "@role"
 * reference; `palette.json` holds one mode's colours for those roles and its four palettes. Pure
 * functions with no DOM and no node built-ins, so node and the browser load this same file.
 */
'use strict';

/** The palette names an option can ask for with `"palette": "..."`; categorical is the default. */
const PALETTES = ['categorical', 'highlight', 'sequential', 'diverging'];

// How a line series tells itself apart from its neighbours without colour, from four series up.
const DASHES = ['solid', [7, 4], [2, 3.5]];
const SYMBOLS = ['circle', 'rect', 'triangle', 'diamond', 'roundRect'];

const isObject = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);

/** Return `value` as a list: one object or a list of them, as ECharts takes them. */
const asList = (value) => (Array.isArray(value) ? value : value == null ? [] : [value]);

/** Merge `overlay` over `base`: objects merge key by key, everything else is replaced. */
function merge(base, overlay) {
  if (!isObject(base) || !isObject(overlay)) return overlay;
  const out = {...base};
  for (const [key, value] of Object.entries(overlay)) out[key] = merge(base[key], value);
  return out;
}

/** Replace every "@role" string under `node` by the colour, or list of colours, `colors` gives it. */
function resolve(node, colors) {
  if (typeof node === 'string' && node.startsWith('@')) {
    const name = node.slice(1);
    const value = name in (colors.roles ?? {}) ? colors.roles[name] : colors[name];
    if (value === undefined) throw new Error(`theme.json uses @${name}, which palette.json lacks`);
    return Array.isArray(value) ? [...value] : value;
  }
  if (Array.isArray(node)) return node.map((item) => resolve(item, colors));
  if (isObject(node)) {
    return Object.fromEntries(Object.entries(node).map(([k, v]) => [k, resolve(v, colors)]));
  }
  return node;
}

/**
 * Build one ECharts theme per palette name from the theme structure and one mode's colours.
 * `structure.palettes` holds what each non-default palette changes; it is not a theme key itself.
 */
function build(structure, colors) {
  const {palettes = {}, ...theme} = structure;
  const base = resolve(theme, colors);
  const themes = {categorical: base};
  for (const [name, overlay] of Object.entries(palettes)) {
    themes[name] = merge(base, resolve(overlay, colors));
  }
  return themes;
}

/**
 * Split the palette an option asks for from the option: `{name, option}`, where `option` is a copy
 * without the `palette` key, which is not an ECharts key. An unknown name throws.
 */
function pick(option) {
  const {palette = 'categorical', ...rest} = option;
  if (!PALETTES.includes(palette)) {
    throw new Error(`palette ${JSON.stringify(palette)} is not one of ${PALETTES.join(', ')}`);
  }
  return {name: palette, option: rest};
}

/**
 * Apply the series rules the theme layer owns, on a copy of the option: from four line series up,
 * each line gets its own dash and marker, so no pair of lines is told apart by colour alone; the
 * highlight palette draws its first line heavier than the rest. What the author wrote wins.
 */
function decorate(option, name = 'categorical') {
  const series = asList(option.series);
  const lines = series.filter((s) => isObject(s) && s.type === 'line');
  if (lines.length < (name === 'highlight' ? 2 : 4)) return option;
  const decorated = series.map((s) => {
    if (!isObject(s) || s.type !== 'line') return s;
    const at = lines.indexOf(s);
    const out = {...s, lineStyle: {...s.lineStyle}};
    if (lines.length >= 4) {
      out.lineStyle.type ??= DASHES[at % DASHES.length];
      out.symbol ??= SYMBOLS[at % SYMBOLS.length];
      out.showSymbol ??= true;
      out.symbolSize ??= 6;
    }
    if (name === 'highlight' && lines.length > 1) out.lineStyle.width ??= at === 0 ? 3.5 : 1.75;
    return out;
  });
  return {...option, series: Array.isArray(option.series) ? decorated : decorated[0]};
}

module.exports = {PALETTES, asList, build, decorate, isObject, merge, pick, resolve};
