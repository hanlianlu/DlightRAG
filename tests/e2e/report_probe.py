# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Measurements taken inside a running report: what is painted, how big, how it looks to a reader."""

from typing import Any

from tests.e2e.report_frame import ReportPage
from tests.support import colour

# Every visible piece of text, with the colour it is drawn in and the colour painted behind it.
_COLOUR_JS = """
  const parse = (css) => {
    css = css.trim();
    let m = css.match(/^rgba?\\(([^)]+)\\)$/);
    if (m) {
      const p = m[1].split(/[\\s,/]+/).filter(Boolean).map(Number);
      return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1];
    }
    m = css.match(/^color\\(srgb ([^)]+)\\)$/);
    if (m) {
      const p = m[1].split(/[\\s/]+/).filter(Boolean).map(Number);
      return [p[0] * 255, p[1] * 255, p[2] * 255, p.length > 3 ? p[3] : 1];
    }
    return [0, 0, 0, 0];
  };
  const over = (top, bottom) => {
    const a = top[3] + bottom[3] * (1 - top[3]);
    if (a === 0) return [0, 0, 0, 0];
    return [0, 1, 2].map((i) => (top[i] * top[3] + bottom[i] * bottom[3] * (1 - top[3])) / a).concat(a);
  };
  const behind = (el) => {
    const layers = [];
    for (let n = el; n; n = n.parentElement) {
      const c = parse(getComputedStyle(n).backgroundColor);
      if (c[3] > 0) layers.push(c);
      if (c[3] >= 0.999) break;
    }
    let base = parse(getComputedStyle(document.documentElement).backgroundColor);
    if (base[3] < 0.999) base = [255, 255, 255, 1];
    return layers.reduceRight((acc, layer) => over(layer, acc), base);
  };
"""

_TEXT_SCAN = (
    "() => {\n"
    + _COLOUR_JS
    + """
  const visible = (el) => {
    const style = getComputedStyle(el);
    if (style.visibility !== 'visible' || style.display === 'none') return false;
    const box = el.getBoundingClientRect();
    return box.width > 0 && box.height > 0 && !el.closest('[hidden], option');
  };
  const found = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const text = node.textContent.trim();
    const el = node.parentElement;
    if (!text || !el || !visible(el) || el.closest('script, style, title')) continue;
    const style = getComputedStyle(el);
    const svg = el instanceof SVGElement;
    const colour = svg ? parse(style.fill) : parse(style.color);
    const bg = behind(el);
    const fg = over(colour, bg);
    found.push({
      text: text.slice(0, 40),
      tag: el.tagName.toLowerCase() + (el.className && typeof el.className === 'string' ? '.' + el.className.split(' ')[0] : ''),
      size: parseFloat(style.fontSize),
      weight: Number(style.fontWeight) || 400,
      fg, bg,
    });
  }
  return found;
}
"""
)

_FACTS = """
(id) => {
  const handle = Report.chart(id);
  const instance = handle && handle.instance;
  if (!instance) return null;
  const option = instance.getOption();
  const grid = instance.getModel().getComponent('grid', 0);
  const rect = grid && grid.coordinateSystem ? grid.coordinateSystem.getRect() : null;
  const figure = document.querySelector(`figure[data-chart="${id}"]`);
  const body = figure.querySelector('.chart-body');
  const first = (list, ...path) => path.reduce((v, k) => (v == null ? v : v[k]), list && list[0]);
  return {
    width: instance.getWidth(),
    height: instance.getHeight(),
    body: [body.clientWidth, body.clientHeight],
    profile: figure.dataset.profile,
    series: (option.series || []).map((s) => ({type: s.type, name: s.name, data: s.data, color: (s.itemStyle || {}).color})),
    plot: rect ? [rect.width, rect.height] : null,
    legendFont: first(option.legend, 'textStyle', 'fontSize'),
    axisFont: first(option.xAxis, 'axisLabel', 'fontSize'),
    palette: option.color,
    horizontal: !!(option.yAxis && option.yAxis[0] && option.yAxis[0].type === 'category'),
  };
}
"""

_FOCUS = (
    "() => {\n"
    + _COLOUR_JS
    + """
  const el = document.activeElement;
  if (!el || el === document.body) return null;
  const style = getComputedStyle(el);
  return {
    behind: behind(el.parentElement || el),
    index: [...document.querySelectorAll('button, select, input, a[href], [tabindex]')].indexOf(el),
    tag: el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (el.className ? '.' + String(el.className).split(' ')[0] : ''),
    style: style.outlineStyle,
    width: parseFloat(style.outlineWidth),
    offset: parseFloat(style.outlineOffset),
    colour: style.outlineColor,
  };
}
"""
)

_TOKENS = """
(names) => {
  const style = getComputedStyle(document.documentElement);
  return Object.fromEntries(names.map((n) => [n, style.getPropertyValue(n).trim()]));
}
"""


def text_scan(page: ReportPage) -> list[dict[str, Any]]:
    return page.eval(_TEXT_SCAN)


def contrast_of(item: dict[str, Any]) -> float:
    """The contrast of one scanned piece of text against what is painted behind it."""
    fg, bg = item["fg"], item["bg"]
    return colour.contrast(
        (fg[0] / 255, fg[1] / 255, fg[2] / 255, 1.0), (bg[0] / 255, bg[1] / 255, bg[2] / 255, 1.0)
    )


def is_large(item: dict[str, Any]) -> bool:
    """WCAG's large text: 24px, or 18.66px in bold."""
    return item["size"] >= 24 or (item["size"] >= 18.66 and item["weight"] >= 700)


def failing_text(page: ReportPage) -> list[str]:
    """Every visible text under the contrast it needs (4.5:1, large text 3:1), described."""
    return [
        f"{i['tag']} {i['text']!r} {contrast_of(i):.2f}:1 (needs {3 if is_large(i) else 4.5})"
        for i in text_scan(page)
        if contrast_of(i) < (3 if is_large(i) else 4.5)
    ]


def chart_facts(page: ReportPage, chart_id: str) -> dict[str, Any] | None:
    return page.eval(_FACTS, chart_id)


def focus_ring(page: ReportPage) -> dict[str, Any] | None:
    return page.eval(_FOCUS)


def tokens(page: ReportPage, names: list[str]) -> dict[str, str]:
    return page.eval(_TOKENS, names)
