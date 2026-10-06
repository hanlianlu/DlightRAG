// In-page probes for the eval checker. Evaluated once per frame (the artifact's frame and every nested
// frame it creates); each function returns JSON-serialisable data and never throws. It reads the
// document (layout, ECharts instances, text); it patches no API and registers only passive listeners.
(() => {
  if (window.__evalProbe) return;
  const P = {};
  const SKIP = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'TEMPLATE', 'HEAD', 'TITLE', 'META', 'LINK']);
  const safe = (fn, fallback) => { try { return fn(); } catch (e) { return fallback === undefined ? { error: String(e && e.message || e) } : fallback; } };

  // ---------------------------------------------------------------- helpers
  // Own escape: a page can shadow the global `CSS` (a `const CSS = '...'` of its own), and then CSS.escape is gone.
  function cssEscape(value) {
    return String(value).replace(/[^a-zA-Z0-9_\u00A0-\uFFFF-]/g, (c) => '\\' + c).replace(/^(\d)/, '\\3$1 ').replace(/^-(\d)/, '-\\3$1 ');
  }
  function hash(s) {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); }
    return (h >>> 0).toString(16) + ':' + s.length;
  }
  function uniqueSel(el) {
    if (!el || el.nodeType !== 1) return '';
    const parts = [];
    let cur = el;
    while (cur && cur.nodeType === 1 && cur !== document.documentElement) {
      if (cur.id && document.querySelectorAll('#' + cssEscape(cur.id)).length === 1) { parts.unshift('#' + cssEscape(cur.id)); break; }
      let idx = 1; let sib = cur;
      while ((sib = sib.previousElementSibling)) idx++;
      parts.unshift(cur.tagName.toLowerCase() + ':nth-child(' + idx + ')');
      cur = cur.parentElement;
    }
    return parts.join(' > ');
  }
  function desc(el) {
    if (!el || el.nodeType !== 1) return '';
    let s = el.tagName.toLowerCase();
    if (el.id) s += '#' + el.id;
    const cls = (typeof el.className === 'string' ? el.className : (el.className && el.className.baseVal) || '').trim().split(/\s+/).filter(Boolean).slice(0, 3);
    if (cls.length) s += '.' + cls.join('.');
    return s;
  }
  function rectOf(el) { const r = el.getBoundingClientRect(); return { x: +r.x.toFixed(1), y: +r.y.toFixed(1), w: +r.width.toFixed(1), h: +r.height.toFixed(1) }; }
  function visible(el) {
    if (!el || el.nodeType !== 1) return false;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    const cs = getComputedStyle(el);
    return cs.visibility !== 'hidden' && cs.display !== 'none' && parseFloat(cs.opacity) !== 0;
  }
  // The visual scale of an element within this frame: the product of the scale of every CSS transform on
  // it and its ancestors (and of `zoom`), or the screen matrix for SVG content. Never a rect/offset ratio:
  // offsetWidth is rounded, so a ratio reads 0.995 for an unscaled element.
  function vscale(el) {
    if (el instanceof SVGElement && el.getScreenCTM) { const m = el.getScreenCTM(); return m ? Math.hypot(m.a, m.b) : 1; }
    let s = 1;
    for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.transform && cs.transform !== 'none') {
        try { const m = new DOMMatrix(cs.transform); s *= Math.hypot(m.a, m.b) || 1; } catch (err) { /* not a matrix */ }
      }
      const z = parseFloat(cs.zoom);
      if (z && z !== 1) s *= z;
    }
    return s;
  }
  function inEcharts(el) { return !!(el.closest && el.closest('[_echarts_instance_]')); }
  function parseFont(font) { const m = /(\d+(?:\.\d+)?)px/.exec(font || ''); return m ? parseFloat(m[1]) : null; }
  // A script block that is the ECharts library, by content and not by banner.
  function isLibrary(text) { return text.length > 200000 && text.indexOf('getInstanceByDom') >= 0 && text.indexOf('registerTheme') >= 0; }

  P.helpers = { hash, uniqueSel, desc };
  P.scaleOf = (el) => vscale(el);

  // ---------------------------------------------------------------- document
  P.info = () => safe(() => {
    const de = document.documentElement;
    return {
      href: location.href,
      title: document.title,
      innerWidth, innerHeight,
      scrollWidth: de.scrollWidth, clientWidth: de.clientWidth, bodyScrollWidth: document.body ? document.body.scrollWidth : 0,
      scrollHeight: de.scrollHeight,
      overflowX: { html: getComputedStyle(de).overflowX, body: document.body ? getComputedStyle(document.body).overflowX : '' },
      hasEcharts: typeof window.echarts === 'object' && !!window.echarts,
      echartsVersion: (window.echarts && window.echarts.version) || null,
      colorScheme: getComputedStyle(de).colorScheme,
      bodyBg: document.body ? getComputedStyle(document.body).backgroundColor : '',
      htmlBg: getComputedStyle(de).backgroundColor,
    };
  });

  // Elements reaching outside the viewport horizontally that no scroll container of their own clips.
  P.overflow = () => safe(() => {
    const vw = document.documentElement.clientWidth;
    const offenders = [];
    const cutOff = [];
    const bodyHides = ['hidden', 'clip'].includes(getComputedStyle(document.documentElement).overflowX) || ['hidden', 'clip'].includes(getComputedStyle(document.body).overflowX);
    for (const el of document.body.querySelectorAll('*')) {
      if (SKIP.has(el.tagName)) continue;
      const r = el.getBoundingClientRect();
      if (r.width < 1 || r.height < 1) continue;
      if (r.right <= vw + 1 && r.left >= -1) continue;
      const cs = getComputedStyle(el);
      if (cs.visibility === 'hidden' || cs.display === 'none') continue;
      if (cs.position === 'fixed') continue;
      // clipped by an ancestor scroll/clip container that itself sits inside the viewport?
      let clipped = false;
      for (let a = el.parentElement; a && a !== document.body && a !== document.documentElement; a = a.parentElement) {
        const acs = getComputedStyle(a);
        if (['auto', 'scroll', 'hidden', 'clip'].includes(acs.overflowX)) {
          const ar = a.getBoundingClientRect();
          if (ar.right <= vw + 1 && ar.left >= -1) { clipped = true; break; }
        }
      }
      if (clipped) continue;
      const item = { sel: desc(el), right: +r.right.toFixed(1), left: +r.left.toFixed(1), w: +r.width.toFixed(1), text: (el.innerText || '').trim().slice(0, 30) };
      (bodyHides ? cutOff : offenders).push(item);
    }
    // Keep only the outermost offenders: a child of an offender is the same finding.
    const top = (list) => list.filter((it, i) => !list.some((o, j) => j !== i && o.left <= it.left + 1 && o.right >= it.right - 1 && o.w > it.w + 1)).slice(0, 8);
    return { vw, scrollWidth: document.documentElement.scrollWidth, bodyScrollWidth: document.body.scrollWidth, offenders: top(offenders), cutOffByBodyOverflow: top(cutOff), bodyHides };
  });

  // ---------------------------------------------------------------- scripts, references, iframes
  P.scripts = () => safe(() => Array.from(document.scripts).map((s) => {
    const text = s.textContent || '';
    const lib = isLibrary(text);
    return { id: s.id || null, type: s.type || '', src: s.getAttribute('src'), len: text.length, isLibrary: lib, text: lib ? '' : text.slice(0, 700000) };
  }));

  P.iframes = () => safe(() => Array.from(document.querySelectorAll('iframe')).map((f) => ({
    sel: desc(f), title: f.title || '', hasSrcdoc: f.hasAttribute('srcdoc'), srcdocLen: (f.getAttribute('srcdoc') || '').length,
    src: f.getAttribute('src'), sandbox: f.getAttribute('sandbox'), rect: rectOf(f), offsetW: f.offsetWidth, offsetH: f.offsetHeight,
    scale: +(vscale(f)).toFixed(3), visible: visible(f),
  })));

  const URL_RE = /url\(\s*(['"]?)(.*?)\1\s*\)/gi;
  const IMPORT_RE = /@import\s+(?:url\(\s*)?['"]?([^'")\s;]+)/gi;
  const isLocal = (u) => { const t = (u || '').trim(); return t === '' || t.startsWith('#') || /^(data|blob|about|javascript):/i.test(t); };
  P.references = () => safe(() => {
    const found = [];
    const add = (kind, el, attr, value) => { if (!isLocal(value)) found.push({ kind, el: desc(el), attr, value: String(value).slice(0, 160) }); };
    for (const el of document.querySelectorAll('*')) {
      for (const attr of ['src', 'href', 'poster', 'action', 'formaction', 'data', 'background', 'xlink:href', 'ping', 'manifest', 'cite', 'longdesc']) {
        if (el.hasAttribute(attr)) add('attr', el, attr, el.getAttribute(attr));
      }
      if (el.hasAttribute('srcset')) for (const part of el.getAttribute('srcset').split(',')) add('attr', el, 'srcset', part.trim().split(/\s+/)[0]);
      if (el.tagName === 'LINK') found.push({ kind: 'link', el: desc(el), attr: 'rel', value: (el.getAttribute('rel') || '') + ' ' + (el.getAttribute('href') || '') });
      if (el.tagName === 'BASE') found.push({ kind: 'base', el: desc(el), attr: 'href', value: el.getAttribute('href') || '' });
      if (el.tagName === 'META' && /refresh/i.test(el.getAttribute('http-equiv') || '')) found.push({ kind: 'meta-refresh', el: desc(el), attr: 'content', value: el.getAttribute('content') || '' });
      const style = el.getAttribute('style');
      if (style) { let m; URL_RE.lastIndex = 0; while ((m = URL_RE.exec(style))) add('style-attr', el, 'style', m[2]); }
      if (el.tagName === 'STYLE') {
        const css = el.textContent || ''; let m;
        URL_RE.lastIndex = 0; while ((m = URL_RE.exec(css))) add('style', el, 'url()', m[2]);
        IMPORT_RE.lastIndex = 0; while ((m = IMPORT_RE.exec(css))) add('style', el, '@import', m[1]);
      }
    }
    return found.slice(0, 60);
  });

  // ---------------------------------------------------------------- charts
  function chartTexts(inst) {
    // Every text element ECharts draws (titles, legend, axes, labels), whatever the renderer.
    const out = [];
    const list = inst.getZr().storage.getDisplayList(true);
    for (const e of list) {
      if (!e.style || typeof e.style.text !== 'string' || !e.style.text.trim()) continue;
      const size = typeof e.style.fontSize === 'number' ? e.style.fontSize : parseFont(e.style.font);
      out.push({ text: e.style.text.slice(0, 30), size: size == null ? 12 : size });
    }
    return out;
  }
  // Drawn texts whose boxes cover each other: the collisions a person sees (a subtitle over an axis name, x labels run together).
  // Rotated text is skipped (its axis-aligned box overlaps a neighbour's by design); a pair counts when over a fifth of the smaller box is covered (adjacent labels run together well before half).
  function overlapPairs(boxes) {
    const out = [];
    for (let i = 0; i < boxes.length && i < 300; i++) {
      for (let j = i + 1; j < boxes.length && j < 300; j++) {
        const a = boxes[i], b = boxes[j];
        const w = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
        const h = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
        if (w <= 1 || h <= 1) continue;
        const cover = (w * h) / Math.min(a.w * a.h, b.w * b.h);
        if (cover > 0.2) out.push({ a: a.text, b: b.text, cover: +cover.toFixed(2) });
      }
    }
    return out;
  }
  function chartTextBoxes(inst) {
    const boxes = [];
    const list = inst.getZr().storage.getDisplayList(true);
    for (const e of list) {
      if (!e.style || typeof e.style.text !== 'string' || !e.style.text.trim() || e.ignore || e.invisible || e.style.opacity === 0) continue;
      let r; let m = null;
      try { r = e.getBoundingRect().clone(); m = e.getComputedTransform ? e.getComputedTransform() : null; } catch (err) { continue; }
      if (m) { if (Math.abs(m[1]) > 0.01 || Math.abs(m[2]) > 0.01) continue; r.applyTransform(m); }
      if (r.width < 2 || r.height < 2) continue;
      boxes.push({ text: e.style.text.slice(0, 16), x: r.x, y: r.y, w: r.width, h: r.height });
    }
    return boxes;
  }
  function viewTexts(inst, model) {
    const out = [];
    const view = inst.getViewOfComponentModel(model);
    if (!view || !view.group) return out;
    view.group.traverse((e) => {
      if (e.style && typeof e.style.text === 'string' && e.style.text.trim()) {
        out.push({ text: e.style.text.slice(0, 30), size: typeof e.style.fontSize === 'number' ? e.style.fontSize : (parseFont(e.style.font) || 12) });
      }
    });
    return out;
  }
  function groupRect(inst, model) {
    const view = inst.getViewOfComponentModel(model);
    if (!view || !view.group) return null;
    let r = view.group.getBoundingRect().clone();
    if (view.group.transform) r = r.applyTransform(view.group.transform);
    return { x: r.x, y: r.y, w: r.width, h: r.height };
  }
  function components(gm, type) { const a = []; gm.eachComponent(type, (m) => a.push(m)); return a; }
  function describeChart(el, inst) {
    const gm = inst.getModel();
    const opt = inst.getOption() || {};
    const asArr = (v) => (Array.isArray(v) ? v : v ? [v] : []);
    const series = asArr(opt.series).map((s) => ({ type: s.type, name: s.name == null ? null : String(s.name), n: Array.isArray(s.data) ? s.data.length : null, stack: s.stack || null }));
    const dataset = asArr(opt.dataset).map((d) => (Array.isArray(d.source) ? d.source.length : null));
    const titles = components(gm, 'title').map((m) => ({
      text: String(m.get('text') || ''), subtext: String(m.get('subtext') || ''), show: m.get('show') !== false,
      textSize: parseFont(m.getModel('textStyle').getFont()), subtextSize: parseFont(m.getModel('subtextStyle').getFont()),
      rect: safe(() => groupRect(inst, m), null),
    }));
    const legends = components(gm, 'legend').map((m) => ({
      show: m.get('show') !== false, type: m.get('type') || 'plain', orient: m.get('orient'), size: parseFont(m.getModel('textStyle').getFont()),
      rect: safe(() => groupRect(inst, m), null), entries: (m.getData ? m.getData().length : null),
    }));
    const axes = [];
    for (const t of ['xAxis', 'yAxis', 'angleAxis', 'radiusAxis']) {
      for (const m of components(gm, t)) {
        axes.push({ kind: t, type: m.get('type'), labelShown: m.getModel('axisLabel').get('show') !== false, labelSize: parseFont(m.getModel('axisLabel').getFont()),
          name: m.get('name') || '', labelRotate: m.getModel('axisLabel').get('rotate') });
      }
    }
    const grids = components(gm, 'grid').map((m) => { const r = m.coordinateSystem && m.coordinateSystem.getRect(); return r ? { x: r.x, y: r.y, w: r.width, h: r.height } : null; }).filter(Boolean);
    const body = []; // text that is neither title nor legend: axis labels, names, series labels
    const titleSet = new Set(); const legendSet = new Set();
    for (const m of components(gm, 'title')) for (const t of viewTexts(inst, m)) titleSet.add(t.text + '|' + t.size);
    for (const m of components(gm, 'legend')) for (const t of viewTexts(inst, m)) legendSet.add(t.text + '|' + t.size);
    const texts = chartTexts(inst);
    for (const t of texts) { const k = t.text + '|' + t.size; if (!titleSet.has(k) && !legendSet.has(k)) body.push(t); }
    return {
      series, dataset, titles, legends, axes, grids,
      dataZoom: asArr(opt.dataZoom).length, visualMap: asArr(opt.visualMap).length, toolbox: asArr(opt.toolbox).length,
      overlaps: safe(() => overlapPairs(chartTextBoxes(inst)).slice(0, 6), []),
      textCount: texts.length, minTextSize: texts.length ? Math.min(...texts.map((t) => t.size)) : null,
      minText: texts.length ? texts.reduce((a, b) => (b.size < a.size ? b : a)) : null,
      bodySizes: body.map((t) => t.size),
      allSizes: texts.map((t) => t.size),
    };
  }

  P.charts = () => safe(() => {
    const out = { hasLibrary: typeof window.echarts === 'object' && !!window.echarts, echarts: [], hand: [] };
    if (out.hasLibrary && window.echarts.getInstanceByDom) {
      for (const el of document.querySelectorAll('*')) {
        let inst = null;
        try { inst = window.echarts.getInstanceByDom(el); } catch (e) { inst = null; }
        if (!inst || (inst.isDisposed && inst.isDisposed())) continue;
        const r = rectOf(el);
        const scale = vscale(el);
        const item = {
          sel: uniqueSel(el), desc: desc(el), rect: r, scale: +scale.toFixed(3), clientW: el.clientWidth, clientH: el.clientHeight,
          instW: inst.getWidth(), instH: inst.getHeight(), renderer: safe(() => inst.getZr().painter.type, 'unknown'), visible: visible(el),
          parentW: el.parentElement ? el.parentElement.clientWidth : null,
          padX: (parseFloat(getComputedStyle(el).paddingLeft) || 0) + (parseFloat(getComputedStyle(el).paddingRight) || 0),
          padY: (parseFloat(getComputedStyle(el).paddingTop) || 0) + (parseFloat(getComputedStyle(el).paddingBottom) || 0),
        };
        Object.assign(item, safe(() => describeChart(el, inst), { describeError: true }));
        out.echarts.push(item);
      }
    }
    // Chart-like drawings that are not an ECharts instance: big svg / canvas.
    for (const el of document.querySelectorAll('svg, canvas')) {
      if (inEcharts(el)) continue;
      if (el.parentElement && el.parentElement.closest('svg')) continue; // a nested svg is part of its parent drawing
      const r = el.getBoundingClientRect();
      if (r.width < 150 || r.height < 90) continue;
      if (getComputedStyle(el).display === 'none') continue;
      const tag = el.tagName.toLowerCase();
      let texts = null; let shapes = null;
      if (tag === 'svg') {
        texts = el.querySelectorAll('text').length;
        shapes = el.querySelectorAll('rect,path,circle,line,polyline,polygon,ellipse').length;
        if (texts < 3 || shapes < 4) continue; // an illustration, not a chart
      }
      let overlaps = [];
      if (tag === 'svg') {
        const boxes = [];
        for (const t of el.querySelectorAll('text')) {
          const m = t.getScreenCTM && t.getScreenCTM();
          if (m && (Math.abs(m.b) > 0.01 || Math.abs(m.c) > 0.01)) continue; // rotated
          const br = t.getBoundingClientRect();
          if (br.width < 2 || br.height < 2 || !(t.textContent || '').trim()) continue;
          boxes.push({ text: (t.textContent || '').trim().slice(0, 16), x: br.x, y: br.y, w: br.width, h: br.height });
        }
        overlaps = overlapPairs(boxes).slice(0, 6);
      }
      out.hand.push({ sel: uniqueSel(el), desc: desc(el), tag, rect: rectOf(el), texts, shapes, overlaps });
    }
    // Large images (PNG or SVG data URIs) are, in a report, almost always a chart rendered elsewhere (echarts-render).
    out.images = [];
    for (const img of document.querySelectorAll('img')) {
      if (!visible(img)) continue;
      const nw = img.naturalWidth; const nh = img.naturalHeight;
      if (nw < 300 || nh < 180) continue; // a phone-sized chart (echarts-render --width 380) is still a chart
      const ar = nw / nh; if (ar < 0.5 || ar > 2.8) continue;
      const r = img.getBoundingClientRect();
      out.images.push({ sel: uniqueSel(img), desc: desc(img), natural: [nw, nh], rect: rectOf(img), scale: +(r.width / nw).toFixed(3), kind: (img.currentSrc || img.src || '').slice(0, 22), alt: (img.alt || '').slice(0, 60) });
    }
    return out;
  });

  // A snapshot of what the page plots and says, to see whether a control changed it.
  P.snapshot = () => safe(() => {
    const charts = [];
    if (window.echarts && window.echarts.getInstanceByDom) {
      for (const el of document.querySelectorAll('*')) {
        let inst = null;
        try { inst = window.echarts.getInstanceByDom(el); } catch (e) { inst = null; }
        if (!inst || (inst.isDisposed && inst.isDisposed())) continue;
        const opt = inst.getOption() || {};
        const asArr = (v) => (Array.isArray(v) ? v : v ? [v] : []);
        const data = JSON.stringify({
          dataset: asArr(opt.dataset).map((d) => d.source),
          series: asArr(opt.series).map((s) => ({ type: s.type, name: s.name, data: s.data, encode: s.encode, stack: s.stack })),
          x: asArr(opt.xAxis).map((a) => a.data), y: asArr(opt.yAxis).map((a) => a.data),
        });
        charts.push({ key: uniqueSel(el), data: hash(data), title: hash(JSON.stringify(asArr(opt.title).map((t) => [t.text, t.subtext]))), series: asArr(opt.series).length });
      }
    }
    const svgs = [];
    for (const el of document.querySelectorAll('svg, canvas')) {
      if (inEcharts(el)) continue;
      const r = el.getBoundingClientRect();
      if (r.width < 150 || r.height < 90) continue;
      let h = '';
      if (el.tagName.toLowerCase() === 'svg') h = hash(el.innerHTML);
      else { try { h = hash(el.toDataURL()); } catch (e) { h = 'canvas-unreadable'; } }
      svgs.push({ key: uniqueSel(el), data: h });
    }
    const text = document.body.innerText || '';
    return { charts, svgs, text: hash(text), textLen: text.length, lines: text.split('\n').map((l) => l.trim()).filter(Boolean).length };
  });
  // The visible lines of text, to tell a panel switch (most lines change) from a filter (a few).
  P.lines = () => safe(() => (document.body.innerText || '').split('\n').map((l) => l.trim()).filter(Boolean));

  // ---------------------------------------------------------------- text, fonts, controls
  P.fonts = (opts) => safe(() => {
    const frameScale = (opts && opts.scale) || 1;
    const out = [];
    const cache = new Map();
    const walker = document.createTreeWalker(document.body, 4 /* NodeFilter.SHOW_TEXT: a page may shadow the global */);
    let n;
    while ((n = walker.nextNode())) {
      const text = n.nodeValue.trim();
      if (!text) continue;
      const el = n.parentElement;
      if (!el || SKIP.has(el.tagName) || (el.closest && el.closest('script,style,noscript,template,title'))) continue;
      let info = cache.get(el);
      if (!info) {
        const r = el.getBoundingClientRect();
        const cs = getComputedStyle(el);
        const vis = r.width >= 3 && r.height >= 3 && cs.visibility !== 'hidden' && cs.display !== 'none' && parseFloat(cs.opacity) !== 0;
        info = { vis, size: vis ? parseFloat(cs.fontSize) * vscale(el) * frameScale : 0, css: parseFloat(cs.fontSize), sel: desc(el) };
        cache.set(el, info);
      }
      if (!info.vis) continue;
      out.push({ size: +info.size.toFixed(2), css: info.css, text: text.slice(0, 40), sel: info.sel });
    }
    const below = out.filter((o) => o.size < 11);
    const uniq = []; const seen = new Set();
    for (const o of below) { const k = o.sel + '|' + o.size; if (!seen.has(k)) { seen.add(k); uniq.push(o); } }
    return { count: out.length, min: out.length ? Math.min(...out.map((o) => o.size)) : null, below: uniq.slice(0, 10), belowCount: below.length };
  });

  const CONTROL_SEL = 'button, [role="button"], [role="tab"], [role="radio"], [role="switch"], [role="checkbox"], [role="option"], select, input:not([type="hidden"]), summary, a[href], [onclick], [tabindex]:not([tabindex="-1"])';
  function activeState(el) {
    const a = (n) => el.getAttribute(n);
    for (const n of ['aria-pressed', 'aria-selected', 'aria-checked', 'aria-current']) { if (a(n) === 'true' || (n === 'aria-current' && a(n) && a(n) !== 'false')) return true; }
    if (el.checked === true) return true;
    const cls = typeof el.className === 'string' ? el.className.split(/\s+/) : [];
    return cls.some((c) => /^(active|on|selected|current|is-active|is-selected|pressed|checked)$/i.test(c));
  }
  function controlKind(el) {
    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute('role');
    if (role === 'tab') return 'tab';
    if (tag === 'select') return 'select';
    if (tag === 'input') return el.type === 'range' ? 'range' : (['checkbox', 'radio'].includes(el.type) ? el.type : (['text', 'search', 'number', 'date', 'email'].includes(el.type) ? 'text' : 'button'));
    if (tag === 'a') return 'link';
    if (tag === 'summary') return 'summary';
    return 'button';
  }
  const NATIVE_SEL = 'button, [role="button"], [role="tab"], [role="radio"], [role="switch"], [role="checkbox"], [role="option"], select, input, summary, a[href]';
  P.controls = (opts) => safe(() => {
    const maxItems = (opts && opts.max) || 80;
    const items = [];
    const seenEls = new Set();
    const cands = Array.from(document.querySelectorAll(CONTROL_SEL));
    // custom controls: cursor:pointer leaves without a role (a div.chip with a click listener)
    for (const el of document.body.querySelectorAll('div, span, li, label, td, th, p')) {
      if (el.children.length > 2 || el.closest(NATIVE_SEL)) continue;
      if (getComputedStyle(el).cursor === 'pointer') cands.push(el);
    }
    for (const el of cands) {
      if (seenEls.has(el)) continue;
      seenEls.add(el);
      if (el.closest('script,style,template,noscript') || el.disabled || el.getAttribute('aria-disabled') === 'true') continue;
      if (el.tagName === 'A') { const href = el.getAttribute('href') || ''; if (!href.startsWith('#') && !/^javascript:/i.test(href)) continue; }
      if (!visible(el)) continue;
      const r = el.getBoundingClientRect();
      const kind = controlKind(el);
      if (kind === 'text') continue;
      let labelText = el.getAttribute('aria-label') || el.innerText || '';
      if (!labelText && el.tagName === 'INPUT') {
        const lab = el.closest('label') || (el.id && document.querySelector('label[for="' + cssEscape(el.id) + '"]'));
        labelText = (lab && lab.innerText) || el.title || (kind === 'range' || kind === 'checkbox' || kind === 'radio' ? (el.name || el.id || '') : el.value) || '';
      }
      const label = ((labelText || el.value || el.title || '') + '').trim().replace(/\s+/g, ' ').slice(0, 40);
      const item = { sel: uniqueSel(el), desc: desc(el), tag: el.tagName.toLowerCase(), kind, role: el.getAttribute('role'), label, active: activeState(el), rect: rectOf(el), tabIndex: el.tabIndex,
        controls: el.getAttribute('aria-controls'), dataKeys: Array.from(el.attributes).filter((a) => a.name.startsWith('data-')).map((a) => a.name + '=' + a.value.slice(0, 20)).slice(0, 4) };
      if (kind === 'range') Object.assign(item, { min: +el.min || 0, max: el.max === '' ? 100 : +el.max, step: el.step, value: el.value });
      if (kind === 'select') Object.assign(item, { value: el.value, options: Array.from(el.options).map((o) => o.value).slice(0, 40) });
      // The row it belongs to: the nearest ancestor holding at least one sibling control of the same kind.
      let g = el.parentElement; let group = null;
      for (let depth = 0; g && depth < 3 && g !== document.body; depth++, g = g.parentElement) {
        const same = Array.from(g.querySelectorAll(el.tagName.toLowerCase())).filter((x) => x !== el && controlKind(x) === kind && visible(x));
        if (same.length >= 1) { group = g; break; }
      }
      item.group = group ? uniqueSel(group) : null;
      item.groupDesc = group ? desc(group) : null;
      items.push(item);
      if (items.length >= maxItems) break;
    }
    return items;
  });

  P.tapTargets = () => safe(() => {
    const out = []; let total = 0;
    const seen = new Set();
    const sel = 'button, [role="button"], [role="tab"], [role="radio"], [role="switch"], select, summary, input:not([type="hidden"]), a[href], [onclick]';
    for (const el of document.querySelectorAll(sel)) {
      if (seen.has(el) || el.disabled || !visible(el)) continue;
      seen.add(el);
      const cs = getComputedStyle(el);
      if (el.tagName === 'A' && cs.display === 'inline' && el.closest('p, li, td, span, small')) continue; // inline links in prose
      let box = el;
      if (el.tagName === 'INPUT' && ['checkbox', 'radio'].includes(el.type)) { const lab = el.closest('label') || (el.id && document.querySelector('label[for="' + cssEscape(el.id) + '"]')); if (lab && visible(lab)) box = lab; }
      const r = box.getBoundingClientRect();
      total++;
      if (r.height < 35.5) out.push({ sel: desc(el), label: ((el.getAttribute('aria-label') || el.innerText || el.value || '') + '').trim().replace(/\s+/g, ' ').slice(0, 24), h: +r.height.toFixed(1), w: +r.width.toFixed(1) });
    }
    return { total, small: out.length, items: out.slice(0, 10) };
  });

  // Tab candidates: ARIA tablists, else rows of buttons/links in a nav-like container.
  P.tabCandidates = () => safe(() => {
    const groups = [];
    const lists = Array.from(document.querySelectorAll('[role="tablist"]'));
    for (const tl of lists) {
      const tabs = Array.from(tl.querySelectorAll('[role="tab"]')).filter(visible);
      if (tabs.length >= 2) groups.push({ kind: 'aria', sel: uniqueSel(tl), items: tabs.map((t) => ({ sel: uniqueSel(t), label: (t.innerText || '').trim().slice(0, 30), active: activeState(t), tabIndex: t.tabIndex, controls: t.getAttribute('aria-controls') })) });
    }
    if (!groups.length) {
      const containers = new Set();
      for (const el of document.querySelectorAll('nav, [class*="tab" i], [id*="tab" i], [class*="nav" i], [class*="pager" i], [class*="segment" i]')) containers.add(el);
      for (const c of containers) {
        if (c.closest('[role="tablist"]')) continue;
        const items = Array.from(c.querySelectorAll(':scope > button, :scope > a[href^="#"], :scope > [role="button"], :scope > li > a, :scope > li > button, :scope > div > button, :scope > div > a[href^="#"]')).filter(visible);
        if (items.length < 2 || items.length > 14) continue;
        // one row (or one scrolling strip)
        const tops = items.map((i) => Math.round(i.getBoundingClientRect().top / 8));
        if (new Set(tops).size > 2) continue;
        groups.push({ kind: 'nav', sel: uniqueSel(c), desc: desc(c), items: items.map((t) => ({ sel: uniqueSel(t), label: (t.innerText || '').trim().slice(0, 30), active: activeState(t), tabIndex: t.tabIndex, controls: t.getAttribute('aria-controls') })) });
      }
    }
    const data = Array.from(document.querySelectorAll('[data-page]')).length;
    return { groups, dataPageSections: data, tabpanels: document.querySelectorAll('[role="tabpanel"]').length, sections: document.querySelectorAll('main section, section.panel, section[id]').length };
  });

  // ---------------------------------------------------------------- text for the boilerplate scan
  // Leaf-most text blocks of the rendered document, hidden tab panels included (they are shown on a click).
  P.textBlocks = () => safe(() => {
    const blocks = [];
    const hasTextMemo = new Map();
    const blockMemo = new Map();
    const blockish = (el) => { if (!blockMemo.has(el)) blockMemo.set(el, !['inline', 'contents'].includes(getComputedStyle(el).display)); return blockMemo.get(el); };
    function hasText(el) {
      if (hasTextMemo.has(el)) return hasTextMemo.get(el);
      let r = false;
      for (const n of el.childNodes) {
        if (n.nodeType === 3 && n.nodeValue.trim()) { r = true; break; }
        if (n.nodeType === 1 && !SKIP.has(n.tagName) && hasText(n)) { r = true; break; }
      }
      hasTextMemo.set(el, r); return r;
    }
    const hbMemo = new Map();
    function hasBlockText(el) {
      if (hbMemo.has(el)) return hbMemo.get(el);
      let r = false;
      for (const c of el.children) {
        if (SKIP.has(c.tagName)) continue;
        if (hasText(c) && (blockish(c) || hasBlockText(c))) { r = true; break; }
      }
      hbMemo.set(el, r); return r;
    }
    function textOf(el) {
      let s = '';
      for (const n of el.childNodes) {
        if (n.nodeType === 3) s += n.nodeValue;
        else if (n.nodeType === 1 && !SKIP.has(n.tagName)) s += ' ' + textOf(n);
      }
      return s;
    }
    function walk(el) {
      if (SKIP.has(el.tagName)) return;
      if (!hasText(el)) return;
      if (!hasBlockText(el)) {
        const t = textOf(el).replace(/\s+/g, ' ').trim();
        if (t) blocks.push({ text: t.slice(0, 1200), el: desc(el), path: uniqueSel(el), visible: visible(el) });
        return;
      }
      let own = '';
      for (const n of el.childNodes) if (n.nodeType === 3) own += n.nodeValue;
      own = own.replace(/\s+/g, ' ').trim();
      if (own) blocks.push({ text: own.slice(0, 600), el: desc(el), path: uniqueSel(el), visible: visible(el) });
      for (const c of el.children) walk(c);
    }
    walk(document.body);
    // chart titles are drawn text too
    const titles = [];
    if (window.echarts && window.echarts.getInstanceByDom) {
      for (const el of document.querySelectorAll('*')) {
        let inst = null;
        try { inst = window.echarts.getInstanceByDom(el); } catch (e) { inst = null; }
        if (!inst || (inst.isDisposed && inst.isDisposed())) continue;
        for (const t of chartTexts(inst)) titles.push({ text: t.text, el: 'echarts:' + desc(el), path: uniqueSel(el), visible: visible(el) });
      }
    }
    return blocks.concat(titles);
  });

  // timeline_or_events: entries that BEGIN with a year or a date, inside a list, a table or a .timeline element.
  // Hidden tab panels count (a person reaches them with a click); <template>, <script> and <noscript> do not.
  // Entries: <li> of a ul/ol or role=list; <tr> of a table; the children of an element whose class is `timeline`,
  // `timeline-*`, `*-timeline` or `tl` (an abbreviation real reports use), or whose id contains "timeline".
  // An entry nested in another qualifying entry is counted once, as the outer one.
  P.timeline = () => safe(() => {
    const MONTH = '(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\\.?';
    const YEAR = '(?:1[89]\\d\\d|20\\d\\d)';
    const BEGINS = new RegExp('^\\s*(?:' + YEAR + '(?!\\d)|\\d{1,2}月\\d{1,2}日|' + MONTH + '\\s+(?:\\d{1,2},?\\s*)?' + YEAR + '|\\d{1,2}[/.-]\\d{1,2}[/.-]' + YEAR + ')', 'i');
    // textContent glues neighbours ("2018" + "82" -> "201882"): flatten with a space at every element boundary
    const flat = (el) => { let t = ''; for (const n of el.childNodes) { if (n.nodeType === 3) t += n.nodeValue; else if (n.nodeType === 1 && !SKIP.has(n.tagName)) t += ' ' + flat(n) + ' '; } return t; };
    const norm = (el) => flat(el).replace(/\s+/g, ' ').trim();
    const cands = new Map();
    const add = (el, kind, host) => { if (!cands.has(el)) cands.set(el, { kind, host }); };
    for (const li of document.querySelectorAll('ul > li, ol > li')) add(li, 'list', li.parentElement);
    for (const li of document.querySelectorAll('[role="list"] > [role="listitem"]')) add(li, 'list', li.parentElement);
    for (const tr of document.querySelectorAll('table tr')) add(tr, 'table', tr.closest('table'));
    const isTimelineHost = (el) => {
      const id = (el.id || '').toLowerCase();
      if (id.includes('timeline')) return true;
      const cls = typeof el.className === 'string' ? el.className.toLowerCase().split(/\s+/) : [];
      return cls.some((c) => c === 'tl' || c === 'timeline' || c.startsWith('timeline') || c.endsWith('timeline'));
    };
    for (const host of document.querySelectorAll('*')) {
      if (host.children.length >= 2 && isTimelineHost(host)) for (const child of host.children) add(child, 'timeline', host);
    }
    const hits = [];
    for (const [el, info] of cands) { if (BEGINS.test(norm(el))) hits.push({ el, ...info }); }
    const set = new Set(hits.map((h) => h.el));
    const kept = hits.filter((h) => { for (let a = h.el.parentElement; a; a = a.parentElement) if (set.has(a)) return false; return true; });
    const byHost = new Map();
    for (const h of kept) {
      if (!byHost.has(h.host)) byHost.set(h.host, { kind: h.kind, host: h.host, entries: [] });
      byHost.get(h.host).entries.push(norm(h.el).slice(0, 60));
    }
    const containers = Array.from(byHost.values()).sort((a, b) => b.entries.length - a.entries.length).slice(0, 6).map((c) => ({
      kind: c.kind, desc: desc(c.host), n: c.entries.length, sample: c.entries.slice(0, 3), visible: visible(c.host),
    }));
    return { total: kept.length, containers };
  });

  // Text against its effective background, for the dark-scheme check: the wrapper sets color-scheme "light dark",
  // so a page that paints no background of its own turns dark in a dark browser while its text colours stay put.
  P.contrast = () => safe(() => {
    const de = document.documentElement;
    const prefersDark = matchMedia('(prefers-color-scheme: dark)').matches;
    const parse = (c) => { const m = /rgba?\(([^)]+)\)/.exec(c || ''); if (!m) return null; const p = m[1].split(/[ ,\/]+/).filter(Boolean).map(parseFloat); return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 }; };
    const over = (top, bottom) => { const a = top.a + bottom.a * (1 - top.a); if (!a) return { r: 0, g: 0, b: 0, a: 0 }; return { r: (top.r * top.a + bottom.r * bottom.a * (1 - top.a)) / a, g: (top.g * top.a + bottom.g * bottom.a * (1 - top.a)) / a, b: (top.b * top.a + bottom.b * bottom.a * (1 - top.a)) / a, a }; };
    // the used scheme: the CSS property when the page sets it, else the wrapper's <meta name=color-scheme> ("light dark")
    const cssScheme = getComputedStyle(de).colorScheme || 'normal';
    const meta = document.querySelector('meta[name="color-scheme"]');
    const usedScheme = cssScheme !== 'normal' ? cssScheme : (meta ? meta.content : 'light');
    const schemeDark = prefersDark && /dark/.test(usedScheme);
    let canvas = schemeDark ? { r: 18, g: 18, b: 18, a: 1 } : { r: 255, g: 255, b: 255, a: 1 };
    const htmlBg = parse(getComputedStyle(de).backgroundColor); const bodyBg = parse(getComputedStyle(document.body).backgroundColor);
    if (htmlBg && htmlBg.a > 0) canvas = over(htmlBg, canvas); else if (bodyBg && bodyBg.a > 0) canvas = over(bodyBg, canvas);
    const lum = (c) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); }; return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b); };
    const ratio = (a, b) => { const l1 = lum(a), l2 = lum(b); return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05); };
    // null when an ancestor paints an image or gradient behind the text: the colour under the text is unknown, so it is not judged
    const bgOf = (el) => { const chain = []; for (let e = el; e && e.nodeType === 1; e = e.parentElement) chain.unshift(e); let bg = canvas; for (const e of chain) { const cs = getComputedStyle(e); if (cs.backgroundImage && cs.backgroundImage !== 'none') return null; const c = parse(cs.backgroundColor); if (c && c.a > 0) bg = over(c, bg); } return bg; };
    const out = []; let chars = 0; let low = 0; const seen = new Set();
    const walker = document.createTreeWalker(document.body, 4 /* NodeFilter.SHOW_TEXT: a page may shadow the global */);
    let n; let guard = 0;
    while ((n = walker.nextNode()) && guard++ < 6000) {
      const text = n.nodeValue.trim(); if (!text) continue;
      const el = n.parentElement; if (!el || seen.has(el) || SKIP.has(el.tagName) || el.closest('script,style,template,noscript')) continue;
      seen.add(el);
      const r = el.getBoundingClientRect(); const cs = getComputedStyle(el);
      if (r.width < 3 || r.height < 3 || cs.visibility === 'hidden' || cs.display === 'none') continue;
      const fg0 = parse(cs.color); if (!fg0) continue;
      const bg = bgOf(el); if (!bg) continue; const fg = over(fg0, bg); const cr = ratio(fg, bg);
      chars += text.length;
      if (cr < 3) { low += text.length; if (out.length < 8) out.push({ text: text.slice(0, 24), ratio: +cr.toFixed(2), sel: desc(el), color: cs.color, bg: `rgb(${Math.round(bg.r)},${Math.round(bg.g)},${Math.round(bg.b)})` }); }
    }
    return { schemeDark, prefersDark, canvas: `rgb(${Math.round(canvas.r)},${Math.round(canvas.g)},${Math.round(canvas.b)})`, chars, low, worst: out };
  });

  window.__evalProbe = P;
})();
