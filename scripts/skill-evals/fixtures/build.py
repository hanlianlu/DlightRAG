"""Builds the checker's own test pages (fixtures/out/*.html): one page that satisfies every check, and one page
per failure the checker must catch. They are built, not committed: the ECharts library (state/toolkit) is inlined."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
TOOLKIT = HERE.parent / "state" / "toolkit"
LIB = (TOOLKIT / "echarts.min.js").read_text(encoding="utf-8").replace("</script", "<\\/script")
THEME = (TOOLKIT / "theme.json").read_text(encoding="utf-8")

SALES = [
    {"q": q, "region": region, "sales": sales, "profit": profit}
    for region, rows in {
        "华东": [(530, 102), (575, 111), (604, 120), (667, 138)],
        "华北": [(470, 80), (495, 87), (519, 94), (570, 106)],
        "华南": [(510, 100), (543, 108), (570, 116), (629, 130)],
        "西南": [(270, 42), (293, 47), (311, 51), (348, 60)],
    }.items()
    for q, (sales, profit) in zip(["2023Q1", "2023Q2", "2023Q3", "2023Q4"], rows, strict=False)
]

GOOD = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>销售看板</title><style>
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:#fcfcfb;color:#222;font:15px/1.6 "Noto Sans SC","PingFang SC",system-ui,sans-serif}
main{max-width:1120px;margin:0 auto;padding:0 16px 40px}
h1{font-size:22px;margin:20px 0 4px}
.lede{margin:0 0 12px;color:#52514e}
.bar{display:flex;gap:8px;overflow-x:auto;padding:8px 0}
.chip,[role=tab]{flex:none;min-height:40px;padding:0 14px;border-radius:20px;border:1px solid #c9c8c0;background:#fff;color:#222;font:inherit}
[role=tab]{border-radius:8px}
.chip[aria-pressed=true],[role=tab][aria-selected=true]{background:#1f5fa8;color:#fff;border-color:#1f5fa8}
.tabs{position:sticky;top:0;background:#fcfcfb;z-index:2;border-bottom:1px solid #e1e0d9}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:16px;margin-top:12px}
figure{margin:0;border:1px solid #e1e0d9;border-radius:10px;padding:12px;background:#fff}
figcaption{font-size:15px;font-weight:600}
figcaption small{display:block;font-weight:400;color:#52514e;font-size:12.5px}
.chart{height:300px;width:100%}
.callout{border-left:4px solid #1f5fa8;background:#f3f6fa;padding:10px 14px;margin:12px 0}
.timeline{list-style:none;padding:0;margin:12px 0}
.timeline li{padding:6px 0;border-bottom:1px solid #eee}
label{display:flex;gap:10px;align-items:center;min-height:44px}
input[type=range]{flex:1;min-height:36px}
[hidden]{display:none}
</style></head><body><main>
<h1>销售数据看板</h1>
<p class="lede">华东与华南贡献过半销售额；四个季度均环比增长。</p>
<div class="bar" id="region" role="group" aria-label="地区">
 <button class="chip" aria-pressed="true" data-v="全部">全部</button><button class="chip" aria-pressed="false" data-v="华东">华东</button>
 <button class="chip" aria-pressed="false" data-v="华北">华北</button><button class="chip" aria-pressed="false" data-v="华南">华南</button>
 <button class="chip" aria-pressed="false" data-v="西南">西南</button>
</div>
<div class="tabs"><div class="bar" role="tablist" aria-label="页面">
 <button role="tab" id="t1" aria-selected="true" aria-controls="p1" tabindex="0">概览</button>
 <button role="tab" id="t2" aria-selected="false" aria-controls="p2" tabindex="-1">推演</button>
 <button role="tab" id="t3" aria-selected="false" aria-controls="p3" tabindex="-1">大事记</button>
</div></div>
<section role="tabpanel" id="p1" aria-labelledby="t1"><div class="grid">
 <figure><figcaption>季度销售额与利润<small>单位：万元 · 来源：内部销售数据</small></figcaption><div class="chart" id="c1"></div></figure>
 <figure><figcaption>各季度销售额构成<small>单位：% · 来源：内部销售数据</small></figcaption><div class="chart" id="c2"></div></figure>
</div><div class="callout">结论：华东最大，西南增速最快。方法：按所选地区汇总四个季度。</div></section>
<section role="tabpanel" id="p2" aria-labelledby="t2" hidden>
 <label>利差（%）<input type="range" id="sp" min="-1" max="2" step="0.1" value="0" aria-label="利差"><output id="spv">0.0</output></label>
 <figure><figcaption>未来四个季度推演<small>单位：万元 · 假设：利差每升高 1 个百分点，销售额提高 5%；不确定性：区间为 ±8%</small></figcaption><div class="chart" id="c3"></div></figure>
</section>
<section role="tabpanel" id="p3" aria-labelledby="t3" hidden>
 <ul class="timeline"><li>1950-05 建交</li><li>1957 政府间贸易协定</li><li>1978 工业与科技合作协定</li><li>2007 胡锦涛访瑞</li><li>2010 吉利收购沃尔沃</li><li>2015 孔子学院关闭</li></ul>
</section>
</main>
<script>__LIB__</script>
<script>
echarts.registerTheme('dlight', __THEME__);
const DATA = __DATA__;
const Q = ['2023Q1','2023Q2','2023Q3','2023Q4'];
let region = '全部';
const charts = {};
const rows = () => region === '全部' ? DATA : DATA.filter(r => r.region === region);
const sum = (q, k) => rows().filter(r => r.q === q).reduce((a, r) => a + r[k], 0);
const base = { animationDuration: 300, textStyle: { fontSize: 12 }, grid: { left: 8, right: 12, top: 36, bottom: 8, containLabel: true } };
function opt1() { return { ...base, legend: { top: 0, textStyle: { fontSize: 12 } }, tooltip: { trigger: 'axis' }, xAxis: { type: 'category', data: Q, axisLabel: { fontSize: 12 } },
  yAxis: { type: 'value', axisLabel: { fontSize: 12 } }, series: [ { name: '销售额', type: 'bar', data: Q.map(q => sum(q, 'sales')) }, { name: '利润', type: 'line', data: Q.map(q => sum(q, 'profit')) } ] }; }
function opt2() { const regs = [...new Set(rows().map(r => r.region))];
  return { ...base, legend: { type: 'scroll', bottom: 0, textStyle: { fontSize: 12 } }, tooltip: { trigger: 'item' },
  series: [{ type: 'pie', radius: ['40%', '65%'], center: ['50%', '45%'], label: { fontSize: 12 }, data: regs.map(g => ({ name: g, value: rows().filter(r => r.region === g).reduce((a, r) => a + r.sales, 0) })) }] }; }
function opt3() { const k = 1 + 0.05 * parseFloat(document.getElementById('sp').value);
  const hist = Q.map(q => sum(q, 'sales')); const last = hist[3];
  const fc = [1, 2, 3, 4].map(i => Math.round(last * (1 + 0.04 * i) * k));
  return { ...base, legend: { top: 0, textStyle: { fontSize: 12 } }, xAxis: { type: 'category', data: ['Q1','Q2','Q3','Q4'], axisLabel: { fontSize: 12 } }, yAxis: { type: 'value', axisLabel: { fontSize: 12 } },
  series: [ { name: '推演', type: 'line', data: fc }, { name: '上沿', type: 'line', data: fc.map(v => Math.round(v * 1.08)), lineStyle: { type: 'dashed' } }, { name: '下沿', type: 'line', data: fc.map(v => Math.round(v * 0.92)), lineStyle: { type: 'dashed' } } ] }; }
const makers = { c1: opt1, c2: opt2, c3: opt3 };
function ensure(id) { if (!charts[id]) charts[id] = echarts.init(document.getElementById(id), 'dlight'); return charts[id]; }
function draw(id) { const c = ensure(id); c.setOption(makers[id](), true); c.resize(); }
function drawVisible() { for (const id of Object.keys(makers)) { const el = document.getElementById(id); if (el.offsetParent !== null) draw(id); } }
document.querySelectorAll('#region .chip').forEach(b => b.addEventListener('click', () => {
  region = b.dataset.v; document.querySelectorAll('#region .chip').forEach(x => x.setAttribute('aria-pressed', String(x === b))); drawVisible(); }));
document.getElementById('sp').addEventListener('input', e => { document.getElementById('spv').textContent = (+e.target.value).toFixed(1); draw('c3'); });
const tabs = [...document.querySelectorAll('[role=tab]')];
function show(i) { tabs.forEach((t, j) => { t.setAttribute('aria-selected', String(i === j)); t.tabIndex = i === j ? 0 : -1; document.getElementById(t.getAttribute('aria-controls')).hidden = i !== j; });
  requestAnimationFrame(() => { drawVisible(); }); }
tabs.forEach((t, i) => { t.addEventListener('click', () => show(i)); t.addEventListener('keydown', e => {
  const n = tabs.length; let k = null;
  if (e.key === 'ArrowRight') k = (i + 1) % n; else if (e.key === 'ArrowLeft') k = (i + n - 1) % n; else if (e.key === 'Home') k = 0; else if (e.key === 'End') k = n - 1;
  if (k !== null) { e.preventDefault(); show(k); tabs[k].focus(); } }); });
addEventListener('resize', () => Object.values(charts).forEach(c => c.resize()));
draw('c1'); draw('c2');
</script></body></html>"""

BAD_HAND = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>手写报告</title><style>
body{margin:0;font:13px/1.5 sans-serif;background:#fff;color:#222}
.wrap{width:640px;margin:0 auto;padding:12px}
.chip{height:24px;font-size:9px;margin:2px;border:1px solid #888;background:#fff}
svg text{font-size:9px}
footer{margin-top:30px;font-size:10px;color:#777}
</style></head><body><div class="wrap">
<h1>销售报告</h1><p>各地区销售额如下。</p>
<div><button class="chip" id="b1">全部</button><button class="chip" id="b2">华东</button></div>
<div id="chart"></div>
<footer>本报告由AI生成，仅供参考，不构成投资建议。© 2026 版权所有 All rights reserved. Powered by ECharts. 生成时间：2026-10-05。欢迎联系我们。
<a href="https://example.com/about">关于</a> <img src="https://example.com/logo.png" width="20" height="20"></footer></div>
<script src="https://cdn.example.com/lib.js"></script>
<script>
function niceTicks(max){ return [0, max/2, max]; }
function barChart(el, data){
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg'); svg.setAttribute('viewBox', '0 0 600 260'); svg.setAttribute('width', '600'); svg.setAttribute('height', '260');
  data.forEach((d, i) => { const r = document.createElementNS(ns, 'rect'); r.setAttribute('x', 60 + i * 120); r.setAttribute('y', 220 - d); r.setAttribute('width', 60); r.setAttribute('height', d); svg.appendChild(r);
    const t = document.createElementNS(ns, 'text'); t.setAttribute('x', 60 + i * 120); t.setAttribute('y', 240); t.textContent = 'Q' + (i + 1); svg.appendChild(t);
    const v = document.createElementNS(ns, 'text'); v.setAttribute('x', 60 + i * 120); v.setAttribute('y', 215 - d); v.textContent = d; svg.appendChild(v);
    const ax = document.createElementNS(ns, 'text'); ax.setAttribute('x', 4); ax.setAttribute('y', 220 - i * 50); ax.textContent = niceTicks(200)[i % 3]; svg.appendChild(ax); });
  el.appendChild(svg);
}
barChart(document.getElementById('chart'), [120, 140, 160, 180]);
document.getElementById('b1').addEventListener('click', () => { document.getElementById('chart').innerHTML = ''; barChart(document.getElementById('chart'), [60, 70, 80, 90]); });
</script>
<script>localStorage.setItem('seen', '1');</script>
<script>alert('loaded');</script>
<script>window.open('https://example.com');</script>
<script>fetch('https://example.com/data.json');</script>
</script></body></html>"""


BAD_OVERLAP = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>标签重叠</title>
<style>body{margin:0;background:#fff;font:15px sans-serif}.card{margin:12px;padding:12px;border:1px solid #ddd}#c{width:100%;height:340px}</style></head><body>
<div class="card"><div id="c"></div></div>
<script>__LIB__</script>
<script>
const chart = echarts.init(document.getElementById('c'));
chart.setOption({
  animation: false,
  title: {text: '中瑞关系事件分布 · 全部议题', subtext: '单位：件（按十年分组）；来源：本报告收录的 47 条已核实事件', top: 4, left: 8},
  legend: {top: 22, left: 8, data: ['政治外交', '经贸投资']},
  xAxis: {type: 'category', name: '事件数', nameLocation: 'end', data: ['1950s','1960s','1970s','1980s','1990s','2000s','2010s','2020s'], axisLabel: {interval: 0, fontSize: 13}},
  yAxis: {type: 'value'},
  series: [{name: '政治外交', type: 'bar', data: [3,0,1,1,0,3,7,32]}, {name: '经贸投资', type: 'bar', data: [1,0,0,1,0,2,5,20]}]
});
</script></body></html>"""


# A report with the shape of the Skill's own pages: sliders and scenario buttons that decide charts through the runtime's event path.
#   * a slider's value label changes at once, but its event goes out once per animation frame (the runtime coalesces a moving slider);
#   * the scenario buttons preset the sliders through the page's own script, then follow the same path;
#   * a handler redraws the charts 100 ms after the last event (650 ms in the slow variant) (a debounce, as slider handlers usually have: without it the window between
#     "label changed" and "chart changed" is one frame, too narrow to pin down) from rows given to `Report.chart(id).setRows`, which a
#     `prepare` step turns into series[].data (no dataset).
# A checker that decides "the control changed nothing" the moment the label differs, or resets the control before the chart has followed,
# sees no effect from any control on this page.
SCENARIO = r"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>汇率情景推演</title><style>
:root{color-scheme:light}
*{box-sizing:border-box}
body{margin:0;background:#fcfcfb;color:#222;font:15px/1.6 "Noto Sans SC","PingFang SC",system-ui,sans-serif}
main{max-width:1000px;margin:0 auto;padding:0 16px 40px}
h1{font-size:22px;margin:20px 0 4px}
.lede{margin:0 0 12px;color:#52514e}
.chips{display:flex;gap:8px;padding:8px 0}
.chip{min-height:40px;padding:0 16px;border-radius:20px;border:1px solid #c9c8c0;background:#fff;color:#222;font:inherit}
.chip[aria-pressed=true]{background:#1f5fa8;color:#fff;border-color:#1f5fa8}
.sliders{display:grid;gap:4px 24px;grid-template-columns:repeat(auto-fit,minmax(min(100%,260px),1fr))}
.slicer{display:grid;grid-template-columns:1fr auto;align-items:center;min-height:44px}
.slicer-value{font-variant-numeric:tabular-nums}
.slicer-range{grid-column:1/-1;min-height:36px;width:100%}
figure{margin:12px 0;border:1px solid #e1e0d9;border-radius:10px;padding:12px;background:#fff}
figcaption{font-size:15px;font-weight:600}
figcaption small{display:block;font-weight:400;color:#52514e;font-size:12.5px}
.chart{height:300px;width:100%}
</style></head><body><main>
<h1>汇率情景推演</h1>
<p class="lede">三个滑块决定中心路径；情景按钮一次设定全部滑块。</p>
<div class="chips" role="group" aria-label="情景">
 <button class="chip" aria-pressed="true" data-s="base">基准</button><button class="chip" aria-pressed="false" data-s="up">乐观</button><button class="chip" aria-pressed="false" data-s="down">悲观</button>
</div>
<div class="sliders">
 <div class="slicer"><span class="slicer-label" id="l-spread">利差（pp）</span><span class="slicer-value" aria-hidden="true" id="v-spread"></span><input class="slicer-range" id="i-spread" type="range" aria-labelledby="l-spread" min="-1" max="2" step="0.05" value="0.75"></div>
 <div class="slicer"><span class="slicer-label" id="l-infl">通胀差（pp）</span><span class="slicer-value" aria-hidden="true" id="v-infl"></span><input class="slicer-range" id="i-infl" type="range" aria-labelledby="l-infl" min="0" max="3" step="0.1" value="0.9"></div>
 <div class="slicer"><span class="slicer-label" id="l-risk">风险溢价（%）</span><span class="slicer-value" aria-hidden="true" id="v-risk"></span><input class="slicer-range" id="i-risk" type="range" aria-labelledby="l-risk" min="0" max="15" step="0.5" value="8.5"></div>
</div>
<figure><figcaption>中心路径<small>单位：CNY · 三个滑块都会改变它</small></figcaption><div class="chart" id="main"></div></figure>
<figure><figcaption>区间宽度<small>单位：% · 只随风险溢价变化</small></figcaption><div class="chart" id="band"></div></figure>
</main>
<script>__LIB__</script>
<script>
(function () {
  const defs = { spread: { digits: 2 }, infl: { digits: 1 }, risk: { digits: 1 } };
  const PRESETS = { base: { spread: 0.75, infl: 0.9, risk: 8.5 }, up: { spread: 1.5, infl: 0.4, risk: 5 }, down: { spread: -0.5, infl: 1.8, risk: 12 } };
  const state = { ...PRESETS.base };
  const chips = [...document.querySelectorAll('.chip')];
  const input = (id) => document.getElementById('i-' + id);
  const shown = (id) => document.getElementById('v-' + id);
  function sync(id) { input(id).value = String(state[id]); shown(id).textContent = state[id].toFixed(defs[id].digits); }
  const pending = new Set();
  function emitLater(id) {
    if (!pending.size) requestAnimationFrame(() => { const ids = [...pending]; pending.clear(); ids.forEach((p) => document.dispatchEvent(new CustomEvent('report:slicer', { detail: { id: p, value: state[p] } }))); });
    pending.add(id);
  }
  function setSlicer(id, value) { state[id] = value; sync(id); emitLater(id); }
  for (const id of Object.keys(defs)) { sync(id); input(id).addEventListener('input', () => setSlicer(id, input(id).valueAsNumber)); }
  chips.forEach((b) => b.addEventListener('click', () => {
    for (const [id, value] of Object.entries(PRESETS[b.dataset.s])) setSlicer(id, value);
    chips.forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
  }));
  const charts = { main: echarts.init(document.getElementById('main')), band: echarts.init(document.getElementById('band')) };
  const months = [...Array(13).keys()];
  for (const c of Object.values(charts)) c.setOption({ animationDuration: 200, grid: { left: 8, right: 16, top: 16, bottom: 8, containLabel: true }, textStyle: { fontSize: 12 },
    xAxis: { type: 'category', data: months.map((m) => 'M' + m), axisLabel: { fontSize: 12 } }, yAxis: { type: 'value', scale: true, axisLabel: { fontSize: 12 } }, series: [{ type: 'line', data: [] }] });
  const prepare = (rows) => rows.map((r) => r.y);
  const Report = { chart: (id) => ({ setRows(rows) { charts[id].setOption({ series: [{ type: 'line', smooth: true, data: prepare(rows) }] }); } }) };
  const centre = () => months.map((m) => ({ y: +(0.668 * (1 + (0.02 * state.spread - 0.01 * state.infl - 0.004 * state.risk) * m / 12)).toFixed(4) }));
  const width = () => months.map((m) => ({ y: +(state.risk * 0.1 * Math.sqrt(m / 12)).toFixed(3) }));
  function update() { Report.chart('main').setRows(centre()); Report.chart('band').setRows(width()); }
  let timer = null;
  document.addEventListener('report:slicer', () => { clearTimeout(timer); timer = setTimeout(update, __DEBOUNCE__); });
  update();
  addEventListener('resize', () => Object.values(charts).forEach((c) => c.resize()));
})();
</script></body></html>"""


def png_chart() -> str:
    im = Image.new("RGB", (1600, 1000), (252, 252, 251))
    d = ImageDraw.Draw(im)
    for i, h in enumerate([420, 480, 510, 600]):
        d.rectangle([260 + i * 300, 800 - h, 460 + i * 300, 800], fill=(42, 120, 214))
    d.text((60, 40), "chart", fill=(0, 0, 0))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


STATIC_IMG = """<!doctype html><html><head><meta charset="utf-8"><title>简报</title><style>body{margin:0;font:15px sans-serif}.card{max-width:900px;margin:20px auto;padding:20px}img{width:100%}</style></head>
<body><div class="card"><h1>2024 简报</h1><img alt="营收趋势" src="data:image/png;base64,__PNG__"><p>来源：用户提供。数据截至 2024-12。</p></div></body></html>"""

IFRAME_CHARTS = r"""<!doctype html><html><head><meta charset="utf-8"><title>嵌入图表</title><style>
body{margin:0;font:15px sans-serif}.box{width:100%;overflow:hidden}iframe{border:0;transform-origin:0 0}</style></head>
<body><h1>嵌入图表页</h1><div class="box" id="box"></div>
<script type="text/plain" id="eclib">__LIBJS__</script>
<script>
const body = '<div id="c" style="width:800px;height:500px"></div><script>' + document.getElementById('eclib').textContent + '<\/script><script>const ch=echarts.init(document.getElementById("c"));ch.setOption({title:{text:"季度销售额",textStyle:{fontSize:20}},legend:{top:60,textStyle:{fontSize:13}},xAxis:{type:"category",data:["Q1","Q2","Q3","Q4"]},yAxis:{type:"value"},series:[{name:"销售额",type:"bar",data:[120,140,160,180]}]});<\/script>';
const page = '<!doctype html><meta charset="utf-8">' + body;
const box = document.getElementById('box'); const f = document.createElement('iframe'); f.setAttribute('srcdoc', page);
const sc = Math.min(1, box.clientWidth / 800); f.style.width = '800px'; f.style.height = '500px'; f.style.transform = 'scale(' + sc + ')'; box.style.height = (500 * sc) + 'px'; box.appendChild(f);
</script></body></html>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    good = (
        GOOD.replace("__LIB__", LIB)
        .replace("__THEME__", THEME)
        .replace("__DATA__", json.dumps(SALES, ensure_ascii=False))
    )
    (OUT / "good-dashboard.html").write_text(good, encoding="utf-8")
    (OUT / "bad-hand-rolled.html").write_text(BAD_HAND, encoding="utf-8")
    (OUT / "bad-overlap.html").write_text(BAD_OVERLAP.replace("__LIB__", LIB), encoding="utf-8")
    (OUT / "static-image.html").write_text(
        STATIC_IMG.replace("__PNG__", png_chart()), encoding="utf-8"
    )
    (OUT / "iframe-charts.html").write_text(
        IFRAME_CHARTS.replace("__LIBJS__", LIB), encoding="utf-8"
    )
    (OUT / "scenario-sliders.html").write_text(
        SCENARIO.replace("__LIB__", LIB).replace("__DEBOUNCE__", "100"), encoding="utf-8"
    )
    # the same page with a slow handler: its charts follow 650 ms after the last event (the checker looks again after a longer wait)
    (OUT / "scenario-sliders-slow.html").write_text(
        SCENARIO.replace("__LIB__", LIB).replace("__DEBOUNCE__", "650"), encoding="utf-8"
    )
    for f in sorted(OUT.glob("*.html")):
        print(f.name, f.stat().st_size)


if __name__ == "__main__":
    main()
