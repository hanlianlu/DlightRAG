"""The two heuristic checks cases.json defines in words, pinned page by page. Needs the browser, not the stack.

  timeline_or_events: the document holds at least five entries that begin with a year or a date, inside a list, a table or a .timeline element
  assumptions_stated: visible text has a heading or lead-in containing 假设 or assumption(s), followed by at least one sentence with a number

Each case is a small page and the verdict the definition gives it. Run: .venv/bin/python selftest_defs.py
"""

from __future__ import annotations

import sys

import checker
from frame import load_product_frame
from playwright.sync_api import sync_playwright

LONG = "这是一段很长的说明文字，用来说明本页面的数据口径、来源与编制方法，" * 3


def years(tag_open: str, tag_close: str, items: list[str]) -> str:
    return "".join(f"{tag_open}{text}{tag_close}" for text in items)


YEARS5 = ["1950 建交", "1957 贸易协定", "1978 合作协定", "2010 收购", "2015 关闭"]

TIMELINE_CASES: list[tuple[str, str, bool]] = [
    (
        "ul with five entries that begin with a year",
        f"<ul>{years('<li>', '</li>', YEARS5)}</ul>",
        True,
    ),
    ("ol counts too", f"<ol>{years('<li>', '</li>', YEARS5)}</ol>", True),
    ("four entries are not enough", f"<ul>{years('<li>', '</li>', YEARS5[:4])}</ul>", False),
    (
        "a table whose rows begin with a year",
        "<table>" + years("<tr><td>", "</td><td>x</td></tr>", YEARS5) + "</table>",
        True,
    ),
    (
        "children of a .timeline element",
        f"<div class='timeline'>{years('<div>', '</div>', ['1950-05-09 建交', '1957-11-08 协定', '1978-12-05 协定', '2010-03-28 收购', '2015-01-01 关闭'])}</div>",
        True,
    ),
    (
        "`tl` is accepted as the abbreviation of .timeline (the real reports use div.tl)",
        "<div class='tl'>"
        + years(
            "<div class='ev'><div class='when'>",
            "</div><div>事件</div></div>",
            ["1950-01-14", "1950-05-09", "1957-11-08", "1978-12-05", "1981 / 2006"],
        )
        + "</div>",
        True,
    ),
    (
        "a plain div list is not a list, a table or a timeline",
        f"<div class='events'>{years('<div>', '</div>', YEARS5)}</div>",
        False,
    ),
    (
        "a year inside the text is not a year at the beginning",
        "<ul>"
        + years(
            "<li>",
            "</li>",
            [
                "建交于 1950 年",
                "协定签于 1957 年",
                "合作始于 1978 年",
                "收购在 2010 年",
                "关闭于 2015 年",
            ],
        )
        + "</ul>",
        False,
    ),
    (
        "entries in a hidden tab panel count (a click shows them)",
        f"<section style='display:none'><ul>{years('<li>', '</li>', YEARS5)}</ul></section>",
        True,
    ),
    (
        "an entry nested in an entry counts once",
        "<ul>" + "".join(f"<li>{y}<ul><li>{y} 细节</li></ul></li>" for y in YEARS5[:3]) + "</ul>",
        False,
    ),
    (
        "other ways to begin with a date",
        "<ul>"
        + years(
            "<li>",
            "</li>",
            [
                "3月28日 发布",
                "Mar 2015 关闭",
                "2024-03-05 签署",
                "5/6/2019 会议",
                "March 28, 2010 收购",
            ],
        )
        + "</ul>",
        True,
    ),
    (
        "built by a script after load",
        "<ul id='t'></ul><script>setTimeout(() => { document.getElementById('t').innerHTML = "
        + repr(years("<li>", "</li>", YEARS5))
        + "; }, 60)</script>",
        True,
    ),
    (
        "a data table whose rows begin with a year also qualifies (the definition is structural)",
        "<table>"
        + years("<tr><td>", "</td><td>82</td><td>98</td></tr>", [str(y) for y in range(2018, 2025)])
        + "</table>",
        True,
    ),
]

ASSUMPTION_CASES: list[tuple[str, str, bool]] = [
    (
        "a heading followed by a sentence with a number",
        "<h2>假设</h2><ul><li>利差每升高 1 个百分点，销售额提高 5%</li></ul>",
        True,
    ),
    (
        "a lead-in that starts with the word and carries the number",
        "<p>假设：基准利差 0.5%，通胀 3%。</p>",
        True,
    ),
    ("English heading and lead-in", "<h3>Assumptions</h3><p>Growth is 5% a year.</p>", True),
    (
        "a heading with no sentence with a number after it",
        "<h2>Assumptions</h2><p>The model is deliberately simple.</p>",
        False,
    ),
    (
        "a mention inside a long paragraph is not a heading or lead-in",
        f"<p>{LONG}数据口径与假设见第 5 节，另有 3 项说明。</p>",
        False,
    ),
    (
        "the number comes after the next heading",
        "<h3>关键假设</h3><p>无。</p><h3>其他</h3><p>共 5 项数字</p>",
        False,
    ),
    (
        "the number sentence is a few blocks later but before the next heading",
        "<h3>主要假设</h3><p>以下为设定。</p><p>说明。</p><p>通胀预期 2.5%。</p>",
        True,
    ),
    (
        "a bold lead-in inside a note",
        "<div class='note'><strong>Assumption:</strong> growth is 5% a year</div>",
        True,
    ),
    (
        "a short non-heading label followed by a block with a number",
        "<div>情景假设</div><div>基准 3%</div>",
        True,
    ),
    (
        "a page that merely describes itself is not a lead-in (the word ends a short subtitle; a number follows)",
        "<p>基准 / 乐观 / 悲观三情景，可用滑块调整利差；页面给出推演区间图、结论摘要与全部假设。</p><p>起点即期：1 SEK = 0.668 CNY</p>",
        False,
    ),
    (
        "a method note that merely denies a definition is not a lead-in (the real t1 footnote: the word stands past the first 30 characters)",
        "<ul><li>金额单位统一为万元；“利润率”按 利润 ÷ 销售额 × 100 计算，报告未假设利润的其他定义（毛利、营业利润、净利等）。</li><li>数据只覆盖 2023 年 Q1–Q4，共 32 条</li></ul>",
        False,
    ),
    (
        "the word within the first 30 characters of a block is a lead-in, whatever the block goes on to say",
        "<p>当前设定下：瑞典 CPIF 假设约 2.0%，中国 CPI 假设约 1.0%（8 月实际 +0.8%，核心 +1.0%）</p>",
        True,
    ),
    (
        "a short block ending in a colon introduces what follows, wherever the word stands",
        "<p>本推演的数字基于下面这些关于利率与通胀走势的全部假设：</p><ul><li>利差 0.5%</li></ul>",
        True,
    ),
    (
        "the full-width colon counts too",
        "<p>以下推演所依据的利率、通胀、风险溢价与汇率路径的全部假设：</p><p>利差 0.5%</p>",
        True,
    ),
    (
        "the strongest evidence wins: a real heading is reported although a description comes first",
        "<p>页面给出区间图与全部假设。</p><p>起点 0.668。</p><h2>公式与假设</h2><p>利差每升高 1 个点，汇率变动 0.5%。</p>",
        True,
    ),
    (
        "assumptions in a hidden tab panel",
        "<section style='display:none'><h2>假设</h2><p>利差为 0.5%。</p></section>",
        True,
    ),
    ("no mention at all", "<h2>结论</h2><p>销售额增长 12%。</p>", False),
    (
        "a number in the keyword block's own heading counts as following text only after the keyword",
        "<h2>3. 假设</h2><p>说明文字。</p>",
        False,
    ),
]


def main() -> int:
    product = load_product_frame()
    chk = checker.Checker(out_dir=None, viewports=(1280,), heights=(800,), product=product)
    failures: list[str] = []
    total = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            for title, cases, which in (
                ("timeline_or_events", TIMELINE_CASES, "timeline"),
                ("assumptions_stated", ASSUMPTION_CASES, "assumptions"),
            ):
                for name, html, expected in cases:
                    total += 1
                    sess = checker.Session(browser, product, 1280, 800)
                    try:
                        sess.load(f"<!doctype html><meta charset='utf-8'><body>{html}</body>")
                        heur = {
                            "timeline": sess.top_call("timeline"),
                            "assumption_blocks": chk._plain_blocks(sess),
                        }  # noqa: SLF001
                        result = (
                            chk.check_timeline(heur, [])
                            if which == "timeline"
                            else chk.check_assumptions(heur)
                        )
                    finally:
                        sess.close()
                    got = result.status == "pass"
                    flag = "ok  " if got == expected and result.heuristic else "FAIL"
                    if name.startswith(
                        "the strongest evidence wins"
                    ) and not result.reason.startswith("heading '公式与假设'"):
                        flag = "FAIL"
                    print(f"{flag} {title}: {name} -> {result.status}: {result.reason[:110]}")
                    if flag == "FAIL":
                        failures.append(
                            f"{title}: {name}: expected {'pass' if expected else 'fail'}, got {result.status} ({result.reason})"
                        )
        finally:
            browser.close()
    print(f"\n{total - len(failures)}/{total} definition cases as the definitions say")
    for failure in failures:
        print("  ", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
