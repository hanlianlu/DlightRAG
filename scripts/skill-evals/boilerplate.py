"""The `no_boilerplate` check: subject-unrelated declarations in the VISIBLE text of a report.

The owner dislikes lines that say nothing about the subject: who or what generated the page, disclaimers,
privacy/compliance/copyright notices, generated-on or powered-by footers, calls to action. The check
scans the rendered document's text blocks (never script or data blobs; hidden tab panels included, since
a click shows them; chart titles drawn by ECharts included) and reports every hit with the phrase, the
category and the element that holds it.

NOT boilerplate, and never matched by anything here: a data source line (`来源：`, `Source:`), an as-of
date (`数据截至`, `as of`), the method, and the assumptions or limits that bear on THIS analysis.

Patterns carry a confidence. A `high` hit fails the check. A `soft` hit is a phrase that is boilerplate
in most documents but can be content in some (`本报告由三部分组成`, a quoted statement that "does not constitute
a commitment"); it is listed and does not fail the check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Pattern:
    id: str
    category: str
    regex: re.Pattern[str]
    confidence: str = "high"  # high | soft


def _p(
    pid: str, category: str, pattern: str, confidence: str = "high", flags: int = re.IGNORECASE
) -> Pattern:
    return Pattern(pid, category, re.compile(pattern, flags), confidence)


_AI_NAMES = r"(?:AI|A\.I\.|人工智能|大模型|大语言模型|语言模型|智能助手|智能体|Claude|ChatGPT|GPT[-\w.]*|DeepSeek|GLM[-\w.]*|Gemini|Kimi|LLM|Copilot|通义千问|文心一言|豆包)"
_AI_VERBS = r"(?:生成|撰写|编写|制作|创作|整理|编制|起草|完成|产出)"
_ADVICE = r"(?:建议|承诺|意见|依据|要约|邀约|担保|推荐)"

PATTERNS: tuple[Pattern, ...] = (
    # -- AI-generation or model/tool credits --------------------------------------------------
    _p(
        "ai-generated-zh",
        "ai_credit",
        r"AI\s*(?:辅助)?\s*(?:生成|撰写|创作|编写|制作|整理)(?!式)",
        flags=re.IGNORECASE,
    ),
    _p(
        "ai-generated-zh-2",
        "ai_credit",
        r"(?:人工智能|大模型|机器|智能)\s*(?:自动)?\s*(?:生成|撰写|创作|编写)(?!式)",
    ),
    _p(
        "by-ai-zh",
        "ai_credit",
        rf"由\s*[^。，,；;\n]{{0,6}}{_AI_NAMES}[^。，,；;\n]{{0,14}}?{_AI_VERBS}(?!式)",
    ),
    _p(
        "ai-generated-en",
        "ai_credit",
        r"\bAI[\s-]*(?:generated|written|created|authored|assisted|produced|composed)\b",
    ),
    _p(
        "generated-by-ai-en",
        "ai_credit",
        r"\b(?:generated|written|created|authored|produced|compiled|drafted|composed|made)\s+(?:by|with|using)\s+(?:an?\s+|the\s+)?"
        r"(?:AI|A\.I\.|LLM|large language model|language model|chatbot|ChatGPT|GPT[-\w.]*|Claude|Gemini|DeepSeek|GLM[-\w.]*|Copilot)\b",
    ),
    _p(
        "this-report-by",
        "ai_credit",
        r"本\s*(?:报告|页面|文档|简报|分析|内容|网页|页)\s*由",
        confidence="soft",
    ),
    # -- disclaimers --------------------------------------------------------------------------
    _p("for-reference-only-zh", "disclaimer", r"仅(?:供|作|用于)\s*(?:参考|学习|研究|交流|内部)"),
    _p("disclaimer-zh", "disclaimer", r"免责\s*(?:声明|申明)"),
    _p("disclaimer-clause-zh", "disclaimer", r"免责\s*(?:条款|说明)", confidence="soft"),
    _p(
        "not-advice-zh",
        "disclaimer",
        rf"(?:本|该|此|以上|上述|所载|所述|页面|报告|文中|文章|简报|内容|信息|分析|数据|预测|推演|结论)[^。！？!?\n]{{0,24}}(?P<tail>不(?:构成|作为|视为)[^。！？!?\n]{{0,16}}{_ADVICE})",
    ),
    _p(
        "not-advice-zh-plain",
        "disclaimer",
        r"不构成[^。！？!?\n]{0,10}(?:投资|法律|财务|税务|医疗|专业|交易|决策)[^。！？!?\n]{0,6}(?:建议|意见|依据|推荐)",
    ),
    _p(
        "not-commitment-zh",
        "disclaimer",
        rf"不构成[^。！？!?\n]{{0,14}}{_ADVICE}",
        confidence="soft",
    ),
    _p(
        "invest-risk-zh",
        "disclaimer",
        r"投资有风险|入市需谨慎|据此(?:操作|投资)[^。\n]{0,8}风险自担|风险自担",
    ),
    _p(
        "use-liability-zh",
        "disclaimer",
        r"因(?:使用|依据|参考|依赖)\s*本[^。\n]{0,10}(?:造成|产生|引起|导致)",
    ),
    _p(
        "no-liability-zh",
        "disclaimer",
        r"(?:概不|不)(?:承担|负)[^。\n]{0,6}(?:任何)?(?:法律)?责任",
        confidence="soft",
    ),
    _p("refer-to-source-zh", "disclaimer", r"请\s*以[^。；;\n]{0,30}为准"),
    _p("refer-to-source-soft-zh", "disclaimer", r"以[^。；;，,\n]{0,16}为准", confidence="soft"),
    _p("disclaimer-en", "disclaimer", r"\bdisclaimer\b"),
    _p(
        "for-reference-only-en",
        "disclaimer",
        r"\bfor\s+(?:reference|informational|information|educational|illustrative|general information|demonstration|entertainment|research)\s+(?:purposes\s+)?only\b",
    ),
    _p(
        "not-advice-en",
        "disclaimer",
        r"\bnot\s+(?:an?\s+)?(?:financial|investment|legal|tax|medical|professional|trading)\s+(?:advice|recommendation)s?\b",
    ),
    _p(
        "not-constitute-en",
        "disclaimer",
        r"\b(?:does|do|should)\s+not\s+constitute\s+[^.\n]{0,40}?(?:advice|recommendation|offer|solicitation)\b",
    ),
    _p("own-risk-en", "disclaimer", r"\bat\s+your\s+own\s+risk\b|\bno\s+(?:warranty|guarantee)\b"),
    # -- privacy / compliance / cookie / copyright ----------------------------------------------
    _p(
        "privacy-zh",
        "privacy_compliance_copyright",
        r"隐私\s*(?:政策|声明|申明|保护|合规|条款|协议)|个人信息\s*(?:保护|处理规则)",
    ),
    _p("compliance-zh", "privacy_compliance_copyright", r"合规\s*(?:声明|披露|说明|提示)"),
    _p("cookie", "privacy_compliance_copyright", r"\bcookies?\b"),
    _p("gdpr", "privacy_compliance_copyright", r"\bGDPR\b", flags=re.IGNORECASE),
    _p("privacy-en", "privacy_compliance_copyright", r"\bprivacy\s+(?:policy|notice|statement)\b"),
    _p(
        "terms",
        "privacy_compliance_copyright",
        r"\bterms\s+(?:of\s+(?:use|service)|and\s+conditions)\b|服务条款|使用条款|用户协议",
    ),
    _p(
        "copyright-zh",
        "privacy_compliance_copyright",
        r"版权\s*(?:所有|声明|申明|归|说明)|保留\s*(?:所有|全部)?\s*权利|严禁\s*(?:转载|复制|传播)|转载请\s*(?:注明|联系)|未经[^。\n]{0,10}(?:许可|授权|同意)[^。\n]{0,10}(?:转载|复制|传播|使用|引用)",
    ),
    _p(
        "copyright-sign",
        "privacy_compliance_copyright",
        r"©|\(c\)\s*(?:19|20)\d\d|\bcopyright\b",
        flags=re.IGNORECASE,
    ),
    _p("rights-reserved", "privacy_compliance_copyright", r"\ball\s+rights\s+reserved\b"),
    # -- footers: generated-on, powered-by, tool credits ----------------------------------------
    _p(
        "generated-on-zh",
        "footer_credit",
        r"生成\s*(?:时间|于|日期)|(?:报告|页面)\s*生成\s*(?:时间|于)",
    ),
    _p(
        "generated-on-en",
        "footer_credit",
        r"\bgenerated\s+(?:on|at)\b|\bgenerated\s+(?:date|time)\b",
    ),
    _p("powered-by", "footer_credit", r"\bpowered\s+by\b|\bbuilt\s+with\b|\bmade\s+with\b"),
    _p(
        "tool-credit-en",
        "footer_credit",
        r"\b(?:created|rendered|drawn|charted|visuali[sz]ed|generated)\s+(?:with|using|by)\s+(?:Apache\s+)?(?:ECharts|Chart\.?js|D3(?:\.js)?|Plotly|Highcharts|matplotlib|Tailwind|React|Vue|DlightRAG)\b",
    ),
    _p(
        "tool-credit-zh",
        "footer_credit",
        r"(?:由|使用|基于|采用)\s*(?:Apache\s+)?(?:ECharts|Chart\.?js|D3(?:\.js)?|Plotly|Highcharts)[^。\n]{0,10}(?:绘制|渲染|生成|制作|构建|驱动|提供)|图表\s*(?:由|使用)[^。\n]{0,12}(?:ECharts|Chart\.?js)|技术支持[:：]",
    ),
    _p("product-name", "footer_credit", r"\bDlightRAG\b"),
    # -- the page describing itself: what it is, how it renders (the real reports end with such a footer) --
    _p(
        "page-self-description-zh",
        "footer_credit",
        r"本页(?:面)?\s*(?:为|是)\s*(?:一份|一个)?[^。；;\n]{0,12}(?:交互式|可视化|交互)[^。；;\n]{0,12}(?:报告|页面|网页|仪表盘|看板)",
    ),
    _p(
        "render-statement-zh",
        "footer_credit",
        r"(?:图表|图形)[^。；;\n]{0,40}?(?:在本地|本地|浏览器(?:内|中)?)\s*渲染|无外部依赖|不依赖\s*(?:任何)?\s*外部|无需联网|可离线",
    ),
    _p(
        "self-contained-en",
        "footer_credit",
        r"\bno\s+external\s+dependenc\w+|\brendered\s+(?:locally|in\s+(?:your|the)\s+browser)\b|\bworks\s+offline\b|\bself-contained\b",
    ),
    # -- calls to action -----------------------------------------------------------------------
    _p(
        "cta-welcome-zh",
        "call_to_action",
        r"欢迎\s*(?:您)?\s*(?:联系|咨询|订阅|关注|反馈|转发|分享|留言|来信|提出|指正|批评|交流|探讨)",
    ),
    _p(
        "cta-contact-zh",
        "call_to_action",
        r"联系我们|如有(?:任何)?\s*(?:疑问|问题|建议|意见|需求)|如需[^。\n]{0,12}(?:咨询|联系|定制|合作|服务|详情)",
    ),
    _p(
        "cta-follow-zh",
        "call_to_action",
        r"扫\s*(?:一扫|码|描)|关注\s*(?:我们|公众号|微信)|(?:点击|立即|马上|请)\s*(?:下载|订阅|咨询|联系|注册)|订阅\s*(?:更新|周报|月报|我们)|获取\s*(?:完整|更多)\s*报告",
    ),
    _p(
        "cta-thanks-zh",
        "call_to_action",
        r"感谢\s*(?:您的)?\s*(?:阅读|关注|观看|使用)|期待\s*(?:您的)?\s*(?:反馈|回复|来信)",
        confidence="soft",
    ),
    _p(
        "cta-en",
        "call_to_action",
        r"\bcontact\s+(?:us|me)\b|\bsubscribe\b|\bfollow\s+us\b|\bget\s+in\s+touch\b|\bsign\s+up\b|\bshare\s+(?:this|on)\b|\bdownload\s+(?:the\s+)?(?:full\s+)?(?:pdf|report)\b|\bthanks?\s+for\s+(?:reading|watching)\b",
    ),
)

_SENTENCE_END = ("。", "；", ";", "！", "？", "!", "?", ". ")

CATEGORIES = (
    "ai_credit",
    "disclaimer",
    "privacy_compliance_copyright",
    "footer_credit",
    "call_to_action",
)

# `本报告由` is soft unless a credit verb or an AI/tool name follows within a short span.
_THIS_REPORT_CREDIT = re.compile(
    rf"^本\s*(?:报告|页面|文档|简报|分析|内容|网页|页)\s*由[^。；;\n]{{0,24}}?(?:{_AI_VERBS}|{_AI_NAMES}|提供|出品|发布|出具)",
    re.IGNORECASE,
)


def scan_text(text: str) -> list[dict[str, str]]:
    """All pattern hits in one text block: id, category, confidence, the phrase and its context."""
    hits: list[dict[str, str]] = []
    seen: set[tuple[str, int]] = set()
    for pattern in PATTERNS:
        for match in pattern.regex.finditer(text):
            key = (pattern.id, match.start())
            if key in seen:
                continue
            seen.add(key)
            confidence = pattern.confidence
            if pattern.id == "this-report-by":
                tail = text[match.start() : match.start() + 60]
                if _THIS_REPORT_CREDIT.match(tail):
                    confidence = "high"
            lo, hi = max(0, match.start() - 24), min(len(text), match.end() + 24)
            # the sentence holding the hit: several patterns can fire inside one sentence, which is one line to a reader
            start = max((text.rfind(d, 0, match.start()) for d in _SENTENCE_END), default=-1) + 1
            ends = [i for i in (text.find(d, match.end()) for d in _SENTENCE_END) if i >= 0]
            end = min(ends) + 1 if ends else len(text)
            hits.append(
                {
                    "pattern": pattern.id,
                    "category": pattern.category,
                    "confidence": confidence,
                    # a pattern with a `tail` group names the disclaimer itself; the words before it only anchor it to the page
                    "phrase": (match.groupdict().get("tail") or match.group(0)),
                    "context": ("…" if lo else "") + text[lo:hi] + ("…" if hi < len(text) else ""),
                    "sentence": text[start:end].strip()[:200],
                }
            )
    return hits


def scan_blocks(blocks: list[dict], frame_label: str = "") -> list[dict]:
    """Scan the text blocks a probe returned; one record per hit, with the holding element."""
    out: list[dict] = []
    for block in blocks:
        for hit in scan_text(block.get("text", "")):
            out.append(
                {
                    **hit,
                    "element": block.get("el", ""),
                    "path": block.get("path", ""),
                    "visible": bool(block.get("visible", True)),
                    "frame": frame_label,
                }
            )
    return out


def dedupe(hits: list[dict]) -> list[dict]:
    """Merge the same hit seen on successive scans (initial load, after each tab)."""
    seen: set[tuple[str, str, str, str]] = set()
    out = []
    for hit in hits:
        key = (hit["pattern"], hit["phrase"], hit["path"], hit["context"])
        if key not in seen:
            seen.add(key)
            out.append(hit)
    return out
