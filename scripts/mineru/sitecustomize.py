# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Startup shim for the MinerU sidecar interpreter (and its spawned workers).

MinerU loads document scans with Pillow, whose default decompression-bomb guard
warns at ~89.5MP and raises ``DecompressionBombError`` at ~179MP. Large
multi-page composite scans (e.g. a ~205MP certificate) therefore fail to load
*before* MinerU can parse them, surfacing as::

    MinerU local parse failed: Failed to load file <name>: Image size (N pixels)
    exceeds limit of 178956970 pixels, could be decompression bomb DOS attack.

DlightRAG targets 32GB+ hosts and intentionally accepts large scans, so raise
the ceiling to match DlightRAG's own ``MAX_DECODE_IMAGE_PIXELS`` (250MP: no
warning up to 250MP; hard error only above 500MP). CPython's ``site`` machinery
imports this module automatically because ``scripts/mineru/api.sh`` puts this
directory on ``PYTHONPATH`` — which MinerU's spawned worker processes inherit.
A failure here cannot break the interpreter: ``site`` catches sitecustomize
errors, warns on stderr, and continues startup with Pillow's default ceiling.

It also repairs MinerU's title-leveling prompts. Both builders show the model an
example dict with unquoted integer keys and then tell it not to format the
output, so a model that obliges returns compact pseudo-JSON like ``{0:2,1:3}``.
``json_repair`` mis-splits that into keys such as ``"3,2"`` and the following
``int(k)`` raises, so every title group burns three streamed LLM calls and ends
with no levels at all. Requesting strict JSON with quoted keys — matching the
shape MinerU already sends — removes the ambiguity. The directive is inserted
before the input block because the prompt ends on the model's answer cue.

MinerU threads a ``prompt_builder`` through ``_request_title_levels`` but its
public ``llm_aided_title`` entry point does not expose it, so there is no
configuration path; the builders are read from module globals at call time,
which makes rebinding them sufficient.

Finally it raises the hybrid parse effort. MinerU ships ``DEFAULT_HYBRID_EFFORT
= "medium"``, which force-disables image/chart analysis
(``_resolve_effective_image_analysis``) and feeds the VLM pipeline-YOLO layout
boxes instead of letting it detect blocks itself. On dense multi-panel figures
that leaves most figures split into per-panel crops with missing or misbound
captions. At ``high`` the same paper yields whole figures with every caption
correct, more extracted text overall, and chart/image content that ``medium``
leaves empty — at roughly five times the parse time. The effort is a hardcoded
constant with no environment override, but the API's ``effort`` form field
binds it as a default at function-definition time, so rebinding the constant
before MinerU is imported changes the server default.

It also widens the HTTP keep-alive. Uvicorn closes idle connections after 5s and
MinerU exposes no flag for it, while LightRAG's MinerU client polls a *pooled*
connection every ``poll_interval_seconds`` (DlightRAG default: 5). The two
deadlines coincide, so each poll is a coin flip on whether the server tears the
connection down just as the client reuses it; httpx then raises
``RemoteProtocolError: Server disconnected without sending a response`` and the
whole ingest fails while MinerU keeps parsing, unaware. Longer parses poll more
often and so fail more reliably — a ~470s hybrid/high parse gets ~94 chances.
The value matches docling-serve, which ships ``timeout_keep_alive = 60``: both
sidecars then behave alike, and the margin covers any sane poll interval instead
of breaking again the moment an operator raises a knob that has no upper bound.
The cost is one idle socket per client held longer, which a loopback sidecar
with a single client does not notice.
"""

from functools import wraps

import uvicorn.config
from PIL import Image

Image.MAX_IMAGE_PIXELS = 250_000_000

# docling-serve's own default; must exceed parser_sidecars.mineru.poll_interval_seconds.
_KEEP_ALIVE_SECONDS = 60
_uvicorn_config_init = uvicorn.config.Config.__init__


@wraps(_uvicorn_config_init)
def _config_with_keep_alive(self, *args, **kwargs):
    kwargs.setdefault("timeout_keep_alive", _KEEP_ALIVE_SECONDS)
    _uvicorn_config_init(self, *args, **kwargs)


uvicorn.config.Config.__init__ = _config_with_keep_alive

import mineru.cli.backend_options as _backend_options  # noqa: E402  # type: ignore[import-not-found]
import mineru.utils.llm_aided as _llm_aided  # noqa: E402  # type: ignore[import-not-found]

_backend_options.DEFAULT_HYBRID_EFFORT = "high"

_PROMPT_INPUT_ANCHOR = "Input title list:"
_STRICT_JSON_DIRECTIVE = (
    "严格要求：只返回合法 JSON。key 必须是带双引号的字符串，与输入字典的 key 完全一致；"
    'value 必须是整数。例如：{"0": 1, "1": 2, "2": 2, "3": 3}\n\n'
)


def _request_strict_json(builder):
    @wraps(builder)
    def build(title_dict):
        prompt = builder(title_dict)
        return prompt.replace(
            _PROMPT_INPUT_ANCHOR, _STRICT_JSON_DIRECTIVE + _PROMPT_INPUT_ANCHOR, 1
        )

    return build


for _name in ("_build_title_optimize_prompt", "_build_relative_title_optimize_prompt"):
    setattr(_llm_aided, _name, _request_strict_json(getattr(_llm_aided, _name)))
