# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the DlightRAG RAGAS adapter."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from dlightrag import config as config_module

_ragas_eval_path = Path(__file__).resolve().parents[2] / "scripts" / "ragas_eval.py"
_spec = importlib.util.spec_from_file_location("ragas_eval", _ragas_eval_path)
assert _spec is not None and _spec.loader is not None
_ragas_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ragas_eval)

DlightRAGAdapterEvaluator = _ragas_eval.DlightRAGAdapterEvaluator
DEFAULT_RESULTS_DIR = _ragas_eval.DEFAULT_RESULTS_DIR
_resolve_eval_env = _ragas_eval._resolve_eval_env
_run = _ragas_eval._run


def test_ragas_eval_parser_requires_dataset() -> None:
    with pytest.raises(SystemExit):
        _ragas_eval.build_parser().parse_args(["--api", "http://localhost:8100"])


@pytest.mark.asyncio
async def test_generate_rag_response_translates_answer_contract(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVAL_QUERY_TOP_K", "7")
    captured: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["payload"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "answer": "grounded answer",
                "contexts": {
                    "chunks": [
                        {"content": "first chunk"},
                        {"content": ""},
                        {"content": 42},
                        {"content": "second chunk"},
                    ]
                },
                "answer_images": [
                    {
                        "id": "fig-1",
                        "source_ref": "1-1",
                        "url": "https://example.test/full.png",
                        "thumbnail_url": "https://example.test/thumb.png",
                    }
                ],
                "answer_blocks": [
                    {"type": "markdown", "text": "grounded answer [1-1]."},
                    {"type": "image_ref", "image_id": "fig-1"},
                ],
            },
        )

    evaluator = DlightRAGAdapterEvaluator.__new__(DlightRAGAdapterEvaluator)
    evaluator.rag_api_url = "https://rag.example"
    evaluator._dlightrag_api_key = "secret"

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await evaluator.generate_rag_response("What changed?", client)

    assert captured["url"] == "https://rag.example/answer"
    assert captured["payload"] == {
        "query": "What changed?",
        "stream": False,
        "top_k": 7,
    }
    assert captured["headers"]["authorization"] == "Bearer secret"
    assert result == {"answer": "grounded answer", "contexts": ["first chunk", "second chunk"]}


def test_resolve_eval_env_keeps_api_autoresolution_when_eval_keys_are_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_cfg = SimpleNamespace(api_key="query-key", model="query-model", base_url="https://llm")
    fake_config = SimpleNamespace(
        llm=SimpleNamespace(roles=SimpleNamespace(query=query_cfg), default=query_cfg),
        embedding=SimpleNamespace(
            provider="openai_compatible",
            api_key="embedding-key",
            base_url="https://embed",
        ),
        api_host="127.0.0.1",
        api_port=8100,
        auth_mode="simple",
        api_auth_token="api-token",
    )
    monkeypatch.setattr(config_module, "DlightragConfig", lambda: fake_config)
    monkeypatch.setenv("EVAL_LLM_BINDING_API_KEY", "judge-key")
    monkeypatch.setenv("EVAL_EMBEDDING_BINDING_API_KEY", "judge-embedding-key")
    monkeypatch.delenv("EVAL_LLM_MODEL", raising=False)
    monkeypatch.delenv("DLIGHTRAG_API_URL", raising=False)
    monkeypatch.delenv("DLIGHTRAG_API_TOKEN", raising=False)

    _resolve_eval_env()

    assert os.environ["EVAL_LLM_BINDING_API_KEY"] == "judge-key"
    assert os.environ["EVAL_EMBEDDING_BINDING_API_KEY"] == "judge-embedding-key"
    assert os.environ["EVAL_LLM_MODEL"] == "query-model"
    assert os.environ["DLIGHTRAG_API_URL"] == "http://127.0.0.1:8100"
    assert os.environ["DLIGHTRAG_API_TOKEN"] == "api-token"


def test_resolve_eval_env_does_not_reuse_native_ollama_as_openai_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query_cfg = SimpleNamespace(api_key=None, model="query-model", base_url=None)
    fake_config = SimpleNamespace(
        llm=SimpleNamespace(roles=SimpleNamespace(query=query_cfg), default=query_cfg),
        embedding=SimpleNamespace(
            provider="ollama",
            api_key="ollama-key",
            base_url="http://127.0.0.1:11434",
        ),
        api_host="127.0.0.1",
        api_port=8100,
        auth_mode="none",
        api_auth_token=None,
    )
    monkeypatch.setattr(config_module, "DlightragConfig", lambda: fake_config)
    for key in (
        "OPENAI_API_KEY",
        "EVAL_LLM_BINDING_API_KEY",
        "EVAL_LLM_BINDING_HOST",
        "EVAL_LLM_MODEL",
        "EVAL_EMBEDDING_BINDING_API_KEY",
        "EVAL_EMBEDDING_BINDING_HOST",
        "DLIGHTRAG_API_URL",
        "DLIGHTRAG_API_TOKEN",
    ):
        monkeypatch.delenv(key, raising=False)

    _resolve_eval_env()

    assert "EVAL_EMBEDDING_BINDING_API_KEY" not in os.environ
    assert "EVAL_EMBEDDING_BINDING_HOST" not in os.environ


@pytest.mark.asyncio
async def test_run_defaults_results_dir_to_project_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset.json"
    dataset.write_text('{"test_cases":[]}', encoding="utf-8")
    captured: dict[str, Path] = {}

    class FakeEvaluator:
        def __init__(
            self,
            test_dataset_path: str | None,
            rag_api_url: str | None,
            *,
            api_key: str | None,
        ) -> None:
            self.test_dataset_path = Path(test_dataset_path or "")
            self.rag_api_url = rag_api_url
            self.api_key = api_key
            self.results_dir = tmp_path / "wrong-parent-default"
            self.eval_model = "judge"
            self.eval_embedding_model = "embed"

        async def run(self) -> None:
            captured["results_dir"] = self.results_dir

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_ragas_eval, "DlightRAGAdapterEvaluator", FakeEvaluator)
    monkeypatch.setattr(_ragas_eval, "_resolve_eval_env", lambda: None)
    monkeypatch.setattr(_ragas_eval, "_check_env", lambda: None)
    monkeypatch.setattr(
        "sys.argv",
        ["ragas_eval.py", "--api", "http://localhost:8100", "--dataset", str(dataset)],
    )

    await _run()

    assert captured["results_dir"] == DEFAULT_RESULTS_DIR
    assert (tmp_path / DEFAULT_RESULTS_DIR).is_dir()
