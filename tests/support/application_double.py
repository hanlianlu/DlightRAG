# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One Application double that keeps the real Application's shape.

A hand-rolled ``AsyncMock()`` root with ``SimpleNamespace`` services accepts any
member and any arguments, so a transport could call a method a service no longer
has, or pass arguments the real one rejects, and its tests still passed. This
double is derived from ``Application`` instead: the root admits only its public
attributes, and every service is ``create_autospec`` of the type its property
returns. Reading or setting an unknown member raises ``AttributeError``; a call
the real signature rejects raises ``TypeError``.

Configure a service through its autospecced methods (``return_value``,
``side_effect``); assigning a plain mock or a lambda over one drops the signature
check. Pass a service by keyword only when a test needs a real or hand-built one.
"""

import inspect
import typing
from typing import Any
from unittest.mock import NonCallableMagicMock, create_autospec

from dlightrag.application import Application
from dlightrag.application.answer_runs import AnswerService
from dlightrag.application.config import DlightragConfig
from dlightrag.application.connections import Connections
from dlightrag.application.corpus_admin import CorpusAdmin, CorpusMutationService
from dlightrag.application.health import ApplicationHealth
from dlightrag.application.memory import MemoryService
from dlightrag.application.model_catalogue import ModelCatalogueAdmin
from dlightrag.application.retrieval import RetrievalService
from dlightrag.application.runs import RunService
from dlightrag.application.web_conversations import WebConversationService

# Application imports its service types under TYPE_CHECKING only, so its property
# annotations resolve against these names. A new service type fails loudly here.
_ANNOTATED_TYPES: dict[str, Any] = {
    service_type.__name__: service_type
    for service_type in (
        AnswerService,
        ApplicationHealth,
        Connections,
        CorpusAdmin,
        CorpusMutationService,
        MemoryService,
        ModelCatalogueAdmin,
        RetrievalService,
        RunService,
        WebConversationService,
    )
}

_PUBLIC_ATTRIBUTES = sorted(name for name in dir(Application) if not name.startswith("_"))


def _service_types() -> dict[str, type]:
    """Map each service property of Application to the type it is annotated to return."""
    services: dict[str, type] = {}
    for name in _PUBLIC_ATTRIBUTES:
        member = inspect.getattr_static(Application, name)
        if isinstance(member, property) and name != "config":
            hints = typing.get_type_hints(member.fget, localns=_ANNOTATED_TYPES)
            services[name] = hints["return"]
    return services


SERVICE_TYPES = _service_types()


def application_double(config: DlightragConfig, **services: object) -> Any:
    """Return an Application-shaped double that holds ``config``.

    Each service not passed by keyword is ``create_autospec(<type>, instance=True,
    spec_set=True)``: its async methods are AsyncMocks and its sync methods
    MagicMocks, both bound to the real signatures. ``astart`` and ``aclose`` are
    autospecced from Application the same way.
    """
    unknown = sorted(services.keys() - SERVICE_TYPES.keys())
    if unknown:
        raise TypeError(f"Application has no service named {', '.join(unknown)}")
    double = NonCallableMagicMock(spec_set=_PUBLIC_ATTRIBUTES, name="application")
    lifecycle = create_autospec(Application, instance=True, spec_set=True, name="application")
    double.astart = lifecycle.astart
    double.aclose = lifecycle.aclose
    double.config = config
    for name, service_type in SERVICE_TYPES.items():
        service = (
            services[name]
            if name in services
            else create_autospec(service_type, instance=True, spec_set=True)
        )
        setattr(double, name, service)
    return double


__all__ = ["SERVICE_TYPES", "application_double"]
