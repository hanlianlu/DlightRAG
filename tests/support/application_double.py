# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One Application double that keeps the real Application's shape.

A hand-rolled ``AsyncMock()`` root with ``SimpleNamespace`` services accepts any
member and any arguments, so a transport could call a method a service no longer
has, or pass arguments the real one rejects, and its tests still passed. This
double is derived from ``Application`` instead: the root admits only its public
attributes, every service admits only the members of the class its property
returns, and each service method is ``create_autospec`` of the real one. Reading
or setting an unknown member raises ``AttributeError``; a call the real signature
rejects raises ``TypeError``.

Services and their methods are built when a test first reaches them, so a double
costs only what the test touches.

A service property annotated with one of the application's own classes, such as a
cursor codec, is autospecced from that class too. A property typed as a builtin
or a generic alias (``bool``, ``str``, ``Mapping[...]``) stays a plain MagicMock,
so a test that reads one assigns it a real value.

Configure a service through its autospecced methods (``return_value``,
``side_effect``); assigning a plain mock or a lambda over one drops the signature
check. Pass a service by keyword only when a test needs a real one. A stateful
hand-built fake stays behind the autospec instead, through ``delegate``.
"""

import inspect
import typing
from types import FunctionType
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


def _class_properties(service_type: type) -> dict[str, type]:
    """Map each property of a service annotated with an application class to that class."""
    properties: dict[str, type] = {}
    for name in dir(service_type):
        member = inspect.getattr_static(service_type, name)
        if name.startswith("_") or not isinstance(member, property):
            continue
        hint = typing.get_type_hints(member.fget).get("return")
        if isinstance(hint, type) and hint.__module__ != "builtins":
            properties[name] = hint
    return properties


_PROPERTY_CLASSES = {kind: _class_properties(kind) for kind in SERVICE_TYPES.values()}


def _is_method(owner: type, name: str) -> bool:
    """Whether ``name`` is a method of ``owner`` other than a dunder, which mocks manage."""
    member = inspect.getattr_static(owner, name, None)
    return not name.startswith("__") and isinstance(
        member, FunctionType | staticmethod | classmethod
    )


def _attach_method(parent: NonCallableMagicMock, owner: type, name: str) -> Any:
    """Autospec one method as an ``owner`` instance has it, and attach it to ``parent``.

    Autospeccing a one-method class builds only that method, bound like the real one
    (``self`` dropped), instead of every member of ``owner``.
    """
    holder = type(owner.__name__, (), {name: inspect.getattr_static(owner, name)})
    method = getattr(create_autospec(holder, instance=True, spec_set=True), name)
    parent.attach_mock(method, name)
    return method


class _ServiceDouble(NonCallableMagicMock):
    """A service spec'd on its class whose members are autospecced on first access."""

    def _get_child_mock(self, /, **kwargs: Any) -> Any:
        name = str(kwargs.get("_new_name"))
        if _is_method(self.__class__, name):
            return _attach_method(self, self.__class__, name)
        value_class = _PROPERTY_CLASSES[self.__class__].get(name)
        if value_class is not None:
            value = create_autospec(value_class, instance=True, spec_set=True)
            self.attach_mock(value, name)
            return value
        return super()._get_child_mock(**kwargs)


class _ApplicationDouble(NonCallableMagicMock):
    """Admits Application's public attributes and builds each on first access."""

    def _get_child_mock(self, /, **kwargs: Any) -> Any:
        name = str(kwargs.get("_new_name"))
        if name in SERVICE_TYPES:
            return _ServiceDouble(spec_set=SERVICE_TYPES[name], **kwargs)
        if _is_method(Application, name):
            return _attach_method(self, Application, name)
        return super()._get_child_mock(**kwargs)


def application_double(config: DlightragConfig, **services: object) -> Any:
    """Return an Application-shaped double that holds ``config``.

    Each service not passed by keyword is built on first access, spec'd on its
    class, and each of its methods is ``create_autospec`` of the real one on first
    access: async methods are AsyncMocks and sync methods MagicMocks, both bound to
    the real signatures. ``astart`` and ``aclose`` are autospecced the same way.
    """
    unknown = sorted(services.keys() - SERVICE_TYPES.keys())
    if unknown:
        raise TypeError(f"Application has no service named {', '.join(unknown)}")
    double = _ApplicationDouble(spec_set=_PUBLIC_ATTRIBUTES, name="application")
    double.config = config
    for name, service in services.items():
        setattr(double, name, service)
    return double


def delegate(service: Any, behaviour: object, *methods: str) -> None:
    """Answer each named autospecced method of ``service`` with ``behaviour``'s method.

    The call still meets the real signature first, so a hand-built fake keeps its
    state and behaviour without accepting calls the real service would refuse. A
    name the real service lacks raises ``AttributeError``.
    """
    for name in methods:
        getattr(service, name).side_effect = getattr(behaviour, name)


__all__ = ["SERVICE_TYPES", "application_double", "delegate"]
