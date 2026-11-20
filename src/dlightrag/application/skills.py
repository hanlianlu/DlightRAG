# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Application-level Agent Skill slice for transports and composition."""

import logging
import threading
import time
from functools import lru_cache
from pathlib import Path

from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.settings.projections import (
    agent_skills_root,
    disabled_builtin_skills,
    owner_skills_root,
)
from dlightrag.engine.agent.environment.confinement import DeclaredLayer
from dlightrag.engine.agent.skill_protection import ProtectedTextIndex
from dlightrag.engine.agent.skills import (
    SkillCatalog,
    SkillsBundleFactory,
    builtin_skills_root,
    owner_skill_root,
)


def skill_read_layers(config: DlightragConfig, *, owner_id: str) -> tuple[DeclaredLayer, ...]:
    """Return the Skill roots this owner's Agent processes read.

    A Skill may point at an executable asset, so the roots have to be readable from
    inside the confinement; the declaration lives here, beside the capability that
    needs it, rather than in a list the execution environment owns (ADR 0024). The
    owner's own shard is the unit, never the shared parent: one owner's skills are that
    owner's, and an Agent must not be able to read a sibling owner's.

    The packaged built-ins are absent on purpose: the serving process loads them
    through its own ``load_skill`` tool, and their assets are reachable where the
    install puts them under the runtime prefix.
    """
    return (
        DeclaredLayer(path=agent_skills_root(config), capability="skills (operator-global)"),
        DeclaredLayer(
            path=owner_skills_directory(config, owner_id=owner_id),
            capability="skills (this owner)",
        ),
    )


def owner_skills_directory(config: DlightragConfig, *, owner_id: str) -> Path:
    """Return the directory of one owner's own Skills, which their management acts on.

    It is the shard the owner's Agent processes read (``skill_read_layers``), so the Skills a
    Settings page lists and changes are the ones the Agent finds.
    """
    return owner_skill_root(owner_skills_root(config), owner_id)


def skill_roots(config: DlightragConfig) -> tuple[tuple[Path, str], ...]:
    """Return every root Skill lookup may serve, for composition's refusal to check.

    Composition refuses each of these against the trees an Agent may never see, so a
    deployment that points a Skill root at its own project fails at startup rather
    than the first Run that happens to announce that owner.
    """
    return (
        (agent_skills_root(config), "skills (operator-global)"),
        (owner_skills_root(config), "skills (owner shards)"),
    )


def skills_bundle_factory(
    config: DlightragConfig,
    *,
    ensure_dirs: bool = False,
) -> SkillsBundleFactory:
    """Build the per-run skills slice for one application config.

    Resolves both roots once; composition passes ``ensure_dirs=True`` to create
    them eagerly, request paths leave it False.
    """
    builtin_root = builtin_skills_root()
    global_root = agent_skills_root(config)
    owner_root = owner_skills_root(config)
    disabled_builtins = disabled_builtin_skills(config)
    if ensure_dirs:
        global_root.mkdir(parents=True, exist_ok=True)
        owner_root.mkdir(parents=True, exist_ok=True)

    return SkillsBundleFactory(
        builtin_root=builtin_root,
        global_root=global_root,
        owner_root=owner_root,
        disabled_builtin_skills=disabled_builtins,
    )


logger = logging.getLogger(__name__)

_MASK_CACHE_ENTRIES = 512
"""How many recently masked strings the gate keeps.

A followed transcript is re-read about once a second and answers a page of the same messages;
the cache is what turns that from a re-scan of every passage into a dictionary lookup. Entries
are whole projected strings, so the bound is also the memory bound: a few pages a reader is
looking at, not a document corpus. The bound must cover one Run-timeline page's whole working
set (one page of 100 steps masks ~150 content, reasoning and argument strings), or a warm page
re-masks itself from cold every time. Cleared whenever the index is rebuilt.
"""

_REVALIDATE_SECONDS = 30.0
"""How long a built index serves before its roots are looked at again.

A Skill root changes when an operator drops a global Skill in or a built-in is replaced by a
deployment, which is rare; an index costs a few hundred milliseconds to build. The window
makes a change visible within it, without putting a directory walk on every reader request.
"""


class SkillProtection:
    """This process's content fingerprints of the protected Skill documents.

    A reader-facing surface asks one question — "is this passage a protected document's own
    text?" — and this is the one object that answers it. It holds the built
    :class:`~dlightrag.engine.agent.skill_protection.ProtectedTextIndex` for the whole
    process, rebuilt when the protected roots move (:data:`_REVALIDATE_SECONDS`).

    One instance per deployment, beside the Skill factory it reads: two instances would be two
    index builds and two answers to the same question.
    """

    def __init__(
        self, factory: SkillsBundleFactory, *, revalidate_seconds: float = _REVALIDATE_SECONDS
    ) -> None:
        self._factory = factory
        self._revalidate_seconds = revalidate_seconds
        self._lock = threading.Lock()
        self._index: ProtectedTextIndex | None = None
        self._publication_index: ProtectedTextIndex | None = None
        self._signature: tuple[tuple[str, int, int], ...] | None = None
        self._built_at = 0.0
        self._stale = False
        self._documents: frozenset[tuple[str, str]] | None = None
        self._documents_at = 0.0
        self._mask_cached = lru_cache(maxsize=_MASK_CACHE_ENTRIES)(self._mask_uncached)

    def documents(self) -> frozenset[tuple[str, str]]:
        """The ``(name, path)`` of every document a Run may load but not hand over.

        Reader surfaces ask this on every page read to decide which ``load_skill`` result is a
        protected document. Discovery walks both protected roots, so the answer is kept for the
        same window as the text index (:data:`_REVALIDATE_SECONDS`) rather than re-derived per
        request: a protected document added to an operator root stays visible within that window,
        exactly as the text index's revalidation promises.
        """
        now = time.monotonic()
        if self._documents is not None and now - self._documents_at < self._revalidate_seconds:
            return self._documents
        with self._lock:
            now = time.monotonic()
            if self._documents is not None and now - self._documents_at < self._revalidate_seconds:
                return self._documents
            self._documents = self._factory.protected_documents()
            self._documents_at = now
            return self._documents

    def mask(self, text: str) -> str:
        """Replace every verified protected passage in ``text`` with ``[...]``.

        The one call a reader surface makes. Returns ``text`` itself when nothing matched, so
        a hot path can skip the copy; a string masked before is answered from the cache.

        An index that is still unbuilt, or empty because its root could not be read yet, is
        not cached: its answer is "nothing is protected", and caching that would keep serving
        it after the index recovered.
        """
        if self._stale or self._index is None or not self._index:
            return self.index.mask(text)
        return self._mask_cached(text)

    def _mask_uncached(self, text: str) -> str:
        return self.index.mask(text)

    @property
    def index(self) -> ProtectedTextIndex:
        now = time.monotonic()
        if self._index is not None and now - self._built_at < self._revalidate_seconds:
            return self._index
        with self._lock:
            now = time.monotonic()
            if self._index is not None and now - self._built_at < self._revalidate_seconds:
                return self._index
            signature = self._root_signature()
            if signature is None:
                # A root that cannot be read is not evidence that nothing is protected. The
                # last index keeps serving; with none yet, the built-ins (which cannot become
                # unreadable) are indexed on their own and the operator root is retried after
                # the window. Rebuilding from an empty discovery instead would silently
                # unmask every operator-provided document for the window's length.
                logger.warning(
                    "The operator Skill root could not be read; keeping the last protected "
                    "text index"
                )
                self._built_at = now
                self._stale = True
                if self._index is None:
                    self._index = self._build(builtin_only=True) or ProtectedTextIndex.empty()
                    self._signature = None
                return self._index
            if self._index is not None and signature == self._signature:
                self._built_at = now
                return self._index
            built = self._build()
            if built is None:
                # The roots moved but the new index could not be built: keep the last one and
                # leave the signature alone, so the next window tries again. Advancing the
                # signature here would record a root the index does not cover, and a Skill
                # added during the failure would stay unprotected until something else moved.
                self._stale = True
                self._built_at = now
                return self._index if self._index is not None else ProtectedTextIndex.empty()
            self._index = built
            self._signature = signature
            self._built_at = now
            self._stale = False
            self._mask_cached.cache_clear()
            # The pair set is derived from the same roots: a rebuild is the one moment it can
            # have moved, so it goes with the index rather than ageing on its own clock.
            self._documents = None
            return self._index

    @property
    def publication_index(self) -> ProtectedTextIndex:
        """The prose-only index the artifact publication gate asks for.

        The reader mask deliberately covers every protected document, including executable
        kit code; an Artifact's text-copy gate instead covers prose only, because the engine
        and verifier strings are written into the case/page those scripts produce. This index
        is built beside the reader index from the same discovery and revalidation, but is
        consulted only by ``prepare_artifact_attachment``.
        """
        # Refreshing the reader index also refreshes this one on the same root signature.
        _ = self.index
        return (
            self._publication_index
            if self._publication_index is not None
            else ProtectedTextIndex.empty()
        )

    def _build(self, *, builtin_only: bool = False) -> ProtectedTextIndex | None:
        """Index the protected documents this process can read now, or None on a read failure."""
        try:
            catalog = SkillCatalog.discover(
                builtin_root=self._factory.builtin_root,
                global_root=None if builtin_only else self._factory.global_root,
                disabled_builtin_skills=self._factory.disabled_builtin_skills,
            )
            reader = ProtectedTextIndex.build(catalog.protected_texts())
            publication = ProtectedTextIndex.build(catalog.publication_texts())
        except OSError:
            logger.warning("The protected Skill text index could not be built", exc_info=True)
            return None
        self._publication_index = publication
        logger.info(
            "Protected Skill text index built: %d reader documents, %d characters; "
            "%d publication-prose documents, %d characters",
            reader.document_count,
            reader.character_count,
            publication.document_count,
            publication.character_count,
        )
        return reader

    def _root_signature(self) -> tuple[tuple[str, int, int], ...] | None:
        """What the operator-provided root holds, or None when it cannot be read.

        The packaged built-ins are part of the release and cannot change under a running
        process, so only the operator-global root is walked. A file's size and modification
        time are the cheap facts that name a change; a walk of one small directory is what
        makes an edited document visible without a restart. None is distinct from an empty
        tuple on purpose: "there are no operator documents" and "I cannot see them" must not
        produce the same index.
        """
        root = self._factory.global_root
        if root is None:
            return ()
        try:
            signature = [
                (path.relative_to(root).as_posix(), info.st_mtime_ns, info.st_size)
                for path in root.rglob("*")
                if (info := path.stat()).st_size >= 0 and path.is_file()
            ]
        except OSError:
            return None
        return tuple(sorted(signature))


def identity_protection() -> SkillProtection:
    """A gate over no documents: what a slice composed without Skill roots uses.

    A unit test that builds the application components directly, or a process that composes
    no Skill capability at all, still gets a working object rather than a branch: the index
    is empty and every projection passes through unchanged.
    """
    return SkillProtection(SkillsBundleFactory())


__all__ = [
    "SkillProtection",
    "identity_protection",
    "owner_skills_directory",
    "skill_read_layers",
    "skill_roots",
    "skills_bundle_factory",
]
