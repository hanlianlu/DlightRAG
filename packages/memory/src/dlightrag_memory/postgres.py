# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL storage entry point for the Memory package.

Kept out of the package ``__init__`` so importing the core never drags the
database adapter into an embedding host's import graph; hosts that use
PostgreSQL import it directly:

    from dlightrag_memory.postgres import PostgresMemoryStore
"""

from dlightrag_memory._storage.pg import PostgresMemoryStore

__all__ = ["PostgresMemoryStore"]
