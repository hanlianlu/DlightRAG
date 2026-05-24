# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
# PostgreSQL 18 with pgvector + Apache AGE + pg_textsearch for dlightrag.

FROM pgvector/pgvector:pg18

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates postgresql-server-dev-18 libreadline-dev \
    flex bison \
    && update-ca-certificates \
    && git clone --branch PG18 --depth 1 https://github.com/apache/age.git /tmp/age \
    && cd /tmp/age \
    && make install \
    && cd / \
    && rm -rf /tmp/age \
    && git clone --depth 1 https://github.com/timescale/pg_textsearch.git /tmp/pg_textsearch \
    && cd /tmp/pg_textsearch \
    && make install \
    && cd / \
    && rm -rf /tmp/pg_textsearch \
    && apt-get purge -y build-essential git postgresql-server-dev-18 libreadline-dev flex bison \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Auto-create extensions on database init
COPY init.sql /docker-entrypoint-initdb.d/
