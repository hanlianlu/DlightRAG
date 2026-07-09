# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
# PostgreSQL 18 with pgvector + Apache AGE + pg_textsearch for dlightrag.

FROM pgvector/pgvector:pg18

# Pin Apache AGE. An unpinned `--branch PG18` clone tracks the moving dev branch,
# so any image rebuild silently pulls a newer AGE. That upgraded the binary from
# 1.7.0 to 1.8.0 underneath an existing 1.7.0 extension catalog, and the version
# skew broke every graph read ("type with OID 0 does not exist"). Bump this tag
# deliberately, then run `ALTER EXTENSION age UPDATE;` on the database so the
# catalog matches the new binary.
ARG AGE_REF=PG18/v1.8.0-rc0
ARG PG_TEXTSEARCH_REF=v1.3.1
ARG PG_JIEBA_REF=v2.0.1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates postgresql-server-dev-18 libreadline-dev \
    flex bison cmake \
    && update-ca-certificates \
    && git clone --branch ${AGE_REF} --depth 1 https://github.com/apache/age.git /tmp/age \
    && cd /tmp/age \
    && make install \
    && cd / \
    && rm -rf /tmp/age \
    && git clone --branch ${PG_TEXTSEARCH_REF} --depth 1 https://github.com/timescale/pg_textsearch.git /tmp/pg_textsearch \
    && cd /tmp/pg_textsearch \
    && make install \
    && cd / \
    && rm -rf /tmp/pg_textsearch \
    && git clone --branch ${PG_JIEBA_REF} --depth 1 --recurse-submodules https://github.com/jaiminpan/pg_jieba.git /tmp/pg_jieba \
    && cd /tmp/pg_jieba \
    && mkdir build \
    && cd build \
    && PG_SERVER_INCLUDEDIR="$(pg_config --includedir-server)" \
    && PG_INCLUDEDIR="$(pg_config --includedir)" \
    && cmake \
        -DCMAKE_C_FLAGS="-I${PG_SERVER_INCLUDEDIR} -I${PG_INCLUDEDIR}" \
        -DCMAKE_CXX_FLAGS="-I${PG_SERVER_INCLUDEDIR} -I${PG_INCLUDEDIR}" \
        .. \
    && make install \
    && cd / \
    && rm -rf /tmp/pg_jieba \
    && apt-get purge -y build-essential git postgresql-server-dev-18 libreadline-dev flex bison cmake \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Auto-create extensions on database init
COPY init.sql /docker-entrypoint-initdb.d/
