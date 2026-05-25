#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/postgres/replica-env.sh
source "${SCRIPT_DIR}/replica-env.sh"
dlightrag_load_env_file "${DLIGHTRAG_ENV_FILE:-.env}"

DLIGHTRAG_POSTGRES_DATABASE="${DLIGHTRAG_POSTGRES_DATABASE:-dlightrag}"
DLIGHTRAG_POSTGRES_USER="${DLIGHTRAG_POSTGRES_USER:-dlightrag}"

docker compose --profile replica exec -T postgres-replica psql \
  -U "${DLIGHTRAG_POSTGRES_USER}" \
  -d "${DLIGHTRAG_POSTGRES_DATABASE}" \
  -v ON_ERROR_STOP=1 <<'SQL'
DO $$
BEGIN
  IF NOT pg_is_in_recovery() THEN
    RAISE EXCEPTION 'postgres-replica is not in recovery mode';
  END IF;
  IF current_setting('transaction_read_only') <> 'on' THEN
    RAISE EXCEPTION 'transaction_read_only is %, expected on', current_setting('transaction_read_only');
  END IF;
END
$$;

DO $$
BEGIN
  IF (
    SELECT count(*)
    FROM pg_extension
    WHERE extname IN ('vector', 'age', 'pg_textsearch')
  ) <> 3 THEN
    RAISE EXCEPTION 'required extensions vector, age, and pg_textsearch are not all installed';
  END IF;
END
$$;

LOAD 'age';
SET search_path = ag_catalog, "$user", public;
SELECT '[0,1]'::vector <=> '[1,0]'::vector AS pgvector_distance;

DO $$
BEGIN
  IF to_regclass('public.lightrag_vdb_chunks') IS NOT NULL THEN
    PERFORM content_vector <=> content_vector
    FROM public.lightrag_vdb_chunks
    WHERE content_vector IS NOT NULL
    LIMIT 1;
  END IF;
END
$$;

DO $$
BEGIN
  IF to_regclass('public.idx_lightrag_doc_chunks_bm25') IS NOT NULL
    AND to_regclass('public.lightrag_doc_chunks') IS NOT NULL THEN
    PERFORM id
    FROM public.lightrag_doc_chunks
    ORDER BY content <@> to_bm25query('test', 'idx_lightrag_doc_chunks_bm25')
    LIMIT 1;
  END IF;
END
$$;
SQL
