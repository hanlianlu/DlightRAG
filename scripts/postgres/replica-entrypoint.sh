#!/usr/bin/env bash
set -euo pipefail

PGDATA="${PGDATA:-/var/lib/postgresql/data}"
POSTGRES_DB="${POSTGRES_DB:-dlightrag}"
POSTGRES_PRIMARY_HOST="${POSTGRES_PRIMARY_HOST:-postgres}"
POSTGRES_PRIMARY_PORT="${POSTGRES_PRIMARY_PORT:-5432}"
POSTGRES_REPLICATION_USER="${POSTGRES_REPLICATION_USER:-replicator}"
POSTGRES_REPLICATION_PASSWORD="${POSTGRES_REPLICATION_PASSWORD:-dlightrag-replica}"
POSTGRES_REPLICATION_SLOT="${POSTGRES_REPLICATION_SLOT:-dlightrag_replica}"

if [[ -z "${POSTGRES_REPLICATION_PASSWORD}" ]]; then
  echo "POSTGRES_REPLICATION_PASSWORD must not be empty" >&2
  exit 2
fi
if [[ ! "${POSTGRES_REPLICATION_SLOT}" =~ ^[a-z0-9_]{1,63}$ ]]; then
  echo "POSTGRES_REPLICATION_SLOT must contain only lowercase letters, digits, and underscores" >&2
  exit 2
fi

export PGPASSWORD="${POSTGRES_REPLICATION_PASSWORD}"

if [[ ! -s "${PGDATA}/PG_VERSION" ]]; then
  echo "Replica PGDATA is empty; bootstrapping from ${POSTGRES_PRIMARY_HOST}:${POSTGRES_PRIMARY_PORT}"
  case "${PGDATA}" in
    "" | "/" | "/var" | "/var/lib" | "/var/lib/postgresql")
      echo "Refusing to clean unsafe PGDATA: ${PGDATA}" >&2
      exit 2
      ;;
  esac
  rm -rf "${PGDATA:?}/"*

  until pg_isready \
    -h "${POSTGRES_PRIMARY_HOST}" \
    -p "${POSTGRES_PRIMARY_PORT}" \
    -U "${POSTGRES_REPLICATION_USER}" \
    -d "${POSTGRES_DB}"; do
    sleep 2
  done

  psql \
    -h "${POSTGRES_PRIMARY_HOST}" \
    -p "${POSTGRES_PRIMARY_PORT}" \
    -U "${POSTGRES_REPLICATION_USER}" \
    -d "${POSTGRES_DB}" \
    -v ON_ERROR_STOP=1 \
    -c "SELECT CASE WHEN EXISTS (SELECT 1 FROM pg_replication_slots WHERE slot_name = '${POSTGRES_REPLICATION_SLOT}') THEN '${POSTGRES_REPLICATION_SLOT}' ELSE (pg_create_physical_replication_slot('${POSTGRES_REPLICATION_SLOT}')).slot_name END;"

  pg_basebackup \
    -h "${POSTGRES_PRIMARY_HOST}" \
    -p "${POSTGRES_PRIMARY_PORT}" \
    -D "${PGDATA}" \
    -U "${POSTGRES_REPLICATION_USER}" \
    -Fp \
    -Xs \
    -P \
    -R \
    -S "${POSTGRES_REPLICATION_SLOT}"

  chmod 700 "${PGDATA}"
else
  echo "Replica PGDATA already initialized; skipping base backup"
fi

exec docker-entrypoint.sh "$@"
