#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/postgres/replica-env.sh
source "${SCRIPT_DIR}/replica-env.sh"
dlightrag_load_env_file "${DLIGHTRAG_ENV_FILE:-.env}"

DLIGHTRAG_POSTGRES_DATABASE="${DLIGHTRAG_POSTGRES_DATABASE:-dlightrag}"
DLIGHTRAG_POSTGRES_USER="${DLIGHTRAG_POSTGRES_USER:-dlightrag}"
DLIGHTRAG_POSTGRES_REPLICATION_USER="${DLIGHTRAG_POSTGRES_REPLICATION_USER:-replicator}"
DLIGHTRAG_POSTGRES_REPLICATION_PASSWORD="${DLIGHTRAG_POSTGRES_REPLICATION_PASSWORD:-dlightrag-replica}"
export DLIGHTRAG_POSTGRES_REPLICATION_USER
export DLIGHTRAG_POSTGRES_REPLICATION_PASSWORD

if [[ -z "${DLIGHTRAG_POSTGRES_REPLICATION_USER}" || -z "${DLIGHTRAG_POSTGRES_REPLICATION_PASSWORD}" ]]; then
  echo "DLIGHTRAG_POSTGRES_REPLICATION_USER/PASSWORD must not be empty" >&2
  exit 2
fi

docker compose up -d postgres

sql="$(
  python3 - <<'PY'
import os

role = os.environ["DLIGHTRAG_POSTGRES_REPLICATION_USER"]
password = os.environ["DLIGHTRAG_POSTGRES_REPLICATION_PASSWORD"]
role_lit = "'" + role.replace("'", "''") + "'"
password_lit = "'" + password.replace("'", "''") + "'"
body = (
    "BEGIN "
    f"IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = {role_lit}) THEN "
    f"EXECUTE format('CREATE ROLE %I WITH REPLICATION LOGIN PASSWORD %L', {role_lit}, {password_lit}); "
    "ELSE "
    f"EXECUTE format('ALTER ROLE %I WITH REPLICATION LOGIN PASSWORD %L', {role_lit}, {password_lit}); "
    "END IF; "
    "END;"
)
print(f"DO $dlightrag_repl$ {body} $dlightrag_repl$;")
PY
)"

docker compose exec -T postgres psql \
  -U "${DLIGHTRAG_POSTGRES_USER}" \
  -d "${DLIGHTRAG_POSTGRES_DATABASE}" \
  -v ON_ERROR_STOP=1 <<SQL
${sql}
SQL

docker compose exec -T \
  -e "POSTGRES_REPLICATION_USER=${DLIGHTRAG_POSTGRES_REPLICATION_USER}" \
  postgres bash -ceu '
hba_file="${PGDATA:-/var/lib/postgresql/data}/pg_hba.conf"
quoted_user="${POSTGRES_REPLICATION_USER//\"/\"\"}"
rule="host replication \"${quoted_user}\" samenet scram-sha-256"
if ! grep -Fxq "${rule}" "${hba_file}"; then
  printf "\n# DlightRAG streaming read replica\n%s\n" "${rule}" >> "${hba_file}"
fi
'

docker compose exec -T postgres psql \
  -U "${DLIGHTRAG_POSTGRES_USER}" \
  -d "${DLIGHTRAG_POSTGRES_DATABASE}" \
  -v ON_ERROR_STOP=1 <<'SQL'
SELECT pg_reload_conf();
SQL
