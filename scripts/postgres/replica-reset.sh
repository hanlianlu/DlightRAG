#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/postgres/replica-env.sh
source "${SCRIPT_DIR}/replica-env.sh"
dlightrag_load_env_file "${DLIGHTRAG_ENV_FILE:-.env}"

COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-dlightrag}"
DLIGHTRAG_POSTGRES_DATABASE="${DLIGHTRAG_POSTGRES_DATABASE:-dlightrag}"
DLIGHTRAG_POSTGRES_USER="${DLIGHTRAG_POSTGRES_USER:-dlightrag}"
DLIGHTRAG_POSTGRES_REPLICATION_SLOT="${DLIGHTRAG_POSTGRES_REPLICATION_SLOT:-dlightrag_replica}"

if [[ ! "${DLIGHTRAG_POSTGRES_REPLICATION_SLOT}" =~ ^[a-z0-9_]{1,63}$ ]]; then
  echo "DLIGHTRAG_POSTGRES_REPLICATION_SLOT must contain only lowercase letters, digits, and underscores" >&2
  exit 2
fi

docker compose --profile replica rm -sf postgres-replica
docker volume rm "${COMPOSE_PROJECT_NAME}_pg_replica_data" 2>/dev/null || true

if docker compose ps --services --status running | grep -Fxq postgres; then
  docker compose exec -T postgres psql \
    -U "${DLIGHTRAG_POSTGRES_USER}" \
    -d "${DLIGHTRAG_POSTGRES_DATABASE}" \
    -v ON_ERROR_STOP=1 \
    -v "slot=${DLIGHTRAG_POSTGRES_REPLICATION_SLOT}" <<'SQL'
SELECT pg_drop_replication_slot(slot_name)
FROM pg_replication_slots
WHERE slot_name = :'slot'
  AND NOT active;
SQL
else
  echo "Primary postgres is not running; skipping replication slot drop" >&2
fi
