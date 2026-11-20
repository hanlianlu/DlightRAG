#!/usr/bin/env bash
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
#
# Recover the Docker Compose PostgreSQL from the lock files an unclean shutdown leaves behind.
# The symptom is a container that restarts forever while PostgreSQL never comes up, and the FATAL
# below; `docker logs` can be empty at the same time, because the same shutdown corrupts the
# container's own log file, so read it with `docker start -a dlightrag-postgres-1`:
#
#   FATAL: bogus data in lock file "postmaster.pid": ""
#
# An unclean shutdown (Docker Desktop quitting, a VM or host power loss) can leave a lock file
# created but never flushed to disk: the size is right and every byte is NUL. PostgreSQL cannot
# parse that and refuses to start. Two files can be in that state:
#
#   /var/lib/postgresql/18/docker/postmaster.pid    in the data volume
#   /var/run/postgresql/.s.PGSQL.5432.lock          in the container's own layer
#
# This script backs up and removes the one in the data volume when it carries no readable PID, then
# recreates the container so the layer-held socket lock file goes with the old layer, and waits for
# the health check. It never rebuilds or resets the data volume: after the repair, a database that
# shut down cleanly starts without WAL recovery. It is safe to run on a healthy stack, where it
# reports and exits.
#
# Overrides: PG_COMPOSE_SERVICE (default postgres), PG_USER and PG_DATABASE for the closing probe
# (default dlightrag), PG_RECOVER_WAIT_SECONDS (default 180).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

service="${PG_COMPOSE_SERVICE:-postgres}"
user="${PG_USER:-dlightrag}"
database="${PG_DATABASE:-dlightrag}"
wait_seconds="${PG_RECOVER_WAIT_SECONDS:-180}"

say() { printf 'postgres-recover: %s\n' "$*"; }
fail() {
    printf 'postgres-recover: %s\n' "$*" >&2
    exit 1
}

container_id() {
    docker compose ps -a -q "${service}" 2>/dev/null | head -n 1
}

state_of() { # state_of CONTAINER
    docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' "$1" 2>/dev/null || true
}

wait_healthy() { # wait_healthy CONTAINER WAIT_SECONDS
    local id="$1" seconds="$2" waited=0
    while [ "${waited}" -lt "${seconds}" ]; do
        case "$(state_of "${id}")" in
        healthy) return 0 ;;
        exited | dead | "") return 1 ;;
        esac
        sleep 2
        waited=$((waited + 2))
    done
    return 1
}

show_failure() { # show_failure CONTAINER
    local log
    log="$(docker logs --tail 20 "$1" 2>&1 || true)"
    if [ -n "${log}" ]; then
        printf '%s\n' "${log}" >&2
        return 0
    fi
    say "the container log is empty too; the same shutdown corrupts it, so read the FATAL with:"
    say "  docker start -a $(docker inspect -f '{{.Name}}' "$1" 2>/dev/null | sed 's|^/||')"
}

command -v docker >/dev/null 2>&1 || fail "docker is not on PATH"
docker info >/dev/null 2>&1 || fail "the Docker daemon is not reachable; start Docker Desktop first"

id="$(container_id)"
if [ -z "${id}" ]; then
    say "no ${service} container exists yet; starting the Compose stack"
    docker compose up -d "${service}"
    id="$(container_id)"
    [ -n "${id}" ] || fail "docker compose did not create the ${service} container"
fi

state="$(state_of "${id}")"
if [ "${state}" = "healthy" ]; then
    say "PostgreSQL is healthy; nothing to recover."
    exit 0
fi

# A container that is still starting may only be inside the health check's start period.
case "${state}" in
starting | running)
    say "the container is ${state}; giving the health check 30s to settle"
    if wait_healthy "${id}" 30; then
        say "PostgreSQL is healthy; nothing to recover."
        exit 0
    fi
    ;;
esac

say "recovering PostgreSQL (container state: ${state})"
show_failure "${id}"

volume="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/var/lib/postgresql"}}{{.Name}}{{end}}{{end}}' "${id}" 2>/dev/null || true)"
[ -n "${volume}" ] || fail "the ${service} container mounts no named volume at /var/lib/postgresql; this recovery expects the Compose stack's data volume"
image="$(docker inspect -f '{{.Config.Image}}' "${id}" 2>/dev/null || true)"
[ -n "${image}" ] || fail "cannot read the image the ${service} container runs"

# The container is crash-looping or stopped: take it out of the restart policy first, so nothing
# respawns mid-repair, then stop it so the data volume is not in use while it is inspected.
docker update --restart=no "${id}" >/dev/null 2>&1 || true
if [ "$(docker inspect -f '{{.State.Running}}' "${id}" 2>/dev/null || true)" = "true" ]; then
    docker stop --time 30 "${id}" >/dev/null 2>&1 || docker kill "${id}" >/dev/null 2>&1 || true
fi

backup_dir="$(mktemp -d "${TMPDIR:-/tmp}/dlightrag-postgres-recover.XXXXXX")"
say "lock file backup goes to ${backup_dir}"

repair_status=0
docker run -i --rm \
    -v "${volume}:/pgdata" \
    -v "${backup_dir}:/backup" \
    --entrypoint sh "${image}" -s <<'REPAIR' || repair_status=$?
set -eu
lock=/pgdata/18/docker/postmaster.pid
if [ ! -e "${lock}" ]; then
    echo "postmaster.pid: absent"
    exit 0
fi
# PostgreSQL reads the postmaster's PID from the first line. An empty file, and the
# full-size file of NUL bytes an unclean shutdown leaves behind, are both unreadable
# to it, so they go; a readable one is left for PostgreSQL to clear itself.
first="$(tr -d '\000' < "${lock}" | head -n 1)"
case "${first}" in
*[!0-9]* | "")
    cp "${lock}" /backup/postmaster.pid
    rm -f "${lock}"
    echo "postmaster.pid: removed (no readable PID; backup at /backup/postmaster.pid)"
    ;;
*)
    echo "postmaster.pid: kept (readable; PostgreSQL clears a stale one itself)"
    ;;
esac
REPAIR
[ "${repair_status}" -eq 0 ] || fail "could not inspect the data volume's lock file (docker exit ${repair_status})"

say "recreating the container so the container-layer socket lock file goes with the old layer"
docker compose up -d --no-build --force-recreate "${service}"

id="$(container_id)"
[ -n "${id}" ] || fail "docker compose did not recreate the ${service} container"

if wait_healthy "${id}" "${wait_seconds}"; then
    version="$(docker exec "${id}" psql -U "${user}" -d "${database}" -tAc 'show server_version' 2>/dev/null || true)"
    if [ -n "${version}" ]; then
        say "PostgreSQL is healthy again (server ${version}); the data volume was not rebuilt"
    else
        say "PostgreSQL passes its health check; a psql probe did not answer (custom credentials?)"
    fi
    exit 0
fi

say "the container did not become healthy within ${wait_seconds}s (state: $(state_of "${id}"))"
show_failure "${id}"
fail "the lock-file repair did not bring PostgreSQL up; the data volume is untouched beyond the lock file backup in ${backup_dir}"
