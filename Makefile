LANGFUSE_LOCAL_DIR ?= $(abspath ../langfuse-local)
LANGFUSE_PROJECT ?= langfuse
# Host-facing URL: your browser and `make langfuse-health` reach the stack here.
# Where the DlightRAG app SENDS traces is non-secret config (langfuse_host in
# config.yaml: host.docker.internal:3300 for the Dockerized app, localhost:3300
# for a native run), so bootstrap does not manage it — it only syncs the keys.
LANGFUSE_HOST ?= http://localhost:3300
PYTHON ?= python3
LANGFUSE_COMPOSE = docker compose --env-file "$(LANGFUSE_LOCAL_DIR)/.env" -p $(LANGFUSE_PROJECT) -f "$(LANGFUSE_LOCAL_DIR)/docker-compose.yml"
LANGFUSE_STACK = $(PYTHON) scripts/langfuse/stack.py --dir "$(LANGFUSE_LOCAL_DIR)"
LANGFUSE_BOOTSTRAP = $(PYTHON) scripts/langfuse/headless.py --langfuse-env "$(LANGFUSE_LOCAL_DIR)/.env" --dlightrag-env ".env"
PYTHON_LINT_PATHS = src/ tests/ scripts/ prerequisite_setup.py
PYTHON_SECURITY_PATHS = src/ scripts/ prerequisite_setup.py

.PHONY: mineru-install mineru-api mineru-gradio mineru-title-aided mineru-service-install mineru-service-start mineru-service-stop mineru-service-status mineru-service-logs mineru-service-uninstall langfuse-stack langfuse-bootstrap langfuse-up langfuse-down langfuse-reset langfuse-restart langfuse-status langfuse-logs langfuse-health hooks sync-dev lint lint-security format-check typecheck architecture-check shellcheck-all frontend-install frontend-typecheck frontend-lint frontend-build frontend-audit frontend-ci test-unit ci ci-full test-e2e ci-e2e

mineru-install:
	scripts/mineru/install.sh

mineru-api:
	scripts/mineru/api.sh

mineru-gradio:
	scripts/mineru/gradio.sh

mineru-title-aided:
	scripts/mineru/title_aided.sh

# Background MinerU service — dispatched per OS by service.sh
# (macOS: launchd | Linux/WSL2: systemd --user). One command manages BOTH the
# API backend and the Gradio WebUI (the WebUI reuses the API; opt out with
# MINERU_GRADIO_ENABLE=false in .env.mineru).
mineru-service-install:
	scripts/mineru/service.sh install

mineru-service-start:
	scripts/mineru/service.sh start

mineru-service-stop:
	scripts/mineru/service.sh stop

mineru-service-status:
	scripts/mineru/service.sh status

mineru-service-logs:
	scripts/mineru/service.sh logs

mineru-service-uninstall:
	scripts/mineru/service.sh uninstall

langfuse-stack:
	$(LANGFUSE_STACK)

langfuse-bootstrap: langfuse-stack
	$(LANGFUSE_BOOTSTRAP)

langfuse-up: langfuse-bootstrap
	$(LANGFUSE_COMPOSE) up -d

langfuse-down:
	$(LANGFUSE_COMPOSE) down

# Destructive: removes the local Langfuse data volumes (all traces). DlightRAG's
# own data is a separate Compose project and is NOT touched. Guarded by CONFIRM=1.
# Recover the login password first if possible (it is stored in
# $(LANGFUSE_LOCAL_DIR)/.env as LANGFUSE_INIT_USER_PASSWORD). After a reset run
# `make langfuse-up` to re-initialize from a fresh headless seed.
langfuse-reset:
	@if [ "$(CONFIRM)" != "1" ]; then \
		echo "langfuse-reset DELETES all local Langfuse data (traces)."; \
		echo "DlightRAG's own RAG data is a separate project and is NOT affected."; \
		echo "Re-run to confirm:  make langfuse-reset CONFIRM=1"; \
		exit 1; \
	fi
	$(LANGFUSE_COMPOSE) down -v

langfuse-restart: langfuse-bootstrap
	$(LANGFUSE_COMPOSE) up -d --force-recreate langfuse-web langfuse-worker

langfuse-status:
	$(LANGFUSE_COMPOSE) ps

langfuse-logs:
	$(LANGFUSE_COMPOSE) logs -f langfuse-web langfuse-worker

langfuse-health:
	curl -fsS $(LANGFUSE_HOST)/api/public/health && printf '\n'

# ─────────────────────────────────────────────────────────────────
# CI targets — local dev matrix
# ─────────────────────────────────────────────────────────────────
# One-time setup for new clones / new developers
hooks:
	uv run pre-commit install
	@echo "Pre-commit hooks installed — will run on every git commit."

sync-dev:
	uv sync --group dev

lint:
	uv run ruff check $(PYTHON_LINT_PATHS)

lint-security:
	uv run ruff check $(PYTHON_SECURITY_PATHS) --select S

format-check:
	uv run ruff format --check $(PYTHON_LINT_PATHS)

typecheck:
	uv run pyright

architecture-check:
	uv run lint-imports

shellcheck-all:
	uv run shellcheck $$(git ls-files '*.sh')

frontend-install:
	npm --prefix frontend ci

frontend-typecheck:
	npm --prefix frontend run typecheck

frontend-lint:
	npm --prefix frontend run lint:css

frontend-build:
	npm --prefix frontend run build

frontend-audit:
	npm --prefix frontend audit --omit=dev

frontend-ci: frontend-install frontend-typecheck frontend-lint frontend-build frontend-audit
	@echo "Frontend CI passed."

test-unit:
	uv run pytest tests/unit -q --tb=short

# Fast path: what GitHub Actions runs on every PR/push (~2 min)
ci: sync-dev lint lint-security format-check typecheck architecture-check shellcheck-all frontend-ci test-unit
	@echo "CI (fast) passed."

# Full local: includes integration tests (needs PostgreSQL + pgvector)
ci-full: ci
	uv run pytest tests/integration -v --tb=short
	@echo "CI (full) passed."

# ─────────────────────────────────────────────────────────────────
# Playwright E2E UI tests (headless)
# The bundle is gitignored, so these would otherwise run against a missing UI.
test-e2e: frontend-build
	uv run pytest tests/e2e/ -v -m e2e --tb=short

# Full + E2E: needs PostgreSQL 18 with AGE; model calls are faked in tests
ci-e2e: ci-full test-e2e
	DLIGHTRAG_RUN_E2E_PG18=1 uv run pytest tests/e2e -v --tb=short -m e2e_pg18
	@echo "CI (e2e) passed."
