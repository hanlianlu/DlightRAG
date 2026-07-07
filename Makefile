LANGFUSE_LOCAL_DIR ?= $(abspath ../langfuse-local)
LANGFUSE_PROJECT ?= langfuse
LANGFUSE_HOST ?= http://localhost:3300
PYTHON ?= python3
LANGFUSE_COMPOSE = docker compose --env-file "$(LANGFUSE_LOCAL_DIR)/.env" -p $(LANGFUSE_PROJECT) -f "$(LANGFUSE_LOCAL_DIR)/docker-compose.yml"
LANGFUSE_STACK = $(PYTHON) scripts/langfuse/stack.py --dir "$(LANGFUSE_LOCAL_DIR)"
LANGFUSE_BOOTSTRAP = $(PYTHON) scripts/langfuse/headless.py --langfuse-env "$(LANGFUSE_LOCAL_DIR)/.env" --dlightrag-env ".env" --host "$(LANGFUSE_HOST)"

.PHONY: mineru-install mineru-api mineru-title-aided mineru-service-install mineru-service-start mineru-service-stop mineru-service-status mineru-service-logs mineru-service-uninstall langfuse-stack langfuse-bootstrap langfuse-up langfuse-down langfuse-restart langfuse-status langfuse-logs langfuse-health ci frontend-ci ci-full

mineru-install:
	scripts/mineru/install.sh

mineru-api:
	scripts/mineru/api.sh

mineru-title-aided:
	scripts/mineru/title_aided.sh

mineru-service-install:
	scripts/mineru/launch_agent.sh install

mineru-service-start:
	scripts/mineru/launch_agent.sh start

mineru-service-stop:
	scripts/mineru/launch_agent.sh stop

mineru-service-status:
	scripts/mineru/launch_agent.sh status

mineru-service-logs:
	scripts/mineru/launch_agent.sh logs

mineru-service-uninstall:
	scripts/mineru/launch_agent.sh uninstall

langfuse-stack:
	$(LANGFUSE_STACK)

langfuse-bootstrap: langfuse-stack
	$(LANGFUSE_BOOTSTRAP)

langfuse-up: langfuse-bootstrap
	$(LANGFUSE_COMPOSE) up -d

langfuse-down:
	$(LANGFUSE_COMPOSE) down

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
# Fast path: what GitHub Actions runs on every PR/push (~2 min)
# One-time setup for new clones / new developers
.PHONY: hooks
hooks:
	uv run pre-commit install
	@echo "Pre-commit hooks installed — will run on every git commit."

ci:
	uv sync --group dev
	uv run ruff check src/ tests/ scripts/
	uv run ruff check src/ scripts/ --select S
	uv run ruff format --check src/ tests/ scripts/
	uv run pyright
	uv run lint-imports
	$(MAKE) frontend-ci
	uv run pytest tests/unit -v --tb=short
	@echo "CI (fast) passed."

frontend-ci:
	npm --prefix frontend ci
	npm --prefix frontend run typecheck
	npm --prefix frontend run lint:css
	npm --prefix frontend run build
	npm --prefix frontend audit --omit=dev
	@echo "Frontend CI passed."

# Full local: includes integration tests (needs PostgreSQL + pgvector)
ci-full: ci
	uv run pytest tests/integration -v --tb=short
	@echo "CI (full) passed."

# ─────────────────────────────────────────────────────────────────
# Playwright E2E UI tests (headless)
.PHONY: test-e2e

test-e2e:
	uv run pytest tests/e2e/ -v -m e2e --tb=short

# Full + E2E: needs PostgreSQL 18 with AGE; model calls are faked in tests
ci-e2e: ci-full test-e2e
	DLIGHTRAG_RUN_E2E_PG18=1 uv run pytest tests/e2e -v --tb=short -m e2e_pg18
	@echo "CI (e2e) passed."
