PYTHON_LINT_PATHS = packages/ src/ tests/ scripts/ chart-render/ prerequisite_setup.py
PYTHON_SECURITY_PATHS = packages/ src/ scripts/ chart-render/ prerequisite_setup.py

.PHONY: mineru-install mineru-api mineru-gradio mineru-title-aided mineru-service-install mineru-service-start mineru-service-stop mineru-service-status mineru-service-logs mineru-service-uninstall hyperdx-bootstrap hyperdx-up hyperdx-down hyperdx-status hyperdx-logs hyperdx-reset hooks sync-dev lint lint-security format-check typecheck architecture-check shellcheck-all frontend-install frontend-typecheck frontend-lint frontend-test frontend-browser-install frontend-browser-test frontend-build frontend-audit frontend-ci chart-render-install chart-render-test release-check workspace-wheels test-unit runtime-faults runtime-pg18 load-runtime validate-runtime ci ci-full test-e2e ci-e2e dev-reset

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

# ─────────────────────────────────────────────────────────────────
# ClickStack / HyperDX observability stack (packages/observability)
# ─────────────────────────────────────────────────────────────────
# `hyperdx-bootstrap` writes stack/.env once (generated secrets), sends a probe span through the
# collector, applies retention to every telemetry table, and reports whether the span landed — the
# one check that says a freshly started stack actually ingests. `hyperdx-up` starts the four
# containers.
hyperdx-bootstrap:
	uv run dlightrag-observability bootstrap

hyperdx-up:
	uv run dlightrag-observability up
	uv run dlightrag-observability bootstrap
	@echo "HyperDX UI: http://127.0.0.1:8300"

hyperdx-down:
	uv run dlightrag-observability down

hyperdx-status:
	uv run dlightrag-observability status

hyperdx-logs:
	uv run dlightrag-observability logs

hyperdx-reset:
	uv run dlightrag-observability reset

# ─────────────────────────────────────────────────────────────────
# CI targets — local dev matrix
# ─────────────────────────────────────────────────────────────────
# One-time setup for new clones / new developers
hooks:
	uv run pre-commit install
	@echo "Pre-commit hooks installed — will run on every git commit."

sync-dev:
	uv lock --check
	uv sync --locked --group dev

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
	npm --prefix frontend run lint

frontend-test:
	npm --prefix frontend test

frontend-browser-install: frontend-install
	cd frontend && npx playwright install chromium

frontend-browser-test: frontend-browser-install
	npm --prefix frontend run test:browser

frontend-build:
	npm --prefix frontend run build

frontend-audit:
	npm --prefix frontend audit --omit=dev

frontend-ci: frontend-install frontend-typecheck frontend-lint frontend-test frontend-browser-test frontend-build frontend-audit
	@echo "Frontend CI passed."

release-check:
	uv run python scripts/verify_release_contract.py

workspace-wheels: frontend-build
	rm -rf dist
	uv build --all-packages --no-sources --out-dir dist
	uv run python scripts/verify_workspace_wheels.py --dist dist --smoke-installed

# The chart renderer's one dependency, ECharts: the report toolkit's tests build with it.
chart-render-install:
	npm --prefix chart-render ci --no-audit --no-fund

chart-render-test: chart-render-install
	node --test "chart-render/test/*.test.cjs"

test-unit: chart-render-install
	uv run pytest tests/unit -q --tb=short

# RunRuntime fault-injection matrix. PostgreSQL must be reachable;
# providers and parsers are controlled fakes. The browser command exercises
# reconnect and explicit Corpus Mutation repair/resume presentation.
runtime-faults: frontend-browser-install
	uv run python -c 'import asyncio; from tests.support.pg import require_postgres; asyncio.run(require_postgres())'
	uv run pytest tests/unit tests/integration/test_corpus_mutation_runs_pg.py tests/integration/test_corpus_mutation_phase_faults_pg.py tests/integration/test_run_runtime_lane_independence_pg.py tests/integration/test_run_runtime_query_plans_pg.py tests/integration/test_retrieval_runs_pg.py tests/integration/test_answer_runs_pg.py -q --tb=short
	npm --prefix frontend test
	npm --prefix frontend run test:browser

# Public default-storage convergence with fake models and the supported PG18 image.
runtime-pg18:
	RUN_E2E_PG18=1 uv run pytest tests/e2e/test_pg18_lightrag_smoke.py -m e2e_pg18 -q --tb=short

# Expensive deterministic control-plane campaign; intentionally excluded from CI.
load-runtime:
	@set +e; \
	PYTHONHASHSEED=0 RUN_LOAD_RUNTIME=1 uv run pytest tests/load -m load_runtime -q -s --tb=short; \
	status=$$?; \
	if [ $$status -ne 0 ]; then echo "RUN_RUNTIME_LOAD FAIL pytest_exit=$$status"; fi; \
	exit $$status

validate-runtime: runtime-faults runtime-pg18 load-runtime
	@echo "RunRuntime failure, PG18, and load validation passed; see .test-results/load-runtime/."

dev-reset:
	uv run scripts/reset_development.py --mode docker $(ARGS)

# Pull-request gate: static analysis, frontend checks, isolated wheels, and unit tests.
ci: sync-dev lint lint-security format-check typecheck architecture-check shellcheck-all release-check frontend-ci chart-render-test workspace-wheels test-unit
	@echo "CI (fast) passed."

# Full local: includes integration tests (needs PostgreSQL + pgvector)
ci-full: ci
	uv run pytest tests/integration -v --tb=short
	@echo "CI (full) passed."

# ─────────────────────────────────────────────────────────────────
# Playwright E2E UI tests (headless)
# The bundle is gitignored, so these would otherwise run against a missing UI.
test-e2e: frontend-build chart-render-install
	uv run pytest tests/e2e/ -v -m e2e --tb=short

# Full + E2E: needs PostgreSQL 18; model calls are faked in tests
ci-e2e: ci-full test-e2e
	RUN_E2E_PG18=1 uv run pytest tests/e2e -v --tb=short -m e2e_pg18
	@echo "CI (e2e) passed."
