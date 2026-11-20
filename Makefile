PYTHON_LINT_PATHS = packages/ src/ tests/ scripts/ chart-render/ prerequisite_setup.py
PYTHON_SECURITY_PATHS = packages/ src/ scripts/ chart-render/ prerequisite_setup.py

.PHONY: mineru-install mineru-api mineru-title-aided mineru-service-install mineru-service-start mineru-service-stop mineru-service-status mineru-service-logs mineru-service-uninstall hyperdx-bootstrap hyperdx-up hyperdx-down hyperdx-status hyperdx-logs hyperdx-reset hooks sync-dev lint lint-security format-check typecheck architecture-check shellcheck-all frontend-install frontend-typecheck frontend-lint frontend-test frontend-browser-install frontend-browser-test frontend-browser-cross-engine-install frontend-browser-cross-engine-test frontend-build frontend-audit frontend-ci chart-render-install chart-render-link fonts chart-render-test release-check platform-check release export-traces workspace-wheels test-unit runtime-faults runtime-pg18 load-runtime validate-runtime ci ci-full test-e2e ci-e2e dev-reset postgres-recover eval-serve

mineru-install:
	uv run dlightrag-mineru install

mineru-api:
	uv run dlightrag-mineru api

mineru-title-aided:
	uv run dlightrag-mineru-title-aided

# Background MinerU service — one launcher, dispatched per OS by the package
# (macOS: launchd | Linux/WSL2: systemd --user). See packages/mineru/README.md.
mineru-service-install:
	uv run dlightrag-mineru service install

mineru-service-start:
	uv run dlightrag-mineru service start

mineru-service-stop:
	uv run dlightrag-mineru service stop

mineru-service-status:
	uv run dlightrag-mineru service status

mineru-service-logs:
	uv run dlightrag-mineru service logs

mineru-service-uninstall:
	uv run dlightrag-mineru service uninstall

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
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push
	@echo "Hooks installed — pre-commit runs on every git commit, pre-push (frontend typecheck, node and browser tests) on every git push that touches frontend/."

sync-dev:
	uv lock --check
	uv sync --locked --group dev

lint:
	uv run ruff check $(PYTHON_LINT_PATHS)

lint-security:
	uv run ruff check $(PYTHON_SECURITY_PATHS) --select S

format-check:
	uv run ruff format --check $(PYTHON_LINT_PATHS)

# Pyright's working set is about 2.3 GB on this tree, and the node it runs on takes a heap of about 2 GB
# on a 7 GB CI runner: `make typecheck` got through there in three minutes of garbage collection and
# the core's copy of the tree ran out of heap. Given room, the same run takes 35 seconds on a laptop,
# against 150 under a 2 GB heap.
PYRIGHT_NODE_OPTIONS := $(NODE_OPTIONS) --max-old-space-size=4096

typecheck:
	NODE_OPTIONS="$(PYRIGHT_NODE_OPTIONS)" uv run pyright

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

# The design-system suite on Firefox and WebKit as well: the engines a user on Safari or Firefox runs.
frontend-browser-cross-engine-install: frontend-install
	cd frontend && npx playwright install firefox webkit

frontend-browser-cross-engine-test: frontend-browser-cross-engine-install
	npm --prefix frontend run test:browser:cross-engine

frontend-build:
	npm --prefix frontend run build

frontend-audit:
	npm --prefix frontend audit --omit=dev

frontend-ci: frontend-install frontend-typecheck frontend-lint frontend-test frontend-browser-test frontend-browser-cross-engine-test frontend-build frontend-audit
	@echo "Frontend CI passed."

# Release time: the version contract across the four manifests that name it.
# `make release VERSION=x.y.z` writes those manifests and runs this; the configuration
# baseline names the config schema rather than the version, so `platform-check` holds it.
release-check:
	uv run python scripts/verify_release_contract.py --scope versions

# Every push: the pins and the configuration every deployment holds, whatever the version.
platform-check:
	uv run python scripts/verify_release_contract.py --scope platform
	uv run python scripts/adr_index.py --check

# One release: write the version, re-resolve the lock, then check the contract.
# `make release VERSION=2.2.0` edits; ARGS passes --commit, --tag or --push.
release:
	uv run python scripts/release.py "$(VERSION)" $(ARGS)

workspace-wheels: frontend-build
	rm -rf dist
	uv build --all-packages --no-sources --out-dir dist
	uv run python scripts/verify_workspace_wheels.py --dist dist --smoke-installed

# The chart renderer's one dependency, ECharts: the report toolkit's tests build with it.
chart-render-install:
	npm --prefix chart-render ci --no-audit --no-fund

# The font set the image ships, at the Dockerfile's pins, for a machine that is not the image. The
# chart tests skip without it and an Agent asked for a Chinese, Japanese or Korean PDF has nothing
# to embed, so this prints the export a native Run needs to reach it: fpdf2 reads no environment.
fonts:
	scripts/install-fonts.sh "$(CURDIR)/.chart-tools/fonts/dlightrag"

# Put the two chart commands on the project environment's PATH, the way the image puts
# them on bin/ over lib/echarts-render, so a native Run reaches them from its bash.
chart-render-link: chart-render-install
	@scripts_dir="$$(uv run python -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"; \
	ln -sf "$(CURDIR)/chart-render/echarts_render.py" "$$scripts_dir/echarts-render"; \
	ln -sf "$(CURDIR)/chart-render/html_report.py" "$$scripts_dir/html-report"; \
	echo "linked echarts-render and html-report into $$scripts_dir"

chart-render-test: chart-render-install
	node --test "chart-render/test/*.test.cjs"

test-unit: chart-render-install
	uv run pytest tests/unit -q --tb=short


# RunRuntime fault-injection matrix. PostgreSQL must be reachable;
# providers and parsers are controlled fakes. The page's reconnect and its Corpus Mutation
# repair/resume presentation are held by the frontend tests, which `frontend-ci` runs.
runtime-faults:
	uv run python -c 'import asyncio; from tests.support.pg import require_postgres; asyncio.run(require_postgres())'
	uv run pytest tests/unit tests/integration/test_corpus_mutation_runs_pg.py tests/integration/test_corpus_mutation_phase_faults_pg.py tests/integration/test_run_runtime_lane_independence_pg.py tests/integration/test_run_runtime_query_plans_pg.py tests/integration/test_retrieval_runs_pg.py tests/integration/test_answer_runs_pg.py -q --tb=short

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

# Training data of the owners who contribute (Settings > Privacy), read-only against the
# deployment's PostgreSQL. Nothing is exported for an owner who has not turned it on. ARGS goes to
# scripts/export_traces.py:
#   make export-traces                              one JSONL line per Run on stdout, ids as stored
#   make export-traces ARGS="--dataset $HOME/datasets/v1"
#                                                   a de-identified, split dataset of whole Sessions
#                                                   in a new directory (ADR 0044), under the key
#                                                   DLIGHTRAG_EXPORT_KEY in .env: openssl rand -base64 32
# Both hold conversations, so write them outside the repository.
export-traces:
	uv run scripts/export_traces.py $(ARGS)

dev-reset:
	uv run scripts/reset_development.py --mode docker $(ARGS)

# Recover the Compose PostgreSQL from the lock files an unclean Docker shutdown leaves behind: a
# container that restarts forever with `FATAL: bogus data in lock file`. The repair backs up and
# removes the data volume's unreadable postmaster.pid, gives the container a fresh layer so the
# socket lock file goes with the old one, and waits for the health check; the data volume is never
# rebuilt. Safe on a healthy stack, where it reports and does nothing. See
# docs/postgresql.md#startup-recovery.
postgres-recover:
	scripts/postgres-recover.sh

# An app for the Skill routing eval (docs/skills-routing-eval.md), on a database of its own.
# The deployment signs callers in with Cloudflare JWT, which a driver script cannot present, so this
# app runs with auth_mode none; and it must never touch the owner's data, so it gets its own database,
# created when absent and never dropped: an eval's Runs are the evidence and outlive the server.
# EVAL_ENV is where the provider keys come from; a worktree has no .env, so point it at the main
# checkout's. The schema is the app's own: it applies its migrations on start. Of the extensions it
# needs, it creates pg_trgm, pg_textsearch and pg_jieba itself, but never pgvector, which
# postgres/init.sql creates in the initial database alone -- so the new database gets it here.
# The compose service that holds the database; overridable, so this is not tied to one machine's name.
PG_SERVICE ?= dlightrag-postgres-1
EVAL_PORT ?= 8110
EVAL_DB ?= dlightrag_eval
EVAL_ENV ?= .env

# The eval Agent runs the same Skill the image ships: it needs the chart commands on the
# environment's PATH, the way `chart-render-link` puts them there for a native Run.
eval-serve: chart-render-link
	docker exec $(PG_SERVICE) psql -U dlightrag -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='$(EVAL_DB)'" | grep -q 1 || docker exec $(PG_SERVICE) psql -U dlightrag -d postgres -q -c "CREATE DATABASE $(EVAL_DB)"
	docker exec $(PG_SERVICE) psql -U dlightrag -d $(EVAL_DB) -q -c "CREATE EXTENSION IF NOT EXISTS vector"
	rm -rf /tmp/dlightrag-eval && mkdir -p /tmp/dlightrag-eval
	cp config.yaml /tmp/dlightrag-eval/config.yaml
	grep -v '^DLIGHTRAG_ACCESS__' $(EVAL_ENV) > /tmp/dlightrag-eval/.env || true
	printf 'DLIGHTRAG_ACCESS__AUTH_MODE=none\nDLIGHTRAG_STORAGE__POSTGRES__DATABASE=$(EVAL_DB)\n' >> /tmp/dlightrag-eval/.env
	cd /tmp/dlightrag-eval && PATH="$(CURDIR)/.venv/bin:$$PATH" DLIGHTRAG_INTERFACES__API__PORT=$(EVAL_PORT) DLIGHTRAG_INTERFACES__API__HOST=127.0.0.1 uv run --project $(CURDIR) dlightrag-api

# Pull-request gate: static analysis, frontend checks, isolated wheels, and unit tests.
ci: sync-dev lint lint-security format-check typecheck architecture-check shellcheck-all platform-check frontend-ci chart-render-test workspace-wheels test-unit
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
