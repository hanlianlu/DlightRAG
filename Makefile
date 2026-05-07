LANGFUSE_LOCAL_DIR ?= $(abspath ../langfuse-local)
LANGFUSE_PROJECT ?= langfuse
LANGFUSE_HOST ?= http://localhost:3300
LANGFUSE_COMPOSE = docker compose --env-file "$(LANGFUSE_LOCAL_DIR)/.env" -p $(LANGFUSE_PROJECT) -f "$(LANGFUSE_LOCAL_DIR)/docker-compose.yml"

.PHONY: langfuse-up langfuse-down langfuse-restart langfuse-status langfuse-logs langfuse-health

langfuse-up:
	$(LANGFUSE_COMPOSE) up -d

langfuse-down:
	$(LANGFUSE_COMPOSE) down

langfuse-restart:
	$(LANGFUSE_COMPOSE) up -d --force-recreate langfuse-web langfuse-worker

langfuse-status:
	$(LANGFUSE_COMPOSE) ps

langfuse-logs:
	$(LANGFUSE_COMPOSE) logs -f langfuse-web langfuse-worker

langfuse-health:
	curl -fsS $(LANGFUSE_HOST)/api/public/health && printf '\n'
