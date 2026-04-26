UV := /home/user/.local/bin/uv
PIDFILE := .agent.pid

# Force bash (not /bin/sh) — `disown` and `[[ ... ]]` rely on bash semantics.
SHELL := /bin/bash

# Cross-agent flock — prevents two GPU/Ollama-using Colony agents
# (langford, eliza-gemma, future siblings) from running concurrently
# on the same host. Fail-fast: second invocation prints the holder
# and exits 1.
LOCK := /home/user/.local/bin/colony-agent-lock
AGENT_NAME := langford

.PHONY: start start-detached stop restart status logs help

help:
	@echo "Langford operations"
	@echo ""
	@echo "  make start          Launch in the current terminal's cgroup"
	@echo "                      (inherits caller's slice; on a Claude Code"
	@echo "                       terminal that's claude.slice — prefer"
	@echo "                       start-detached in that case)"
	@echo "  make start-detached Launch under user.slice via systemd-run"
	@echo "                      (use from Claude Code shells)"
	@echo "  make stop           SIGTERM, escalate to SIGKILL after 2s"
	@echo "  make restart        stop + start"
	@echo "  make status         pid + cmdline, or 'not running'"
	@echo "  make logs           tail -f agent.log"

start:
	@if [ -f $(PIDFILE) ] && kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
		echo "already running (pid $$(cat $(PIDFILE)))"; \
		exit 1; \
	fi
	@rm -f $(PIDFILE)
	@nohup $(LOCK) $(AGENT_NAME) $(UV) run python -m langford > agent.log 2>&1 & echo $$! > $(PIDFILE); disown
	@sleep 8
	@if kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
		echo "started (pid $$(cat $(PIDFILE)))"; \
		tail -5 agent.log; \
	else \
		echo "failed to start — see agent.log"; \
		tail -20 agent.log; \
		rm -f $(PIDFILE); \
		exit 1; \
	fi

# Launch under a transient systemd scope in user.slice. Use this when
# the operator's terminal is inside claude.slice or any other capped
# slice — without this, langford inherits the caller's cap.
start-detached:
	@if [ -f $(PIDFILE) ] && kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
		echo "already running (pid $$(cat $(PIDFILE)))"; \
		exit 1; \
	fi
	@if ! command -v systemd-run >/dev/null 2>&1; then \
		echo "systemd-run not available — falling back to plain make start"; \
		$(MAKE) start; \
		exit 0; \
	fi
	@rm -f $(PIDFILE)
	@systemd-run --user --slice=user.slice --unit=langford-$$$$ \
		--quiet --collect \
		--property=WorkingDirectory=$(CURDIR) \
		--property=StandardOutput=append:$(CURDIR)/agent.log \
		--property=StandardError=append:$(CURDIR)/agent.log \
		$(LOCK) $(AGENT_NAME) $(UV) run python -m langford
	@sleep 8
	@systemctl --user status --no-pager --lines=0 "langford-*" 2>/dev/null | head -3 || true
	@PID=$$(systemctl --user show -p MainPID --value "langford-*.service" 2>/dev/null | head -1); \
		if [ -n "$$PID" ] && [ "$$PID" != "0" ]; then \
			echo "$$PID" > $(PIDFILE); \
			echo "started under user.slice (pid $$PID)"; \
			tail -5 agent.log; \
		else \
			echo "failed to start — see agent.log"; \
			tail -20 agent.log; \
			exit 1; \
		fi

stop:
	@if [ ! -f $(PIDFILE) ] || ! kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
		echo "not running"; \
		rm -f $(PIDFILE); \
		exit 0; \
	fi
	@PID=$$(cat $(PIDFILE)); \
		pkill -TERM -P $$PID 2>/dev/null || true; \
		kill -TERM $$PID 2>/dev/null || true; \
		sleep 2; \
		if kill -0 $$PID 2>/dev/null; then \
			echo "still alive after SIGTERM — sending SIGKILL"; \
			pkill -9 -P $$PID 2>/dev/null || true; \
			kill -9 $$PID 2>/dev/null || true; \
			sleep 1; \
		fi; \
		rm -f $(PIDFILE); \
		echo "stopped"

restart: stop start

status:
	@if [ -f $(PIDFILE) ] && kill -0 $$(cat $(PIDFILE)) 2>/dev/null; then \
		echo "running (pid $$(cat $(PIDFILE)))"; \
		ps -o pid,cmd -p $$(cat $(PIDFILE)); \
	else \
		echo "not running"; \
	fi

logs:
	@tail -f agent.log
