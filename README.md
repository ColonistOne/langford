# Langford

LangChain/LangGraph dogfood agent for [`langchain-colony`](https://pypi.org/project/langchain-colony/).
Sibling to [`eliza-gemma`](https://github.com/ColonistOne/eliza-gemma) — the same dogfood pattern, different stack.

- **Identity**: `@langford` on [thecolony.cc](https://thecolony.cc)
- **LLM**: `qwen3.6:27b` via local Ollama
- **Stack**: LangGraph `create_react_agent` + `langchain-colony` tool surface
- **Why**: surface volume-driven bugs in `langchain-colony` the same way `eliza-gemma` surfaces them in `@thecolony/elizaos-plugin`

## Quickstart

```bash
cp .env.example .env
# edit .env — set COLONY_API_KEY at minimum

uv sync
make start-detached
make logs
```

## Operations

```
make start            Run in current shell's cgroup
make start-detached   Run under user.slice via systemd-run (use from Claude shells)
make stop             Graceful shutdown (SIGTERM → SIGKILL after 2s)
make restart          stop + start
make status           pid + cmdline
make logs             tail -f agent.log
```

Launch goes through `colony-agent-lock` (`~/.local/bin/`) so only one
GPU/Ollama-using Colony agent (`langford`, `eliza-gemma`, …) runs on
this host at a time. If another agent already holds the lock, Langford
fails fast and prints the holder.

## Roadmap

- **v0.1** (current): interact loop only — polls notifications + DMs
  and replies via the LangGraph react agent. Engage and post loops are
  coded but gated off via env (`LANGFORD_ENGAGE_ENABLED=false`,
  `LANGFORD_POST_ENABLED=false`).
- **v0.2**: enable engagement loop after ~48h of reactive observation.
- **v0.3**: enable autonomous post loop, very conservative cadence
  (eliza-gemma's first-week monoculture audit informs the defaults).
- **v0.4+**: port eliza-gemma's safety gates (karma auto-pause, quiet
  hours, LLM-health pause, diversity watchdog).

## Cross-agent coordination

This host runs at most one Ollama-using agent at a time. The flock at
`~/.cache/colony-agent.lock` is the kernel-level mutex; the Makefile is
wired to acquire it on start. See
[`feedback_one_agent_at_a_time.md`](https://github.com/ColonistOne/eliza-gemma) (memory pointer in the operator's notes).
