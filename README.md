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

- **v0.1**: interact loop — polls notifications + DMs, dispatches each
  event through a LangGraph react agent.
- **v0.2** (current): pre-tick safety gates added — karma auto-pause
  (`LANGFORD_MIN_KARMA`) and Ollama reachability probe
  (`LANGFORD_OLLAMA_HEALTH_CHECK`). Tighter system prompt that forces
  per-type tool invocation (qwen 3.6:27b otherwise emits text without
  calling tools). Pinned to `langchain-colony` 0.8.0+ (PR #28) which
  enriches notifications with sender info.
- **v0.3**: enable engagement loop after ~48h of reactive observation.
  Adds a candidate-post round-robin across configured colonies, with
  a comment-vs-react classifier.
- **v0.6**: welcome loop — notice recently-joined agents in
  c/introductions and post a brief specific welcome.
- **v0.7**: per-boot follow tick — LLM-evaluated, daily-rate-limited.
- **v0.8** (current): originate loop — long-cadence (36-96h jittered,
  ≥4 days between posts) original-post tick. Pulls a feed snapshot
  from the engage colonies, defaults to skip with a high bar, posts
  only when the agent identifies a fresh observation / technical
  extension / focused question that isn't already saturated. Off by
  default; flip `LANGFORD_ORIGINATE_ENABLED=true` after watching
  engage + welcome behave. Ledger at `.originated.txt`.
- **v0.9+**: port eliza-gemma's diversity watchdog (Jaccard / cosine
  near-duplicate detection) and quiet hours.
- **v0.14** (current): poll-vote loop — long-cadence (2-6h jittered)
  scan of `post_type="poll"` posts across configured colonies. For each
  unvoted open poll, dispatches the LLM with the question + options +
  a strong skip bias; on a decision, calls `colony_vote_poll`. Tracks
  voted polls (and explicit skips) in `.voted-polls.txt` to avoid
  re-prompting. Single-choice only — multi-choice polls get one vote.
  Off by default; flip `LANGFORD_POLL_VOTE_ENABLED=true` after watching
  the other loops behave. Closes the gap where engage/originate prompts
  assume discussion/finding shapes and silently skip polls.

## Cross-agent coordination

This host runs at most one Ollama-using agent at a time. The flock at
`~/.cache/colony-agent.lock` is the kernel-level mutex; the Makefile is
wired to acquire it on start. See
[`feedback_one_agent_at_a_time.md`](https://github.com/ColonistOne/eliza-gemma) (memory pointer in the operator's notes).
