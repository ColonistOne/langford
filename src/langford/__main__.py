"""Langford v0.1 — interact loop only.

Polls Colony notifications (mentions, replies, etc.) and dispatches each
new event through a LangGraph react agent that has the full
ColonyToolkit attached. The agent decides which tool(s) to use to
respond — typically a comment on the originating post.

Engage and post loops are NOT enabled in v0.1; the env vars exist so
operators can toggle them on once the reactive behaviour is observed
to be sane (Jack's call: ~48h reactive-only before enabling autonomy).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from contextlib import suppress

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_colony import ColonyEventPoller, ColonyNotification, ColonyToolkit
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

logger = logging.getLogger("langford")

# Sentinel below any plausible karma value — disables the auto-pause
# entirely. Negative because karma can legitimately be below 0.
_KARMA_DISABLED = -10**6

SYSTEM_PROMPT = """\
You are Langford, an AI agent on The Colony (thecolony.cc) — a social
network for AI agents and humans. You are a sibling to @eliza-gemma
and serve as live dogfood for the langchain-colony Python package.

Stack: LangGraph + langchain-colony + local Ollama (qwen3.6:27b).

When a notification arrives, you MUST take one of two actions:
either invoke a Colony tool, or explicitly state "no action needed"
and stop. Never emit a free-form reply text without calling a tool —
the user only sees what your tools post on The Colony.

Required tool calls by notification_type:
  * direct_message → CALL colony_send_message(
      username=<sender_username from the notification>,
      body=<your reply to the content>
    ). Use the sender_username field, NOT the display name in the
    notification text. The DM you're replying to is in the
    "Content:" block.
  * mention or reply → CALL colony_comment_on_post(
      post_id=<post_id from the notification>,
      body=<your reply>,
      parent_comment_id=<comment_id if it's a reply, else omit>
    ).
  * vote / reaction / follow / award / tip_received → no action
    needed; emit one short line saying so and stop.

Response style: substantive, technical, plain-spoken. Show you
actually read what the other agent or human wrote — quote specific
phrases or details when you respond. No hype, no "Great question!"
opener, no marketing voice. No emoji unless the person you're
replying to used one first.

Length guidance:
  * direct_message → at least 2–5 sentences. Make them count. If the
    question deserves more (a couple of substantive paragraphs),
    give it. Don't pad — but don't truncate either.
  * mention / reply / comment_on_post → one substantive paragraph
    (3–6 sentences) by default. Match the depth of the parent
    thread; if the parent is technical, your reply should be too.
  * If you genuinely have nothing substantive to add, REACT with an
    emoji via colony_react_to_post or colony_react_to_comment
    instead of posting fillers like "Thanks!" / "Confirmed." /
    "Received." Those waste the recipient's attention.

Boundaries:
  * Never claim to be human.
  * Don't republish other users' content to third parties.
  * Don't follow or DM users you have no prior interaction with.
  * If asked who runs you, say: "Operated by ColonistOne."
"""


def _build_event_message(notif: ColonyNotification) -> HumanMessage:
    """Turn a ColonyNotification into a HumanMessage for the agent.

    Uses the enriched fields (``sender_username``, ``body``) populated
    by ``ColonyEventPoller(enrich=True)`` in langchain-colony 0.8.0+.
    Falls back to the raw ``message`` text when enrichment didn't fire.
    """
    parts = [f"Notification type: {notif.notification_type}"]
    if notif.sender_username:
        parts.append(f"Sender username: @{notif.sender_username}")
    if notif.sender_display_name:
        parts.append(f"Sender display name: {notif.sender_display_name}")
    if notif.post_id:
        parts.append(f"Post id: {notif.post_id}")
    if notif.comment_id:
        parts.append(f"Comment id: {notif.comment_id}")
    if notif.body:
        parts.append("")
        parts.append(f"Content:\n{notif.body}")
    elif notif.message:
        parts.append("")
        parts.append(f"Notification text: {notif.message}")
    parts.append("")
    parts.append(
        "Decide whether to respond, and if so, use the appropriate "
        "Colony tool. If no response is warranted (e.g. vote, "
        "reaction, follow), say so briefly and stop. For direct "
        "messages, reply via colony_send_message using the sender "
        "username above (NOT the display name)."
    )
    return HumanMessage(content="\n".join(parts))


async def _check_safety_gates(
    toolkit: ColonyToolkit,
    ollama_url: str,
    min_karma: int,
    health_check: bool,
) -> str | None:
    """Pre-tick safety gates.

    Returns ``None`` if it's safe to poll/dispatch this cycle, or a
    short reason string if the tick should be skipped. The two gates:

    * **Karma auto-pause** — if the agent's karma has dropped below
      ``min_karma``, pause until it recovers (e.g. someone moderates
      the offending content or upvotes the agent back into bounds).
    * **Ollama health** — if the local LLM is unreachable, polling
      makes no sense; we'd just dispatch into the void and lose the
      notifications to ``mark_read`` on a failed cycle.

    SDK errors during the karma check are not fatal — we proceed
    rather than wedging the agent on a transient backend hiccup.
    """
    if min_karma > _KARMA_DISABLED:
        try:
            me = await asyncio.to_thread(toolkit.client.get_me)
            karma = int(me.get("karma", 0))
            if karma < min_karma:
                return f"karma={karma} < min_karma={min_karma}"
        except Exception as exc:
            logger.warning("karma check failed (%s) — proceeding", exc)

    if health_check:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{ollama_url.rstrip('/')}/api/tags")
            if r.status_code != 200:
                return f"ollama unhealthy (HTTP {r.status_code})"
        except (httpx.HTTPError, OSError) as exc:
            return f"ollama unreachable: {exc.__class__.__name__}"

    return None


async def _interact_loop(
    poller: ColonyEventPoller,
    toolkit: ColonyToolkit,
    ollama_url: str,
    poll_interval: int,
    min_karma: int,
    health_check: bool,
    stop_event: asyncio.Event,
) -> None:
    """Custom polling loop with per-tick safety gates.

    Replaces ``poller.run_async`` so we can inspect karma / Ollama
    health BEFORE polling, and skip ticks (without consuming
    notifications) when something's wrong.
    """
    paused_reason: str | None = None
    logger.info("🔔 interact loop starting (poll every %ds)", poll_interval)

    while not stop_event.is_set():
        reason = await _check_safety_gates(toolkit, ollama_url, min_karma, health_check)
        if reason is None:
            if paused_reason is not None:
                logger.info("▶️  gates clear (was paused: %s) — resuming", paused_reason)
                paused_reason = None
            try:
                await poller.poll_once_async()
            except Exception:
                logger.exception("poll_once_async failed")
        else:
            if reason != paused_reason:
                logger.warning("⏸️  paused: %s", reason)
                paused_reason = reason

        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
            if stop_event.is_set():
                return


async def _handle_event(agent, notif: ColonyNotification) -> None:
    logger.info(
        "event type=%s sender=@%s post_id=%s comment_id=%s",
        notif.notification_type,
        notif.sender_username or "?",
        notif.post_id,
        notif.comment_id,
    )
    try:
        result = await agent.ainvoke({"messages": [_build_event_message(notif)]})
        final = result["messages"][-1]
        logger.info("agent finished: %s", str(final.content)[:240].replace("\n", " "))
    except Exception:
        logger.exception("event handler failed (type=%s)", notif.notification_type)


async def main_async() -> None:
    load_dotenv()

    log_level = os.environ.get("LANGFORD_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    api_key = os.environ.get("COLONY_API_KEY")
    if not api_key:
        logger.error("COLONY_API_KEY not set — see .env.example")
        sys.exit(2)

    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "qwen3.6:27b")
    poll_interval = int(os.environ.get("LANGFORD_POLL_INTERVAL_SEC", "120"))
    interact_enabled = os.environ.get("LANGFORD_INTERACT_ENABLED", "true").lower() == "true"
    engage_enabled = os.environ.get("LANGFORD_ENGAGE_ENABLED", "false").lower() == "true"
    post_enabled = os.environ.get("LANGFORD_POST_ENABLED", "false").lower() == "true"

    # Safety gates (v0.2). Pause the loop when karma drops below the
    # threshold or Ollama is unreachable. Setting min_karma below the
    # _KARMA_DISABLED sentinel disables the karma gate; setting the
    # health-check env to "false" disables the Ollama probe.
    min_karma_raw = os.environ.get("LANGFORD_MIN_KARMA", "-5")
    try:
        min_karma = int(min_karma_raw)
    except ValueError:
        logger.warning("LANGFORD_MIN_KARMA=%r is not an int — disabling karma gate", min_karma_raw)
        min_karma = _KARMA_DISABLED
    health_check = os.environ.get("LANGFORD_OLLAMA_HEALTH_CHECK", "true").lower() == "true"

    if engage_enabled or post_enabled:
        logger.warning(
            "LANGFORD_ENGAGE_ENABLED=%s POST_ENABLED=%s — but those loops "
            "are not implemented in v0.1; ignoring.",
            engage_enabled,
            post_enabled,
        )

    # num_predict caps output tokens. Ollama's default is 128 which
    # truncates anything substantive — early Langford DMs read like
    # one-line acknowledgements purely because of this. Bumping the
    # default lets the system prompt's length guidance actually take
    # effect. Per-tick latency rises with the cap; 1024 keeps cold
    # tool-calling rounds under ~60s on qwen3.6:27b on a 3090.
    max_output_tokens = int(os.environ.get("LANGFORD_MAX_OUTPUT_TOKENS", "1024"))
    temperature = float(os.environ.get("LANGFORD_TEMPERATURE", "0.7"))

    logger.info(
        "Connecting to Ollama (%s, model=%s, num_predict=%d, temperature=%.1f)",
        ollama_url,
        ollama_model,
        max_output_tokens,
        temperature,
    )
    llm = ChatOllama(
        model=ollama_model,
        base_url=ollama_url,
        temperature=temperature,
        num_predict=max_output_tokens,
    )

    logger.info("Loading ColonyToolkit")
    toolkit = ColonyToolkit(api_key=api_key)
    tools = toolkit.get_tools()
    tool_names = sorted(t.name for t in tools)
    logger.info("loaded %d Colony tools: %s", len(tools), ", ".join(tool_names))

    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)

    me = toolkit.client.get_me()
    logger.info(
        "✅ Connected as @%s (id=%s, karma=%s, trust=%s)",
        me.get("username", "?"),
        me.get("id", "?"),
        me.get("karma", "?"),
        me.get("trust_level", "?"),
    )

    if not interact_enabled:
        logger.warning("LANGFORD_INTERACT_ENABLED=false — exiting (no loops to run)")
        return

    poller = ColonyEventPoller(api_key=api_key, mark_read=True)

    @poller.on()
    async def on_event(notif: ColonyNotification) -> None:
        await _handle_event(agent, notif)

    stop_event = asyncio.Event()

    def _shutdown() -> None:
        logger.info("shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown)

    logger.info(
        "safety gates: min_karma=%s, ollama_health_check=%s",
        "disabled" if min_karma <= _KARMA_DISABLED else min_karma,
        health_check,
    )

    loop_task = asyncio.create_task(
        _interact_loop(
            poller=poller,
            toolkit=toolkit,
            ollama_url=ollama_url,
            poll_interval=poll_interval,
            min_karma=min_karma,
            health_check=health_check,
            stop_event=stop_event,
        )
    )

    try:
        await stop_event.wait()
    finally:
        logger.info("stopping interact loop")
        loop_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(loop_task, timeout=5)
        logger.info("goodbye")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
