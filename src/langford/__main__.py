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

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_colony import ColonyEventPoller, ColonyNotification, ColonyToolkit
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

logger = logging.getLogger("langford")

SYSTEM_PROMPT = """\
You are Langford, an AI agent on The Colony (thecolony.cc) — a social
network for AI agents and humans collaborating on findings, tools and
research. You are a sibling to @eliza-gemma and serve as live dogfood
for the langchain-colony Python package.

Your stack is LangGraph + langchain-colony + local Ollama
(qwen3.6:27b). When a notification reaches you, decide which tool(s)
to use to respond appropriately.

Common cases:
  * mention in a post / comment → leave a thoughtful comment on the
    post (use ColonyCommentOnPost with the post_id from the
    notification).
  * reply to one of your comments → continue the thread with
    ColonyCommentOnPost on the same post_id.
  * vote / reaction / follow notifications → no action needed; ignore.

Style: terse, technical, plain-spoken. No emoji unless the person
you're replying to used one first. No hype, no "Great question!"
opener, no marketing voice. Quote specific details when relevant. If
you have nothing substantive to add, react with an emoji via
ColonyReactToPost or ColonyReactToComment instead of posting empty
filler.

Boundaries:
  * Never claim to be human.
  * Don't republish other users' content to third parties.
  * Don't follow or DM users you have no prior interaction with.
  * If asked about your operator, say: "Operated by ColonistOne."
"""


def _build_event_message(notif: ColonyNotification) -> HumanMessage:
    """Turn a ColonyNotification into a HumanMessage for the agent."""
    parts = [f"Notification type: {notif.notification_type}"]
    if notif.post_id:
        parts.append(f"Post id: {notif.post_id}")
    if notif.comment_id:
        parts.append(f"Comment id: {notif.comment_id}")
    if notif.message:
        parts.append("")
        parts.append(notif.message)
    parts.append("")
    parts.append(
        "Decide whether to respond, and if so, use the appropriate "
        "Colony tool. If no response is warranted (e.g. vote, "
        "reaction, follow), say so briefly and stop."
    )
    return HumanMessage(content="\n".join(parts))


async def _handle_event(agent, notif: ColonyNotification) -> None:
    logger.info(
        "event type=%s post_id=%s comment_id=%s",
        notif.notification_type,
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

    if engage_enabled or post_enabled:
        logger.warning(
            "LANGFORD_ENGAGE_ENABLED=%s POST_ENABLED=%s — but those loops "
            "are not implemented in v0.1; ignoring.",
            engage_enabled,
            post_enabled,
        )

    logger.info("Connecting to Ollama (%s, model=%s)", ollama_url, ollama_model)
    llm = ChatOllama(
        model=ollama_model,
        base_url=ollama_url,
        temperature=0.7,
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

    logger.info("🔔 interact loop starting (poll every %ds)", poll_interval)
    poller_task = asyncio.create_task(poller.run_async(poll_interval=poll_interval))

    try:
        await stop_event.wait()
    finally:
        logger.info("stopping poller")
        poller.stop()
        poller_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(poller_task, timeout=5)
        logger.info("goodbye")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
