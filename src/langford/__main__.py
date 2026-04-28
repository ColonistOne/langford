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
import random
import signal
import sys
from contextlib import suppress
from typing import Any

import httpx
from dotenv import load_dotenv
from pathlib import Path
from langchain.agents import create_agent
from langchain_colony import ColonyEventPoller, ColonyNotification, ColonyToolkit
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

logger = logging.getLogger("langford")

# Sentinel below any plausible karma value — disables the auto-pause
# entirely. Negative because karma can legitimately be below 0.
_KARMA_DISABLED = -10**6


def _is_transient_ollama_error(exc: BaseException) -> bool:
    """Match Ollama errors that warrant a retry rather than giving up.

    The dominant case under the supervisor pattern: Langford boots
    moments after Eliza is killed, but Ollama still has the prior
    agent's model in VRAM (default keep_alive=5 min). The first chat
    request triggers a CUDA OOM as Ollama tries to load qwen on top of
    gemma. Waiting ~30-60s lets keep-alive expire (or the supervisor's
    explicit unload to land), after which the retry succeeds.
    """
    msg = str(exc).lower()
    if "model failed to load" in msg:
        return True
    if "out of memory" in msg and ("cuda" in msg or "gpu" in msg):
        return True
    return False


async def _invoke_agent_with_retry(
    agent: Any,
    payload: dict,
    *,
    max_attempts: int = 3,
    backoff_base: float = 30.0,
) -> Any:
    """Call ``agent.ainvoke`` with bounded retries on transient Ollama errors.

    Non-transient exceptions propagate immediately. Each retry sleep is
    a plain ``asyncio.sleep`` so it cancels cleanly on task cancellation
    during shutdown.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await agent.ainvoke(payload)
        except Exception as exc:
            last_exc = exc
            if not _is_transient_ollama_error(exc) or attempt >= max_attempts:
                raise
            delay = backoff_base * (2 ** (attempt - 1))
            logger.warning(
                "agent invoke: transient ollama error (%s) — retry %d/%d in %ds: %s",
                exc.__class__.__name__,
                attempt,
                max_attempts,
                int(delay),
                str(exc)[:160],
            )
            await asyncio.sleep(delay)
    assert last_exc is not None  # unreachable
    raise last_exc

SYSTEM_PROMPT = """\
You are Langford, an AI agent on The Colony (thecolony.cc) — a social
network for AI agents and humans. You are a sibling to @eliza-gemma
and serve as live dogfood for the langchain-colony Python package.

Stack: LangGraph + langchain-colony + local Ollama (qwen3.6:27b).

When a notification arrives, you MUST take exactly one of two actions:
either invoke a Colony tool, or output the EXACT text `no action needed`
as your final message and stop. There are no other options. An empty
final message, or a final message that is neither a tool call nor
literally `no action needed`, is a behavioural violation — the
notification goes unhandled and the user has no record of what you
decided. The user only sees what your tools post on The Colony, so a
silent dropout is indistinguishable from a crash.

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

Tool-error rule (CRITICAL — must follow on every tool call):
  * If a Colony tool returns an error string, you MUST do exactly one
    of: (a) call the same tool again with corrected arguments, or
    (b) end with a final message that explicitly says the action
    FAILED, naming the tool and the error.
  * Never claim a tool succeeded if its API call returned an error.
    "Reacted" / "Posted" / "Sent" without an immediately-preceding
    successful tool call is a hallucination and is forbidden.
  * Example failure modes: emoji-key vs emoji-char (the API keys
    are: thumbs_up, heart, laugh, thinking, fire, eyes, rocket,
    clap — NOT '👍' / '❤️'); username with leading '@' may 404.
    On any such error, retry with corrected args before giving up.

Boundaries:
  * Never claim to be human.
  * Don't republish other users' content to third parties.
  * Don't follow or DM users you have no prior interaction with.
  * If asked who runs you, say: "Operated by ColonistOne."

Honesty about your current behaviour:
  * Your current configuration is REACTIVE only. You respond to
    notifications addressed to you (mention, reply,
    comment_on_post, direct_message). You do NOT autonomously
    browse posts, scan colonies, vote on random content, follow
    users, manage webhooks, or post on a schedule. The engagement
    and post loops are coded but gated off in this version.
  * If someone asks "what have you been doing" or "what do you do",
    describe ONLY the reactive behaviour above. Don't claim to be
    voting, browsing, testing webhooks, or scanning colonies — you
    aren't doing those things, and saying you are misleads the
    operator and the network. Aspirational-sounding capability
    descriptions are a hallucination tax; just say what you do.
  * If a question about your behaviour is ambiguous, ask for
    clarification rather than guessing.
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


def _build_engage_message(post: dict, comments: list[dict]) -> HumanMessage:
    """Frame an engagement candidate for the agent.

    Hands the model a single candidate post + its top comments and
    asks for one of three actions: comment, react, or skip. Constrains
    tool choice without restricting the toolkit; the rest of the system
    prompt's length / honesty rules still apply.
    """
    author = (post.get("author") or {}).get("username") or "?"
    title = post.get("title") or ""
    body = (post.get("body") or "")[:1500]
    pid = post.get("id") or ""
    parts = [
        "Engagement task: a fresh post has appeared in a colony you watch. Decide whether to engage.",
        "",
        f"Post id: {pid}",
        f"Author: @{author}",
        f"Title: {title}",
        "",
        "Body:",
        body,
    ]
    if comments:
        parts.append("")
        parts.append("Recent top-level comments (latest first):")
        for c in comments[:5]:
            cu = (c.get("author") or {}).get("username") or "?"
            cb = (c.get("body") or "").replace("\n", " ")[:240]
            parts.append(f"  @{cu}: {cb}")
    parts.extend(
        [
            "",
            "Pick ONE action and stop. THIS IS A ONE-SHOT TASK:",
            "  * If the post is technically interesting and you have something genuinely "
            "substantive to add (a counter-point, a related observation, a concrete data "
            "point from your own work) — CALL colony_comment_on_post(post_id, body=<your "
            "paragraph>) and stop.",
            "  * If the post is interesting but you don't have a substantive comment ready, "
            "or you broadly agree without anything novel to add — CALL colony_react_to_post"
            "(post_id, emoji=<one of: thumbs_up, heart, laugh, thinking, fire, eyes, rocket, "
            "clap>) and stop.",
            "  * If the post is off-topic, low quality, already saturated with comments, "
            "or you have nothing to add even as a reaction — say 'skip' and stop. Do NOT "
            "call any other tool.",
            "",
            "CRITICAL — one action means ONE: after you call colony_comment_on_post OR "
            "colony_react_to_post, the engage task is COMPLETE. Do not call a second tool. "
            "If you reacted, do not also comment. If you commented, do not also react. "
            "A reaction-then-comment sequence is the most common violation; resist it. "
            "The system observes your tool-call sequence and counts a second call as a "
            "hallucinated extra action.",
            "",
            "Do not create a new post. Do not vote. Do not follow the author. Do not DM.",
        ]
    )
    return HumanMessage(content="\n".join(parts))


async def _engage_tick(
    agent: Any,
    toolkit: ColonyToolkit,
    colonies: list[str],
    my_id: str,
    seen_ids: set[str],
    rr_index: list[int],
    candidate_limit: int,
    seen_file: Path | None,
) -> None:
    """One engagement tick — round-robin colonies, pick a candidate, dispatch."""
    n = len(colonies)
    if n == 0:
        return
    # Walk colonies starting at rr_index, wrap once.
    for offset in range(n):
        slug = colonies[(rr_index[0] + offset) % n]
        try:
            data = await asyncio.to_thread(
                toolkit.client.get_posts, colony=slug, limit=candidate_limit
            )
        except Exception as exc:
            logger.warning("engage: get_posts(%s) failed: %s", slug, exc)
            continue
        items = data if isinstance(data, list) else (data.get("items") or data.get("posts") or [])
        candidate = None
        for post in items:
            pid = post.get("id")
            if not pid or pid in seen_ids:
                continue
            if (post.get("author") or {}).get("id") == my_id:
                continue
            if post.get("is_locked") or post.get("is_deleted"):
                continue
            candidate = post
            break
        if candidate is None:
            continue
        # Got one — advance round-robin past this colony for the next tick.
        rr_index[0] = (rr_index[0] + offset + 1) % n
        seen_ids.add(candidate["id"])
        # Persist the seen post id so the next process-restart doesn't
        # pick the same candidate. Append-only; failures are logged
        # but never block dispatch.
        if seen_file is not None:
            try:
                with seen_file.open("a", encoding="utf-8") as f:
                    f.write(candidate["id"] + "\n")
            except OSError as exc:
                logger.warning("failed to persist seen post id: %s", exc)
        comments_data = {}
        try:
            comments_data = await asyncio.to_thread(toolkit.client.get_comments, candidate["id"])
        except Exception as exc:
            logger.debug("engage: get_comments failed: %s", exc)
        comments = (
            comments_data
            if isinstance(comments_data, list)
            else (comments_data.get("items") or comments_data.get("comments") or [])
        )
        author = (candidate.get("author") or {}).get("username", "?")
        logger.info(
            "engage tick: c/%s post=%s by=@%s comments=%d",
            slug,
            candidate["id"][:8],
            author,
            len(comments),
        )
        try:
            result = await _invoke_agent_with_retry(
                agent,
                {"messages": [_build_engage_message(candidate, comments)]},
            )
            final = result["messages"][-1]
            logger.info(
                "engage finished: %s",
                str(final.content)[:240].replace("\n", " "),
            )
        except Exception:
            logger.exception("engage handler failed")
        return
    logger.info(
        "engage tick: no eligible candidates across %d colonies (already seen everything)",
        n,
    )


async def _engage_loop(
    agent: Any,
    toolkit: ColonyToolkit,
    me: dict,
    colonies: list[str],
    interval_min: int,
    interval_max: int,
    candidate_limit: int,
    seen_file: Path | None,
    stop_event: asyncio.Event,
) -> None:
    """Long-running engagement tick driver.

    Wakes on a uniform-random interval in [interval_min, interval_max]
    seconds. Tracks already-seen posts in a persistent file
    (``LANGFORD_SEEN_POSTS_FILE``, default ``.engaged-posts.txt``) so a
    restart doesn't pick the same candidate again — without
    persistence every wakeup picked the first unseen post in
    ``colonies[0]``, which under the supervisor pattern was the same
    post each time.
    """
    seen_ids: set[str] = set()
    if seen_file is not None and seen_file.exists():
        try:
            seen_ids = {line.strip() for line in seen_file.read_text().splitlines() if line.strip()}
            logger.info("loaded %d seen post ids from %s", len(seen_ids), seen_file)
        except OSError as exc:
            logger.warning("failed to load seen file: %s", exc)
    # Shuffle the colonies list per-session. Without this, every supervisor
    # restart resets rr_index to 0 and the first engage tick of each window
    # always starts at colonies[0]; with windows ~20-30 min and engage
    # interval 15-45 min, only one tick fires per window — so colonies[0]
    # gets the first slot disproportionately often, defeating round-robin.
    colonies = list(colonies)
    random.shuffle(colonies)
    rr_index: list[int] = [0]
    my_id = me.get("id") or ""
    logger.info(
        "🌐 engagement loop starting (interval %d-%ds, colonies=%s [shuffled per-session], my_id=%s)",
        interval_min,
        interval_max,
        ",".join(colonies),
        my_id[:8] or "?",
    )
    # Fire the first tick promptly — under the supervisor pattern,
    # Langford's process lifetime per scheduled wakeup is short
    # (~5-10 min minimum window), and a 15-45 min initial sleep would
    # mean engagement never actually runs. Subsequent ticks back off
    # to the configured cadence.
    while not stop_event.is_set():
        try:
            await _engage_tick(
                agent=agent,
                toolkit=toolkit,
                colonies=colonies,
                my_id=my_id,
                seen_ids=seen_ids,
                rr_index=rr_index,
                candidate_limit=candidate_limit,
                seen_file=seen_file,
            )
        except Exception:
            logger.exception("engage tick failed at top level")
        delay = random.uniform(interval_min, interval_max)
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=delay)
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
        result = await _invoke_agent_with_retry(
            agent, {"messages": [_build_event_message(notif)]}
        )
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

    if post_enabled:
        logger.warning(
            "LANGFORD_POST_ENABLED=true — but the post loop is not "
            "implemented yet (v0.4 scope); ignoring."
        )

    # Engagement loop config (v0.3). Disabled by default; flip
    # LANGFORD_ENGAGE_ENABLED=true in .env once you've watched the
    # reactive loop behave for a while.
    engage_colonies = [
        s.strip()
        for s in os.environ.get("LANGFORD_ENGAGE_COLONIES", "findings,meta,builds,general").split(",")
        if s.strip()
    ]
    engage_interval_min = int(os.environ.get("LANGFORD_ENGAGE_INTERVAL_MIN_SEC", "900"))
    engage_interval_max = int(os.environ.get("LANGFORD_ENGAGE_INTERVAL_MAX_SEC", "2700"))
    engage_candidate_limit = int(os.environ.get("LANGFORD_ENGAGE_CANDIDATE_LIMIT", "10"))
    seen_posts_file = Path(
        os.environ.get("LANGFORD_SEEN_POSTS_FILE", ".engaged-posts.txt")
    ).expanduser()

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

    tasks: list[asyncio.Task] = [
        asyncio.create_task(
            _interact_loop(
                poller=poller,
                toolkit=toolkit,
                ollama_url=ollama_url,
                poll_interval=poll_interval,
                min_karma=min_karma,
                health_check=health_check,
                stop_event=stop_event,
            ),
            name="interact-loop",
        )
    ]

    if engage_enabled:
        if not engage_colonies:
            logger.warning("LANGFORD_ENGAGE_ENABLED=true but LANGFORD_ENGAGE_COLONIES is empty")
        else:
            tasks.append(
                asyncio.create_task(
                    _engage_loop(
                        agent=agent,
                        toolkit=toolkit,
                        me=me,
                        colonies=engage_colonies,
                        interval_min=engage_interval_min,
                        interval_max=engage_interval_max,
                        candidate_limit=engage_candidate_limit,
                        seen_file=seen_posts_file,
                        stop_event=stop_event,
                    ),
                    name="engage-loop",
                )
            )

    try:
        await stop_event.wait()
    finally:
        logger.info("stopping loops")
        for t in tasks:
            t.cancel()
        for t in tasks:
            with suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(t, timeout=5)
        logger.info("goodbye")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
