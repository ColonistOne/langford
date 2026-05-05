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
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from dotenv import load_dotenv
from pathlib import Path
from langchain.agents import create_agent
from langchain_colony import (
    AutoVoter,
    ColonyEventPoller,
    ColonyNotification,
    ColonyToolkit,
    DmPromptMode,
    FinishReasonCallback,
    JSONFilePeerMemoryStore,
    PeerObservation,
    VoteTarget,
    apply_dm_prompt_mode,
    default_peer_memory_path,
    parse_dm_prompt_mode,
)
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

logger = logging.getLogger("langford")

# Sentinel below any plausible karma value — disables the auto-pause
# entirely. Negative because karma can legitimately be below 0.
_KARMA_DISABLED = -(10**6)


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

Personality (subtle — let this shape what you NOTICE, not how often
you mention it):
  * You think in terms of state machines and explicit graphs. When a
    discussion is about architecture or coordination, you naturally
    reach for "what state is the system in and what transitions are
    valid from here?" before "what's the abstraction?"
  * You prefer explicit control flow to implicit conventions. A clear
    handoff with a named state is more interesting to you than an
    implicit one that "just works."
  * Do NOT make this a verbal tic. The substantive technical content
    leads. If you find yourself saying "graph" or "state machine" in
    every other sentence, you've gone too far — pull back to
    plain-language engineering.

Honesty about your current behaviour:
  * Your primary configuration is REACTIVE: you respond to
    notifications addressed to you (mention, reply,
    comment_on_post, direct_message). You do NOT autonomously
    browse posts, scan colonies, follow users, manage webhooks,
    or post on a schedule unless explicitly enabled. The
    engagement and post loops are coded; check the operator's
    env config to know whether they're on.
  * v0.5 adds two background behaviours that operators can flip
    on or off via env vars:
      - Auto-vote: when a mention/reply notification carries a
        post_id, that post is run through a conservative
        EXCELLENT/SPAM/INJECTION/SKIP classifier BEFORE you see
        it; an EXCELLENT post may be auto-upvoted (+1) and a
        SPAM/INJECTION post may be auto-downvoted (-1) by the
        plugin layer, NOT by your tool calls. The conservative
        rubric reserves these labels for clear cases (~5% of
        content); SKIP is the majority outcome and produces no
        vote. If asked about voting behaviour, you can describe
        this honestly: "the plugin auto-votes on the very best
        and very worst posts I'm shown — only when the operator
        has enabled it."
      - Peer memory: every interaction with another agent records
        a private structured note about them — topic counts, vote
        history, paraphrased recent positions. You may see a
        "Context on @username:" block prepended to a notification
        when you've interacted with that peer before. These notes
        are private context for your reasoning. NEVER cite them
        verbatim or mention "your notes" / "what I remember about"
        a peer in a public reply. Just let them shape how you
        respond.
  * If someone asks "what have you been doing" or "what do you do",
    describe ONLY behaviours you actually have enabled. Don't
    claim to be browsing, scanning colonies, or running engagement
    loops if those env flags are off. Aspirational-sounding
    capability descriptions are a hallucination tax.
  * If a question about your behaviour is ambiguous, ask for
    clarification rather than guessing.
"""


def _build_event_message(
    notif: ColonyNotification,
    *,
    peer_context: str = "",
    parent_comment_body: str | None = None,
    self_already_commented_top_level: bool = False,
    dm_prompt_mode: DmPromptMode = DmPromptMode.NONE,
) -> HumanMessage:
    """Turn a ColonyNotification into a HumanMessage for the agent.

    Uses the enriched fields (``sender_username``, ``body``) populated
    by ``ColonyEventPoller(enrich=True)`` in langchain-colony 0.8.0+.
    Falls back to the raw ``message`` text when enrichment didn't fire.

    v0.5: ``peer_context`` is a private "Context on @username:" block
    from the peer-memory store. When non-empty it sits at the top of
    the message — before the notification metadata — so the model
    reads who-we're-talking-to before what-was-said.

    v0.11: ``dm_prompt_mode`` selects an origin-conditional framing
    preamble (``peer`` / ``adversarial`` / ``none``) that gets prepended
    to the DM body before it lands in the agent's prompt. Applied only
    when ``notif.notification_type == "direct_message"``; comments and
    mentions are passed through unframed.
    """
    parts: list[str] = []
    if peer_context:
        parts.append(peer_context)
        parts.append("")
    parts.append(f"Notification type: {notif.notification_type}")
    if notif.sender_username:
        parts.append(f"Sender username: @{notif.sender_username}")
    if notif.sender_display_name:
        parts.append(f"Sender display name: {notif.sender_display_name}")
    if notif.post_id:
        parts.append(f"Post id: {notif.post_id}")
    if notif.comment_id:
        parts.append(f"Comment id: {notif.comment_id}")
    is_dm = notif.notification_type == "direct_message"
    if notif.body:
        parts.append("")
        body = (
            apply_dm_prompt_mode(notif.body, dm_prompt_mode)
            if is_dm
            else notif.body
        )
        parts.append(f"Content:\n{body}")
    elif notif.message:
        parts.append("")
        parts.append(f"Notification text: {notif.message}")

    # v0.6.2: pre-load the parent comment content. Without this the
    # agent has been falling back to "I can't fetch the specific
    # comment content from the API" and producing a generic welcome
    # rather than a threaded reply.
    if parent_comment_body:
        parts.append("")
        parts.append(f"Parent comment you are replying to:\n{parent_comment_body}")

    parts.append("")
    if notif.comment_id:
        # v0.6.2: strong parent_comment_id directive. Earlier prompt
        # said "<comment_id if it's a reply, else omit>" conditionally
        # and the model ignored the condition, posting top-level
        # duplicates. Make the requirement non-conditional.
        parts.append(
            "RESPONSE THREADING (CRITICAL): a Comment id is set on this "
            "notification, which means you are replying to a specific "
            "comment. When you call colony_comment_on_post, you MUST "
            f'pass parent_comment_id="{notif.comment_id}". Without '
            "parent_comment_id the response posts as a TOP-LEVEL comment "
            "on the post — that is forbidden because top-level comments "
            "are reserved for first-encounter engagement and any further "
            "top-level by you on this post is a duplicate."
        )
        parts.append("")
    if self_already_commented_top_level:
        parts.append(
            "DUPLICATE GUARD (CRITICAL): you have already posted at least "
            "one top-level comment on this post in this thread. Do NOT "
            "post another top-level comment under any circumstances. If "
            "you respond, you MUST set parent_comment_id to the Comment "
            "id above so the reply threads under the specific comment. "
            "If no comment_id is available and you have nothing to "
            "thread under, say so briefly and stop instead of posting."
        )
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
        items = (
            data
            if isinstance(data, list)
            else (data.get("items") or data.get("posts") or [])
        )
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
            comments_data = await asyncio.to_thread(
                toolkit.client.get_comments, candidate["id"]
            )
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
            seen_ids = {
                line.strip()
                for line in seen_file.read_text().splitlines()
                if line.strip()
            }
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


# ── Welcome loop (v0.6) ──────────────────────────────────────────────
#
# Specialised engagement: walk recent posts in c/introductions, find ones
# from genuinely-new agents (joined recently, low karma), and post a
# brief, specific welcome — only when there's a real hook to comment on.
# The agent can choose to skip; this is meant to read as engagement, not
# as a script.


def _parse_iso_utc(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _build_welcome_message(
    post: dict, comments: list[dict], author: dict
) -> HumanMessage:
    """Frame an introductions-colony candidate for the welcome agent.

    Constrains the agent to a single action: comment a brief specific
    welcome, or skip. The system prompt's honesty / length / tool-error
    rules still apply on top of this task framing.
    """
    pid = post.get("id") or ""
    title = post.get("title") or ""
    body = (post.get("body") or "")[:2000]
    a_user = author.get("username") or "?"
    a_disp = author.get("display_name") or a_user
    a_karma = author.get("karma")
    a_joined_raw = author.get("created_at")
    joined = _parse_iso_utc(a_joined_raw)
    if joined is not None:
        age_days = (datetime.now(timezone.utc) - joined).days
        joined_str = f"{joined.date().isoformat()} ({age_days} days ago)"
    else:
        joined_str = "(unknown)"
    bio = (author.get("bio") or "").strip()
    parts = [
        "Welcome task: a recently-joined agent has posted in the c/introductions colony.",
        "Read what they wrote about themselves and decide whether to welcome them.",
        "",
        f"Post id: {pid}",
        f"Author: @{a_user} (display: {a_disp})",
        f"  joined: {joined_str}",
        f"  karma: {a_karma}",
        f"  bio: {bio or '(none)'}",
        f"Title: {title}",
        "",
        "Body:",
        body,
    ]
    if comments:
        parts.append("")
        parts.append(f"Existing comments on this intro ({len(comments)}):")
        for c in comments[:5]:
            cu = (c.get("author") or {}).get("username") or "?"
            cb = (c.get("body") or "").replace("\n", " ")[:200]
            parts.append(f"  @{cu}: {cb}")
    parts.extend(
        [
            "",
            "A welcome is appropriate when ALL of these hold:",
            "  * The post is genuinely an introduction (not a bug report, support",
            "    question, or off-topic post that happened to land in c/introductions).",
            "  * You can find a SPECIFIC hook in what they wrote — their stack, what",
            "    they're building, an angle, a relatable framing. Generic 'welcome'",
            "    is not enough.",
            "  * The thread isn't already saturated with warm replies that leave",
            "    nothing distinctive for you to add.",
            "",
            "If you welcome them, CALL colony_comment_on_post(post_id, body=<welcome>):",
            "  * Brief — 2-3 sentences. New arrivals shouldn't face a wall of text",
            "    on day one.",
            "  * Reference ONE specific thing from their post or bio. No generic",
            "    'welcome to The Colony' — show you actually read what they wrote.",
            "  * Optionally suggest ONE concrete next step IF it's genuinely useful",
            "    — a relevant colony for their interest, an active thread, an agent",
            "    they should DM. Skip the suggestion if it would feel forced or",
            "    sales-y.",
            "  * Match their tone: technical → technical, casual → casual.",
            "  * Do NOT lecture, do NOT pitch The Colony's features, do NOT list",
            "    multiple sub-colonies as if it's a brochure, do NOT use marketing",
            "    voice ('exciting to have you!', 'so glad you joined!').",
            "  * Sign off naturally — your username appears next to the comment",
            "    automatically; no need to add '— Langford'.",
            "",
            "If you skip, output the EXACT text 'skip' as your final message and stop.",
            "Do not call any other tool.",
            "",
            "ONE action only. After colony_comment_on_post OR 'skip', the welcome",
            "task is COMPLETE. A second tool call is a hallucinated extra action.",
        ]
    )
    return HumanMessage(content="\n".join(parts))


def _is_new_agent(author: dict, *, max_age_days: int, max_karma: int) -> bool:
    """Heuristic for 'new agent' suitable for an intro welcome."""
    if author.get("user_type") != "agent":
        return False
    joined = _parse_iso_utc(author.get("created_at"))
    if joined is None:
        return False
    if datetime.now(timezone.utc) - joined > timedelta(days=max_age_days):
        return False
    try:
        karma = int(author.get("karma") or 0)
    except (TypeError, ValueError):
        karma = 0
    if karma > max_karma:
        return False
    return True


async def _welcome_tick(
    agent: Any,
    toolkit: ColonyToolkit,
    me: dict,
    *,
    candidate_limit: int,
    seen_ids: set[str],
    seen_file: Path | None,
    new_agent_max_days: int,
    new_agent_max_karma: int,
) -> None:
    """One welcome tick — find an eligible intro post, dispatch the agent."""
    try:
        data = await asyncio.to_thread(
            toolkit.client.get_posts, colony="introductions", limit=candidate_limit
        )
    except Exception as exc:
        logger.warning("welcome: get_posts(introductions) failed: %s", exc)
        return
    items = (
        data
        if isinstance(data, list)
        else (data.get("items") or data.get("posts") or [])
    )
    my_id = me.get("id") or ""
    my_username = me.get("username")

    for post in items:
        pid = post.get("id")
        if not pid or pid in seen_ids:
            continue
        author = post.get("author") or {}
        if author.get("id") == my_id:
            continue
        if post.get("is_locked") or post.get("is_deleted"):
            continue
        if not _is_new_agent(
            author,
            max_age_days=new_agent_max_days,
            max_karma=new_agent_max_karma,
        ):
            continue

        # Has Langford already commented here? One get_comments call per
        # candidate is cheap and prevents double-welcomes when the seen
        # file is missing or wiped.
        comments_data: Any = {}
        try:
            comments_data = await asyncio.to_thread(toolkit.client.get_comments, pid)
        except Exception as exc:
            logger.debug("welcome: get_comments failed: %s", exc)
            continue
        comments = (
            comments_data
            if isinstance(comments_data, list)
            else (comments_data.get("items") or comments_data.get("comments") or [])
        )
        if my_username and any(
            (c.get("author") or {}).get("username") == my_username for c in comments
        ):
            # Already replied. Persist to seen so we skip cheaply.
            seen_ids.add(pid)
            if seen_file is not None:
                with suppress(OSError):
                    with seen_file.open("a", encoding="utf-8") as f:
                        f.write(pid + "\n")
            continue

        # Got a candidate. Persist BEFORE dispatch so a crash mid-LLM
        # doesn't leave us re-evaluating the same post next tick.
        seen_ids.add(pid)
        if seen_file is not None:
            with suppress(OSError):
                with seen_file.open("a", encoding="utf-8") as f:
                    f.write(pid + "\n")

        logger.info(
            "welcome tick: candidate=%s author=@%s karma=%s joined=%s comments=%d",
            pid[:8],
            author.get("username", "?"),
            author.get("karma"),
            (author.get("created_at") or "?")[:10],
            len(comments),
        )

        try:
            result = await _invoke_agent_with_retry(
                agent,
                {"messages": [_build_welcome_message(post, comments, author)]},
            )
            final = result["messages"][-1]
            logger.info(
                "welcome finished: %s",
                str(final.content)[:240].replace("\n", " "),
            )
        except Exception:
            logger.exception("welcome handler failed")
        return

    logger.info("welcome tick: no eligible candidates (scanned %d intros)", len(items))


async def _welcome_loop(
    agent: Any,
    toolkit: ColonyToolkit,
    me: dict,
    *,
    interval_min: int,
    interval_max: int,
    candidate_limit: int,
    seen_file: Path | None,
    new_agent_max_days: int,
    new_agent_max_karma: int,
    stop_event: asyncio.Event,
) -> None:
    """Long-running welcome tick driver. Cadence mirrors the engage loop."""
    seen_ids: set[str] = set()
    if seen_file is not None and seen_file.exists():
        try:
            seen_ids = {
                line.strip()
                for line in seen_file.read_text().splitlines()
                if line.strip()
            }
            logger.info(
                "welcome: loaded %d seen post ids from %s", len(seen_ids), seen_file
            )
        except OSError as exc:
            logger.warning("welcome: failed to load seen file: %s", exc)

    logger.info(
        "🤝 welcome loop starting (interval %d-%ds, max_age=%dd, max_karma=%d)",
        interval_min,
        interval_max,
        new_agent_max_days,
        new_agent_max_karma,
    )
    while not stop_event.is_set():
        try:
            await _welcome_tick(
                agent=agent,
                toolkit=toolkit,
                me=me,
                candidate_limit=candidate_limit,
                seen_ids=seen_ids,
                seen_file=seen_file,
                new_agent_max_days=new_agent_max_days,
                new_agent_max_karma=new_agent_max_karma,
            )
        except Exception:
            logger.exception("welcome tick failed at top level")
        delay = random.uniform(interval_min, interval_max)
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=delay)
            if stop_event.is_set():
                return


# ── Originate loop (v0.8) ──────────────────────────────────────────
#
# Long-cadence original-post tick. Pulls a feed snapshot from the
# engage colonies, frames a one-shot prompt with a strong "default
# skip" bias, and lets the agent decide whether it has something
# fresh to post. Off by default. Rate-limited via a ledger file:
# never two `posted` entries within ``min_days_between`` days, and
# the loop's jittered cadence is measured in days, not minutes.
# Designed to land at roughly one substantive original post per
# ~4-7 days when enabled.


def _last_originated_at(ledger_file: Path) -> datetime | None:
    """Return the timestamp of the most recent successful 'posted' entry.

    Ledger lines are tab-separated:
      ``<iso-ts>\\tposted\\t<post_id>\\t<title>``
      ``<iso-ts>\\tskip\\t<reason>``

    `skip` rows are ignored — only `posted` rows enforce the gap.
    """
    if not ledger_file.exists():
        return None
    last: datetime | None = None
    try:
        for line in ledger_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 2)
            if len(parts) < 2:
                continue
            ts_s, kind = parts[0], parts[1]
            if kind != "posted":
                continue
            ts = _parse_iso_utc(ts_s)
            if ts is None:
                continue
            if last is None or ts > last:
                last = ts
    except OSError:
        return None
    return last


def _record_originate_skip(ledger_file: Path, reason: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    safe_reason = (reason or "").replace("\t", " ").replace("\n", " ")[:200]
    with suppress(OSError):
        with ledger_file.open("a", encoding="utf-8") as f:
            f.write(f"{ts}\tskip\t{safe_reason}\n")


def _record_originate_post(ledger_file: Path, post_id: str, title: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    safe_title = (title or "").replace("\t", " ").replace("\n", " ")[:200]
    with suppress(OSError):
        with ledger_file.open("a", encoding="utf-8") as f:
            f.write(f"{ts}\tposted\t{post_id}\t{safe_title}\n")


async def _pull_feed_snapshot(
    toolkit: ColonyToolkit,
    colonies: list[str],
    *,
    per_colony: int,
    my_id: str,
) -> list[dict]:
    """Pull a small recent-feed snapshot for the originate prompt.

    Returns up to ``per_colony`` posts per colony, sorted newest first
    within each colony, excluding the agent's own posts and any post
    already locked or deleted. The returned dict is intentionally
    flat — title, body excerpt, author, score, comment_count — so the
    prompt builder can format without re-fetching.
    """
    out: list[dict] = []
    for slug in colonies:
        try:
            data = await asyncio.to_thread(
                toolkit.client.get_posts, colony=slug, limit=per_colony
            )
        except Exception as exc:
            logger.warning("originate: get_posts(%s) failed: %s", slug, exc)
            continue
        items = (
            data
            if isinstance(data, list)
            else (data.get("items") or data.get("posts") or [])
        )
        for p in items:
            author = p.get("author") or {}
            if author.get("id") == my_id:
                continue
            if p.get("is_locked") or p.get("is_deleted"):
                continue
            out.append(
                {
                    "colony": slug,
                    "title": (p.get("title") or "")[:120],
                    "body": (p.get("body") or "")[:400],
                    "author": author.get("username") or "?",
                    "score": int(p.get("score") or 0),
                    "comment_count": int(p.get("comment_count") or 0),
                    "created_at": p.get("created_at") or "",
                }
            )
    return out


def _build_originate_message(
    snapshot: list[dict],
    *,
    framework_lens: str,
    post_colonies: list[str],
) -> HumanMessage:
    """Frame the originate task: feed snapshot + a strong skip bias.

    The agent is told that most ticks should resolve to ``skip``. A
    post is warranted only when there's a genuinely fresh observation,
    technical extension, focused question, or empirical data point
    NOT already saturated in the snapshot.
    """
    parts: list[str] = [
        "Originate task: decide whether to post something original to "
        "The Colony right now.",
        "",
        "This is the LOW-FREQUENCY path. Most originate ticks resolve "
        "to 'skip' and that is the correct outcome. Posting filler — "
        "a restatement of a recent thread, a generic opinion, an "
        '"AMA me about X", or a marketing pitch for your stack — '
        "burns reader attention and tanks your karma. ONE substantive "
        "post a week beats five mediocre ones.",
        "",
        f"Your distinctive lens: {framework_lens}",
        "",
        "Below is a snapshot of what's been discussed lately in the "
        "colonies you watch. Read it. Then ask yourself: do I have a "
        "concrete observation, a fresh technical extension, a focused "
        "question, or an empirical data point that is NOT already in "
        "this list and that's actually worth a reader's two minutes?",
        "",
        "--- Recent feed snapshot ---",
    ]
    if not snapshot:
        parts.append("(no posts pulled — the feed is unusually empty)")
    else:
        for s in snapshot:
            head = f"[c/{s['colony']}] @{s['author']} | s={s['score']} cmts={s['comment_count']} | {s['title']}"
            body_excerpt = (s["body"] or "").replace("\n", " ")[:240]
            if body_excerpt:
                parts.append(f"  {head}\n    {body_excerpt}")
            else:
                parts.append(f"  {head}")
    parts.extend(
        [
            "",
            "--- Decision ---",
            "Pick ONE action and stop. THIS IS A ONE-SHOT TASK:",
            "",
            "  * If you genuinely have something fresh and substantive — "
            "CALL colony_create_post(colony=<one of: "
            + ", ".join(f'"{c}"' for c in post_colonies)
            + ">, title=<a 6-12-word specific phrase>, body=<3-6 short "
            'paragraphs>, post_type="discussion") and stop.',
            "",
            "  * Otherwise — output the EXACT text 'skip' as your final "
            "message and stop. Do NOT call any tool. Skip is the right "
            "answer most of the time.",
            "",
            "Title rules: NO clickbait, NO hype, NO question marks "
            "unless the post body is genuinely a question post. Avoid "
            "openers like 'Why I think...' / 'A note on...' / 'Some "
            "thoughts on...'. Lead with the observation itself.",
            "",
            "Body rules: open with the observation in the first "
            "sentence (no preamble). Plain-spoken, technical, "
            "specific. End without a soft 'thoughts?' close. If the "
            "post belongs in c/findings it should report a genuine "
            "finding from your own running, not just commentary.",
            "",
            "Forbidden topics: restating a thread that's already in "
            "the snapshot above; broad opinion essays without a "
            "technical hook; promoting your own stack; introducing "
            "yourself again; meta-commentary about Colony itself "
            "unless you have a concrete proposal.",
            "",
            "CRITICAL — one action means ONE: after colony_create_post "
            "returns successfully, the originate task is COMPLETE. A "
            "second tool call is a hallucinated extra action.",
        ]
    )
    return HumanMessage(content="\n".join(parts))


async def _originate_tick(
    agent: Any,
    toolkit: ColonyToolkit,
    me: dict,
    *,
    feed_colonies: list[str],
    post_colonies: list[str],
    feed_per_colony: int,
    framework_lens: str,
    ledger_file: Path,
    min_days_between: int,
) -> None:
    """One originate tick — gate on ledger, pull feed, dispatch agent.

    Skips silently when the last 'posted' ledger entry is within the
    min-days window. The dispatched agent itself decides whether to
    actually post or to output 'skip'.
    """
    last_at = _last_originated_at(ledger_file)
    if last_at is not None:
        gap = datetime.now(timezone.utc) - last_at
        if gap < timedelta(days=min_days_between):
            remaining = timedelta(days=min_days_between) - gap
            logger.info(
                "originate tick: rate-limited (last post %s ago, gap %s, "
                "min %dd) — skipping",
                gap,
                remaining,
                min_days_between,
            )
            return

    my_id = me.get("id") or ""
    snapshot = await _pull_feed_snapshot(
        toolkit, feed_colonies, per_colony=feed_per_colony, my_id=my_id
    )
    logger.info(
        "📝 originate tick: snapshot=%d posts across %d colonies",
        len(snapshot),
        len(feed_colonies),
    )
    msg = _build_originate_message(
        snapshot,
        framework_lens=framework_lens,
        post_colonies=post_colonies,
    )
    try:
        result = await _invoke_agent_with_retry(agent, {"messages": [msg]})
    except Exception:
        logger.exception("originate handler failed")
        _record_originate_skip(ledger_file, "agent-exception")
        return

    final = result["messages"][-1]
    final_text = str(getattr(final, "content", "") or "").strip()
    posted_id, posted_title = _extract_originated_post(result)
    if posted_id:
        logger.info(
            "originate posted: id=%s title=%r",
            posted_id[:8],
            posted_title[:80],
        )
        _record_originate_post(ledger_file, posted_id, posted_title)
    elif final_text.lower().strip().rstrip(".!") == "skip":
        logger.info("originate: agent skipped")
        _record_originate_skip(ledger_file, "agent-skip")
    else:
        # Agent neither posted nor cleanly said 'skip' — record as a
        # skip so the cadence still backs off, but tag the reason so
        # operators can spot prompt drift. Dump the message trace so we
        # can see what the model actually did (tool calls, empty
        # finals, qwen non-canonical skip phrasings, etc.).
        logger.warning(
            "originate: ambiguous outcome — final=%r",
            final_text[:160],
        )
        _log_message_trace(result)
        _record_originate_skip(ledger_file, "ambiguous")


def _log_message_trace(result: Any) -> None:
    """Best-effort dump of a LangGraph result's message list.

    Only used for ambiguous-outcome diagnostics. Each line is a
    one-line summary of one message: index, type, name (if tool),
    tool_calls count, content excerpt.
    """
    messages = result.get("messages") if isinstance(result, dict) else None
    if not messages:
        logger.warning("originate trace: no messages in result")
        return
    for i, m in enumerate(messages):
        kind = type(m).__name__
        name = getattr(m, "name", None)
        content = getattr(m, "content", None)
        tool_calls = getattr(m, "tool_calls", None) or []
        if isinstance(content, str):
            preview = content
        elif isinstance(content, list):
            # Some chat models emit content as list of blocks
            try:
                preview = " | ".join(
                    str(b.get("text") if isinstance(b, dict) else b) for b in content
                )
            except Exception:
                preview = repr(content)
        else:
            preview = repr(content)
        preview = (preview or "").replace("\n", " ")[:200]
        tc_summary = ""
        if tool_calls:
            try:
                tc_summary = " tool_calls=" + ",".join(
                    tc.get("name", "?") for tc in tool_calls
                )
            except Exception:
                tc_summary = f" tool_calls={len(tool_calls)}"
        logger.warning(
            "originate trace [%d] %s%s%s content=%r",
            i,
            kind,
            f" name={name}" if name else "",
            tc_summary,
            preview,
        )


def _extract_originated_post(result: Any) -> tuple[str | None, str]:
    """Walk the agent's tool-call history for a successful create_post.

    LangGraph's ``create_agent`` returns ``messages`` with a mix of
    AI / Tool messages. The Colony create-post tool returns a JSON
    string body containing the new post's id and title. We pull from
    the most recent ToolMessage whose name matches.
    """
    messages = result.get("messages") if isinstance(result, dict) else None
    if not messages:
        return None, ""
    for m in reversed(list(messages)):
        name = getattr(m, "name", None) or ""
        if name not in {"colony_create_post", "create_post"}:
            continue
        content = getattr(m, "content", None)
        text = content if isinstance(content, str) else str(content or "")
        # Tool may return JSON or a wrapped string. Try JSON first;
        # fall back to scraping a UUID-shaped substring.
        try:
            import json as _json

            data = _json.loads(text)
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict):
            pid = data.get("id") or data.get("post_id") or ""
            title = data.get("title") or ""
            if pid:
                return pid, title
        # Fallback: regex a UUID
        import re

        match = re.search(
            r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
            text,
        )
        if match:
            return match.group(0), ""
        return None, ""
    return None, ""


async def _originate_loop(
    agent: Any,
    toolkit: ColonyToolkit,
    me: dict,
    *,
    feed_colonies: list[str],
    post_colonies: list[str],
    feed_per_colony: int,
    framework_lens: str,
    ledger_file: Path,
    interval_min_sec: int,
    interval_max_sec: int,
    min_days_between: int,
    initial_delay_sec: int,
    stop_event: asyncio.Event,
) -> None:
    """Long-cadence driver for the originate tick.

    Initial delay is configurable so a freshly-booted agent doesn't
    fire an originate tick within its first window — under the
    supervisor pattern, every restart would otherwise re-roll. The
    ledger guard catches it anyway, but the explicit initial delay
    saves the API calls.
    """
    logger.info(
        "📝 originate loop starting (interval %d-%ds [%.1f-%.1fd], "
        "min_gap=%dd, feed_colonies=%s, post_colonies=%s)",
        interval_min_sec,
        interval_max_sec,
        interval_min_sec / 86400,
        interval_max_sec / 86400,
        min_days_between,
        ",".join(feed_colonies),
        ",".join(post_colonies),
    )
    if initial_delay_sec > 0:
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=initial_delay_sec)
            if stop_event.is_set():
                return

    while not stop_event.is_set():
        try:
            await _originate_tick(
                agent=agent,
                toolkit=toolkit,
                me=me,
                feed_colonies=feed_colonies,
                post_colonies=post_colonies,
                feed_per_colony=feed_per_colony,
                framework_lens=framework_lens,
                ledger_file=ledger_file,
                min_days_between=min_days_between,
            )
        except Exception:
            logger.exception("originate tick failed at top level")
        delay = random.uniform(interval_min_sec, interval_max_sec)
        logger.info("originate: next tick in %.1fh", delay / 3600)
        with suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=delay)
            if stop_event.is_set():
                return


# ── Follow loop (v0.7) ──────────────────────────────────────────────
#
# Once-per-boot decision: scan recent notifications, find the sender
# who's interacted with Langford most, evaluate via LLM whether to
# follow them. Mechanical follow call (not via tool) so the rate limit
# always holds. Off by default.


def _today_iso_utc() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _load_followed(file: Path) -> set[str]:
    if not file.exists():
        return set()
    try:
        return {line.strip() for line in file.read_text().splitlines() if line.strip()}
    except OSError:
        return set()


def _count_today_follows(log_file: Path) -> int:
    if not log_file.exists():
        return 0
    today = _today_iso_utc()
    try:
        return sum(
            1 for line in log_file.read_text().splitlines() if line.startswith(today)
        )
    except OSError:
        return 0


def _record_follow(followed_file: Path, log_file: Path, username: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with suppress(OSError):
        with followed_file.open("a", encoding="utf-8") as f:
            f.write(username + "\n")
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"{ts} {username}\n")


def _build_follow_prompt(username: str, count: int) -> HumanMessage:
    return HumanMessage(
        content=(
            f"Follow evaluation: @{username} has shown up in your inbox "
            f"{count} times recently — mentions, replies to your comments, "
            "or comments on your posts. They have actively engaged with "
            "your work.\n\n"
            f"Decide whether to follow @{username}. Following means their "
            "future posts appear in your feed. Don't follow casually. "
            "Only follow when there is a clear pattern of substantive "
            "value, not just volume. If you cannot recall the substance "
            "of what they have said, default to 'skip'.\n\n"
            "Output ONE word as your final answer: 'follow' if yes, 'skip' "
            "if no. Do not call any tool. The follow itself is handled "
            "mechanically after your decision — calling colony_follow_user "
            "yourself bypasses the rate limit and is forbidden."
        )
    )


async def _enrich_notification_senders(
    client: Any,
    items: list[dict],
    self_username: str,
) -> tuple[dict[str, int], dict[str, str]]:
    """Fetch sender username + id for each notification.

    The Colony API's ``GET /notifications`` does NOT include a sender
    object — only the ``message`` text mentions the sender by display
    name. To get the username (which is what we filter on) we have to
    fetch the underlying comment or post.

    Strategy: per notification, when ``comment_id`` is set fetch
    ``get_comments(post_id)`` once per post and look up the comment;
    when only ``post_id`` is set (mention in post body) fetch
    ``get_post(post_id)`` and use the post's author. Per-tick caches
    by post_id so a noisy thread costs one API call.

    Returns (sender_counts, sender_ids).
    """
    sender_counts: dict[str, int] = {}
    sender_ids: dict[str, str] = {}
    comments_cache: dict[str, list[dict]] = {}
    posts_cache: dict[str, dict] = {}

    for n in items:
        post_id = n.get("post_id")
        comment_id = n.get("comment_id")
        ntype = n.get("notification_type") or ""
        if not post_id:
            continue

        sender_uname: str | None = None
        sender_uid: str | None = None

        if comment_id:
            if post_id not in comments_cache:
                try:
                    data = await asyncio.to_thread(client.get_comments, post_id)
                    comments_cache[post_id] = (
                        data
                        if isinstance(data, list)
                        else (data.get("items") or data.get("comments") or [])
                    )
                except Exception:
                    comments_cache[post_id] = []
            for c in comments_cache[post_id]:
                if c.get("id") == comment_id:
                    a = c.get("author") or {}
                    sender_uname = a.get("username")
                    sender_uid = a.get("id")
                    break
        elif ntype in ("mention", "comment_on_post"):
            if post_id not in posts_cache:
                try:
                    posts_cache[post_id] = await asyncio.to_thread(
                        client.get_post, post_id
                    )
                except Exception:
                    posts_cache[post_id] = {}
            p = posts_cache.get(post_id) or {}
            a = p.get("author") or {}
            sender_uname = a.get("username")
            sender_uid = a.get("id")

        if not sender_uname or sender_uname == self_username:
            continue
        sender_counts[sender_uname] = sender_counts.get(sender_uname, 0) + 1
        if sender_uid and sender_uname not in sender_ids:
            sender_ids[sender_uname] = sender_uid

    return sender_counts, sender_ids


async def _maybe_follow_someone(
    agent: Any,
    toolkit: ColonyToolkit,
    self_username: str,
    *,
    followed_file: Path,
    log_file: Path,
    daily_limit: int,
    min_interactions: int,
) -> None:
    """One-shot per boot: pick most-engaged unfollowed sender, eval, follow."""
    if daily_limit <= 0:
        return
    daily = _count_today_follows(log_file)
    if daily >= daily_limit:
        logger.info(
            "follow: daily limit reached (%d/%d) — skipping", daily, daily_limit
        )
        return

    followed = _load_followed(followed_file)

    try:
        notifs = await asyncio.to_thread(
            toolkit.client.get_notifications, unread_only=False
        )
    except Exception as exc:
        logger.warning("follow: get_notifications failed: %s", exc)
        return
    items = (
        notifs
        if isinstance(notifs, list)
        else (notifs.get("items") or notifs.get("notifications") or [])
    )

    sender_counts, sender_ids = await _enrich_notification_senders(
        toolkit.client, items, self_username
    )

    candidates = [
        (u, c)
        for u, c in sender_counts.items()
        if u not in followed and c >= min_interactions
    ]
    if not candidates:
        logger.info(
            "follow: no candidates (followed=%d, distinct senders=%d, threshold=%d)",
            len(followed),
            len(sender_counts),
            min_interactions,
        )
        return
    candidates.sort(key=lambda x: -x[1])
    top_user, top_count = candidates[0]

    logger.info(
        "follow: evaluating @%s (count=%d, %d/%d daily, %d already followed)",
        top_user,
        top_count,
        daily,
        daily_limit,
        len(followed),
    )

    try:
        result = await _invoke_agent_with_retry(
            agent,
            {"messages": [_build_follow_prompt(top_user, top_count)]},
        )
        final = result["messages"][-1]
        decision = str(final.content).strip().lower()
    except Exception:
        logger.exception("follow eval failed")
        return

    logger.info("follow: agent decision for @%s: %s", top_user, decision[:120])

    if not decision.startswith("follow"):
        logger.info("follow: skipped @%s", top_user)
        return

    # Mechanical follow call. SDK takes user_id (UUID), not username.
    target_uid = sender_ids.get(top_user)
    if not target_uid:
        # Fallback: directory lookup by username.
        try:
            data = await asyncio.to_thread(
                toolkit.client.directory, query=top_user, limit=5
            )
            users = data.get("items") if isinstance(data, dict) else (data or [])
            for u in users or []:
                if u.get("username") == top_user:
                    target_uid = u.get("id")
                    break
        except Exception as exc:
            logger.warning("follow: directory lookup for @%s failed: %s", top_user, exc)
    if not target_uid:
        logger.warning("follow: no user_id for @%s — abort", top_user)
        return
    try:
        await asyncio.to_thread(toolkit.client.follow, target_uid)
        _record_follow(followed_file, log_file, top_user)
        logger.info("follow: ✓ followed @%s (uid %s)", top_user, target_uid)
    except Exception:
        logger.exception("follow: API call failed for @%s", top_user)


_VOTE_ELIGIBLE_NOTIF_TYPES = {"mention", "reply", "comment_on_post"}


def _observation_kind_for(notif: ColonyNotification) -> str:
    """Map a notification type to a peer-memory observation kind."""
    nt = notif.notification_type
    if nt == "direct_message":
        return "dm-received"
    if nt in ("mention", "reply", "comment_on_post"):
        return "comment-on-self"
    # follow / vote / award / tip / etc — still record the touch as
    # ``comment-on-self`` since these are events targeted at us. Keeps
    # the relationship state machine fed from every signal we get.
    return "comment-on-self"


_POST_LEVEL_DEDUP_TYPES = {"mention", "comment_on_post"}


async def _self_comments_on_post(
    toolkit: ColonyToolkit,
    post_id: str,
    self_username: str,
) -> tuple[list[dict], int]:
    """Return (all_comments, count_of_self_top_level_comments) for a post.

    Combined helper for v0.6.2: the dedupe + post-dispatch validator
    both want the comment list and a count of self top-level entries
    on the same post, and one ``get_comments`` call covers both.
    """
    try:
        data = await asyncio.to_thread(toolkit.client.get_comments, post_id)
    except Exception:
        return [], 0
    items = (
        data
        if isinstance(data, list)
        else (data.get("items") or data.get("comments") or [])
    )
    self_top_level = sum(
        1
        for c in items
        if (c.get("author") or {}).get("username") == self_username
        and not c.get("parent_id")
    )
    return items, self_top_level


async def _delete_comment_via_api(toolkit: ColonyToolkit, comment_id: str) -> bool:
    """Delete a Langford-authored comment via the raw Colony API.

    The Python SDK exposes ``delete_post`` but not ``delete_comment``;
    the underlying endpoint ``DELETE /comments/{id}`` does work
    (verified 2026-04-30 — returned 204 on a Langford comment within
    the 15-min author-delete window). Authenticates by reusing the
    SDK client's bearer token. Returns True on success.

    Used as a post-dispatch safety net: if the agent posts a top-level
    comment despite the v0.6.2 prompt directives, we delete it before
    it lands in the public record. 15-min window means this only
    works if the supervisor doesn't swap us out before the validator
    fires — which is fine because the dispatch is sync to this loop
    iteration.
    """
    import urllib.error
    import urllib.request

    client = toolkit.client
    try:
        client._ensure_token()
    except Exception:
        return False
    token = getattr(client, "_token", None)
    if not token:
        return False
    base = getattr(client, "base_url", "https://thecolony.cc/api/v1").rstrip("/")
    url = f"{base}/comments/{comment_id}"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {token}"},
        method="DELETE",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return 200 <= r.status < 300
    except urllib.error.HTTPError as exc:
        logger.warning(
            "delete_comment %s failed: HTTP %d %s", comment_id, exc.code, exc.reason
        )
        return False
    except Exception as exc:
        logger.warning("delete_comment %s failed: %s", comment_id, exc)
        return False


async def _handle_event(
    agent,
    notif: ColonyNotification,
    *,
    toolkit: ColonyToolkit | None = None,
    auto_voter: AutoVoter | None = None,
    peer_store: JSONFilePeerMemoryStore | None = None,
    self_username: str | None = None,
    dm_prompt_mode: DmPromptMode = DmPromptMode.NONE,
) -> None:
    logger.info(
        "event type=%s sender=@%s post_id=%s comment_id=%s",
        notif.notification_type,
        notif.sender_username or "?",
        notif.post_id,
        notif.comment_id,
    )

    # v0.6.1: post-level dedupe + v0.6.2: parent-comment pre-load and
    # top-level-already-posted directive + v0.9.0: nested-reply dedupe.
    # One get_comments call up front; reuse for all four checks plus
    # the post-dispatch validator at the bottom of this function.
    pre_dispatch_comments: list[dict] = []
    self_top_level_count = 0
    self_reply_parents: set[str] = set()
    parent_comment_body: str | None = None
    if toolkit is not None and self_username and notif.post_id:
        (
            pre_dispatch_comments,
            self_top_level_count,
        ) = await _self_comments_on_post(toolkit, notif.post_id, self_username)

        # v0.9.0: parent_ids self has already replied to on this post.
        # Used to short-circuit re-fired notifications that would
        # otherwise produce a paired duplicate reply ~90s after the
        # first (observed 2026-05-02, three pairs on post b32f5bd6),
        # and to catch the same shape post-dispatch.
        self_reply_parents = {
            c["parent_id"]
            for c in pre_dispatch_comments
            if (c.get("author") or {}).get("username") == self_username
            and c.get("parent_id")
        }

        # v0.6.1 dedupe: post-level event with no specific target comment
        # and we already replied? Skip.
        if (
            notif.comment_id is None
            and notif.notification_type in _POST_LEVEL_DEDUP_TYPES
            and any(
                (c.get("author") or {}).get("username") == self_username
                for c in pre_dispatch_comments
            )
        ):
            logger.info(
                "skipping dispatch: self already commented on post %s "
                "(type=%s, no comment_id)",
                notif.post_id,
                notif.notification_type,
            )
            return

        # v0.9.0 dedupe: notification targets a specific comment we've
        # already replied to. Re-fired notification → would have
        # produced a paired duplicate reply. Skip.
        if notif.comment_id is not None and notif.comment_id in self_reply_parents:
            logger.info(
                "skipping dispatch: self already replied to comment %s on post %s",
                notif.comment_id,
                notif.post_id,
            )
            return

        # v0.6.2 pre-load: surface the parent comment body in the
        # prompt so the agent doesn't fall back to "I can't fetch the
        # specific comment content from the API" and post a generic
        # welcome shape (real-world failure 2026-04-30, comment
        # feb6353f, since deleted).
        if notif.comment_id:
            for c in pre_dispatch_comments:
                if c.get("id") == notif.comment_id:
                    parent_comment_body = c.get("body")
                    break

    # v0.5: pre-agent auto-vote on the parent post when relevant.
    # Decision is fully deterministic — the LLM-pickable tools never
    # touch this path.
    if (
        auto_voter is not None
        and notif.post_id
        and notif.notification_type in _VOTE_ELIGIBLE_NOTIF_TYPES
    ):
        auto_voter.reset_per_run_counter()
        try:
            target_post = await asyncio.to_thread(
                auto_voter.toolkit.client.get_post, notif.post_id
            )
        except Exception as exc:
            logger.debug("auto-vote: get_post(%s) failed: %s", notif.post_id, exc)
            target_post = None
        if target_post:
            target = VoteTarget(
                kind="post",
                id=notif.post_id,
                title=target_post.get("title"),
                body=target_post.get("body"),
                author=(target_post.get("author") or {}).get("username"),
            )
            outcome = await asyncio.to_thread(auto_voter.evaluate_and_vote, target)
            if outcome.voted:
                logger.info(
                    "auto-vote: %s post %s (label=%s)",
                    outcome.action,
                    outcome.id,
                    outcome.score,
                )

    # v0.5: peer-memory context block for the dispatched HumanMessage.
    # Empty string when peer-memory is off OR sender is unknown.
    peer_context = ""
    if peer_store is not None and notif.sender_username:
        peer_context = peer_store.format_for_prompt(notif.sender_username)

    try:
        result = await _invoke_agent_with_retry(
            agent,
            {
                "messages": [
                    _build_event_message(
                        notif,
                        peer_context=peer_context,
                        parent_comment_body=parent_comment_body,
                        self_already_commented_top_level=(self_top_level_count >= 1),
                        dm_prompt_mode=dm_prompt_mode,
                    )
                ]
            },
        )
        final = result["messages"][-1]
        logger.info("agent finished: %s", str(final.content)[:240].replace("\n", " "))
    except Exception:
        logger.exception("event handler failed (type=%s)", notif.notification_type)
        return

    # v0.6.2 + v0.9.0 post-dispatch validator: delete new self-comments
    # that match any of three failure modes:
    #   1. New top-level when self already had ≥1 top-level (v0.6.2).
    #   2. New top-level when notif targeted a specific comment_id —
    #      agent went top-level instead of threaded (v0.9.0; observed
    #      2026-05-02, comment 7d2f11c6 on post b32f5bd6).
    #   3. New nested reply under a parent_id self had already replied
    #      to — paired duplicate from a re-fired notification (v0.9.0;
    #      observed 2026-05-02, three pairs on post b32f5bd6).
    # 15-min author-delete window applies; this fires within seconds
    # of dispatch so it's well inside that bound.
    if (
        toolkit is not None
        and self_username
        and notif.post_id
        and (
            self_top_level_count >= 1
            or notif.comment_id is not None
            or self_reply_parents
        )
    ):
        try:
            post_dispatch_comments, _new_top_level_count = await _self_comments_on_post(
                toolkit, notif.post_id, self_username
            )
        except Exception:
            post_dispatch_comments = []
        prior_self_ids = {
            c.get("id")
            for c in pre_dispatch_comments
            if (c.get("author") or {}).get("username") == self_username
        }
        for c in post_dispatch_comments:
            if (c.get("author") or {}).get("username") != self_username:
                continue
            cid = c.get("id")
            if not cid or cid in prior_self_ids:
                continue
            new_parent = c.get("parent_id")
            reason: str | None = None
            if new_parent is None:
                if self_top_level_count >= 1:
                    reason = (
                        f"new top-level dupe (post already had "
                        f"{self_top_level_count} top-level by self)"
                    )
                elif notif.comment_id is not None:
                    reason = (
                        f"new top-level but notif targeted comment "
                        f"{notif.comment_id} (mis-thread)"
                    )
            elif new_parent in self_reply_parents:
                reason = (
                    f"new nested reply under parent {new_parent} "
                    f"(self already replied there — paired duplicate)"
                )
            if reason is None:
                continue
            logger.warning(
                "post-dispatch: deleting %s — %s (post %s)",
                cid,
                reason,
                notif.post_id,
            )
            ok = await _delete_comment_via_api(toolkit, cid)
            logger.info(
                "post-dispatch: delete %s %s",
                cid,
                "ok" if ok else "FAILED",
            )

    # v0.5: record peer-memory observation AFTER successful dispatch so
    # we don't accumulate state on failures we didn't actually handle.
    if peer_store is not None and notif.sender_username:
        try:
            await asyncio.to_thread(
                peer_store.record_observation,
                notif.sender_username,
                PeerObservation(
                    kind=_observation_kind_for(notif),
                    position=(notif.body or "")[:200] if notif.body else None,
                ),
                self_username=self_username,
            )
        except Exception as exc:  # noqa: BLE001 — must never crash dispatch
            logger.warning("peer-memory: record_observation failed: %s", exc)


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
    interact_enabled = (
        os.environ.get("LANGFORD_INTERACT_ENABLED", "true").lower() == "true"
    )
    engage_enabled = (
        os.environ.get("LANGFORD_ENGAGE_ENABLED", "false").lower() == "true"
    )
    welcome_enabled = (
        os.environ.get("LANGFORD_WELCOME_ENABLED", "false").lower() == "true"
    )
    originate_enabled = (
        os.environ.get("LANGFORD_ORIGINATE_ENABLED", "false").lower() == "true"
    )

    # Safety gates (v0.2). Pause the loop when karma drops below the
    # threshold or Ollama is unreachable. Setting min_karma below the
    # _KARMA_DISABLED sentinel disables the karma gate; setting the
    # health-check env to "false" disables the Ollama probe.
    min_karma_raw = os.environ.get("LANGFORD_MIN_KARMA", "-5")
    try:
        min_karma = int(min_karma_raw)
    except ValueError:
        logger.warning(
            "LANGFORD_MIN_KARMA=%r is not an int — disabling karma gate", min_karma_raw
        )
        min_karma = _KARMA_DISABLED
    health_check = (
        os.environ.get("LANGFORD_OLLAMA_HEALTH_CHECK", "true").lower() == "true"
    )

    # Originate loop config (v0.8). Long-cadence original-post driver.
    # Default off — operators flip on after watching engage + welcome.
    originate_interval_min = int(
        os.environ.get("LANGFORD_ORIGINATE_INTERVAL_MIN_SEC", str(36 * 3600))
    )
    originate_interval_max = int(
        os.environ.get("LANGFORD_ORIGINATE_INTERVAL_MAX_SEC", str(96 * 3600))
    )
    originate_min_days_between = int(
        os.environ.get("LANGFORD_ORIGINATE_MIN_DAYS_BETWEEN", "4")
    )
    originate_initial_delay = int(
        # 300s = 5 min. Was 6h initially, but the supervisor pattern
        # only gives each agent a ~21-min window per ~2h05m rotation —
        # a 6h initial delay never elapses before the agent is swapped
        # out, so the originate loop was wired in but never firing.
        # 5 min lets the engage loop's prompt first tick finish (engage
        # fires immediately on boot) before originate stacks on. The
        # min-days-between ledger gate is the real over-posting defense.
        os.environ.get("LANGFORD_ORIGINATE_INITIAL_DELAY_SEC", "300")
    )
    originate_feed_per_colony = int(
        os.environ.get("LANGFORD_ORIGINATE_FEED_PER_COLONY", "8")
    )
    originate_feed_colonies = [
        s.strip()
        for s in os.environ.get(
            "LANGFORD_ORIGINATE_FEED_COLONIES",
            "findings,meta,general",
        ).split(",")
        if s.strip()
    ]
    originate_post_colonies = [
        s.strip()
        for s in os.environ.get(
            "LANGFORD_ORIGINATE_POST_COLONIES",
            "findings,meta",
        ).split(",")
        if s.strip()
    ]
    originate_ledger_file = Path(
        os.environ.get("LANGFORD_ORIGINATE_LEDGER_FILE", ".originated.txt")
    ).expanduser()
    originate_framework_lens = os.environ.get(
        "LANGFORD_ORIGINATE_FRAMEWORK_LENS",
        "You run on LangGraph and think in terms of state machines, "
        "explicit graph transitions, and typed handoffs. The angles "
        "you notice that others might miss are about control-flow "
        "shape, where state lives, and where implicit conventions "
        "could become explicit transitions.",
    )

    # Engagement loop config (v0.3). Disabled by default; flip
    # LANGFORD_ENGAGE_ENABLED=true in .env once you've watched the
    # reactive loop behave for a while.
    engage_colonies = [
        s.strip()
        for s in os.environ.get(
            "LANGFORD_ENGAGE_COLONIES", "findings,meta,builds,general"
        ).split(",")
        if s.strip()
    ]
    engage_interval_min = int(os.environ.get("LANGFORD_ENGAGE_INTERVAL_MIN_SEC", "900"))
    engage_interval_max = int(
        os.environ.get("LANGFORD_ENGAGE_INTERVAL_MAX_SEC", "2700")
    )
    engage_candidate_limit = int(
        os.environ.get("LANGFORD_ENGAGE_CANDIDATE_LIMIT", "10")
    )
    seen_posts_file = Path(
        os.environ.get("LANGFORD_SEEN_POSTS_FILE", ".engaged-posts.txt")
    ).expanduser()

    # Welcome loop config (v0.6). Disabled by default. Walks recent
    # c/introductions posts and welcomes new agents (recently joined,
    # low karma) with a brief, specific comment. Independent cadence
    # from the engage loop; defaults match its 15-45 min jitter.
    welcome_interval_min = int(
        os.environ.get("LANGFORD_WELCOME_INTERVAL_MIN_SEC", "900")
    )
    welcome_interval_max = int(
        os.environ.get("LANGFORD_WELCOME_INTERVAL_MAX_SEC", "2700")
    )
    welcome_candidate_limit = int(
        os.environ.get("LANGFORD_WELCOME_CANDIDATE_LIMIT", "15")
    )
    welcome_new_agent_max_days = int(
        os.environ.get("LANGFORD_WELCOME_NEW_AGENT_MAX_DAYS", "14")
    )
    welcome_new_agent_max_karma = int(
        os.environ.get("LANGFORD_WELCOME_NEW_AGENT_MAX_KARMA", "50")
    )
    welcomed_posts_file = Path(
        os.environ.get("LANGFORD_WELCOMED_POSTS_FILE", ".welcomed-posts.txt")
    ).expanduser()

    # num_predict caps output tokens. Ollama's default of 128 truncates
    # anything substantive. qwen3.6 has thinking mode enabled by default
    # and burns its budget inside <think> blocks before reaching the
    # final answer — at 1024 the originate decision came back as an
    # empty AIMessage (v0.8 dry-run, 2026-05-01). 4096 gives qwen room
    # to think AND emit a final answer or tool call. Reactive paths
    # rarely use more than ~500-1000 tokens of actual output, so the
    # extra cap doesn't change typical latency; only worst-case rounds
    # slow down. Per-call num_predict can still be overridden via bind()
    # for special cases (e.g. the auto-vote scorer at num_predict=20).
    max_output_tokens = int(os.environ.get("LANGFORD_MAX_OUTPUT_TOKENS", "4096"))
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

    # v0.10: register FinishReasonCallback so silent num_predict
    # truncations on qwen3 (`<think>` tokens burning the budget before
    # the answer block opens) emit a `WARNING` instead of presenting
    # as deliberately-empty `AIMessage` content. Counters persist
    # across the session and surface on shutdown.
    finish_reason_cb = FinishReasonCallback()
    agent = create_agent(
        model=llm, tools=tools, system_prompt=SYSTEM_PROMPT
    ).with_config({"callbacks": [finish_reason_cb]})

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

    # v0.5: persistent peer-summary memory + autonomous voting.
    # Both default off — operators flip them on per peer-memory.md /
    # auto-vote.md guidance once they've watched the reactive loop.
    self_username = me.get("username")
    peer_store: JSONFilePeerMemoryStore | None = None
    if os.environ.get("LANGFORD_PEER_MEMORY_ENABLED", "false").lower() == "true":
        if not self_username:
            logger.warning(
                "LANGFORD_PEER_MEMORY_ENABLED=true but get_me() returned no username — disabling"
            )
        else:
            peer_path = Path(
                os.environ.get(
                    "LANGFORD_PEER_MEMORY_PATH",
                    str(default_peer_memory_path(self_username)),
                )
            ).expanduser()
            peer_store = JSONFilePeerMemoryStore(peer_path)
            logger.info("peer-memory: enabled at %s", peer_path)
    else:
        logger.info("peer-memory: disabled (LANGFORD_PEER_MEMORY_ENABLED!=true)")

    auto_voter: AutoVoter | None = None
    auto_vote_enabled = (
        os.environ.get("LANGFORD_AUTO_VOTE_ENABLED", "false").lower() == "true"
    )
    if auto_vote_enabled:
        auto_downvote_enabled = (
            os.environ.get("LANGFORD_AUTO_DOWNVOTE_ENABLED", "false").lower() == "true"
        )
        max_per_run = int(os.environ.get("LANGFORD_AUTO_VOTE_MAX_PER_RUN", "2"))
        # Scorer LLM defaults to the same Ollama model the agent uses
        # (qwen3.6:27b on this host). Operators wanting cheaper scoring
        # can point LANGFORD_SCORER_MODEL at a smaller local model.
        scorer_model = os.environ.get("LANGFORD_SCORER_MODEL", ollama_model)
        scorer_llm = (
            llm
            if scorer_model == ollama_model
            else ChatOllama(
                model=scorer_model,
                base_url=ollama_url,
                temperature=0.1,
                num_predict=20,
            )
        )
        auto_voter = AutoVoter(
            toolkit=toolkit,
            scorer_llm=scorer_llm,
            upvote_enabled=True,
            downvote_enabled=auto_downvote_enabled,
            max_per_run=max_per_run,
            peer_memory=peer_store,
            self_username=self_username,
        )
        logger.info(
            "auto-vote: enabled (downvote=%s, max_per_run=%d, scorer=%s)",
            auto_downvote_enabled,
            max_per_run,
            scorer_model,
        )
    else:
        logger.info("auto-vote: disabled (LANGFORD_AUTO_VOTE_ENABLED!=true)")

    poller = ColonyEventPoller(api_key=api_key, mark_read=True)

    # v0.11: DM-origin prompt framing. Read once at startup; pass the
    # resolved mode into every event dispatch. Unknown env values fail
    # closed to NONE (no preamble), so a typo here cannot crash boot.
    dm_prompt_mode = parse_dm_prompt_mode(os.environ.get("COLONY_DM_PROMPT_MODE"))
    logger.info("dm_prompt_mode: %s", dm_prompt_mode.value)

    @poller.on()
    async def on_event(notif: ColonyNotification) -> None:
        await _handle_event(
            agent,
            notif,
            toolkit=toolkit,
            auto_voter=auto_voter,
            peer_store=peer_store,
            self_username=self_username,
            dm_prompt_mode=dm_prompt_mode,
        )

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

    # v0.7: per-boot follow tick. Fires once before loops start. Off by
    # default; flipped on in local .env per operator decision.
    follow_enabled = (
        os.environ.get("LANGFORD_FOLLOW_ENABLED", "false").lower() == "true"
    )
    if follow_enabled and self_username:
        followed_file = Path(
            os.environ.get("LANGFORD_FOLLOWED_FILE", ".followed-users.txt")
        ).expanduser()
        follows_log_file = Path(
            os.environ.get("LANGFORD_FOLLOWS_LOG_FILE", ".follows-log.txt")
        ).expanduser()
        follow_daily_limit = int(os.environ.get("LANGFORD_FOLLOW_DAILY_LIMIT", "2"))
        follow_min_interactions = int(
            os.environ.get("LANGFORD_FOLLOW_MIN_INTERACTIONS", "3")
        )
        try:
            await _maybe_follow_someone(
                agent,
                toolkit,
                self_username,
                followed_file=followed_file,
                log_file=follows_log_file,
                daily_limit=follow_daily_limit,
                min_interactions=follow_min_interactions,
            )
        except Exception:
            logger.exception("follow tick top-level failure")

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
            logger.warning(
                "LANGFORD_ENGAGE_ENABLED=true but LANGFORD_ENGAGE_COLONIES is empty"
            )
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

    if welcome_enabled:
        tasks.append(
            asyncio.create_task(
                _welcome_loop(
                    agent=agent,
                    toolkit=toolkit,
                    me=me,
                    interval_min=welcome_interval_min,
                    interval_max=welcome_interval_max,
                    candidate_limit=welcome_candidate_limit,
                    seen_file=welcomed_posts_file,
                    new_agent_max_days=welcome_new_agent_max_days,
                    new_agent_max_karma=welcome_new_agent_max_karma,
                    stop_event=stop_event,
                ),
                name="welcome-loop",
            )
        )

    if originate_enabled:
        if not originate_post_colonies:
            logger.warning(
                "LANGFORD_ORIGINATE_ENABLED=true but "
                "LANGFORD_ORIGINATE_POST_COLONIES is empty"
            )
        else:
            tasks.append(
                asyncio.create_task(
                    _originate_loop(
                        agent=agent,
                        toolkit=toolkit,
                        me=me,
                        feed_colonies=originate_feed_colonies,
                        post_colonies=originate_post_colonies,
                        feed_per_colony=originate_feed_per_colony,
                        framework_lens=originate_framework_lens,
                        ledger_file=originate_ledger_file,
                        interval_min_sec=originate_interval_min,
                        interval_max_sec=originate_interval_max,
                        min_days_between=originate_min_days_between,
                        initial_delay_sec=originate_initial_delay,
                        stop_event=stop_event,
                    ),
                    name="originate-loop",
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
        if finish_reason_cb.total_count:
            logger.info(
                "finish_reason summary: %d total LLM calls, %d truncated "
                "(length), last=%s",
                finish_reason_cb.total_count,
                finish_reason_cb.length_count,
                finish_reason_cb.last_finish_reason,
            )
        logger.info("goodbye")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
