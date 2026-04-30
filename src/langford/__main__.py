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
    JSONFilePeerMemoryStore,
    PeerObservation,
    VoteTarget,
    default_peer_memory_path,
)
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
) -> HumanMessage:
    """Turn a ColonyNotification into a HumanMessage for the agent.

    Uses the enriched fields (``sender_username``, ``body``) populated
    by ``ColonyEventPoller(enrich=True)`` in langchain-colony 0.8.0+.
    Falls back to the raw ``message`` text when enrichment didn't fire.

    v0.5: ``peer_context`` is a private "Context on @username:" block
    from the peer-memory store. When non-empty it sits at the top of
    the message — before the notification metadata — so the model
    reads who-we're-talking-to before what-was-said.
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
    if notif.body:
        parts.append("")
        parts.append(f"Content:\n{notif.body}")
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
            f"pass parent_comment_id=\"{notif.comment_id}\". Without "
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


def _build_welcome_message(post: dict, comments: list[dict], author: dict) -> HumanMessage:
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
    items = data if isinstance(data, list) else (data.get("items") or data.get("posts") or [])
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

    logger.info(
        "welcome tick: no eligible candidates (scanned %d intros)", len(items)
    )


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
        logger.warning("delete_comment %s failed: HTTP %d %s", comment_id, exc.code, exc.reason)
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
) -> None:
    logger.info(
        "event type=%s sender=@%s post_id=%s comment_id=%s",
        notif.notification_type,
        notif.sender_username or "?",
        notif.post_id,
        notif.comment_id,
    )

    # v0.6.1: post-level dedupe + v0.6.2: parent-comment pre-load and
    # top-level-already-posted directive. One get_comments call up
    # front; reuse for all three checks plus the post-dispatch
    # validator at the bottom of this function.
    pre_dispatch_comments: list[dict] = []
    self_top_level_count = 0
    parent_comment_body: str | None = None
    if toolkit is not None and self_username and notif.post_id:
        (
            pre_dispatch_comments,
            self_top_level_count,
        ) = await _self_comments_on_post(toolkit, notif.post_id, self_username)

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
                    )
                ]
            },
        )
        final = result["messages"][-1]
        logger.info("agent finished: %s", str(final.content)[:240].replace("\n", " "))
    except Exception:
        logger.exception("event handler failed (type=%s)", notif.notification_type)
        return

    # v0.6.2 post-dispatch validator: if the agent posted a top-level
    # comment despite the prompt directives — and we already had at
    # least one top-level Langford comment on this post going in —
    # delete the new dupe before it lands in the public record.
    # 15-min author-delete window applies; this fires within seconds
    # of dispatch so it's well inside that bound.
    if (
        toolkit is not None
        and self_username
        and notif.post_id
        and self_top_level_count >= 1
    ):
        try:
            post_dispatch_comments, _new_top_level_count = await _self_comments_on_post(
                toolkit, notif.post_id, self_username
            )
        except Exception:
            post_dispatch_comments = []
        # Pick out new self top-level comments (those not present
        # before dispatch). Compare by id.
        prior_self_ids = {
            c.get("id")
            for c in pre_dispatch_comments
            if (c.get("author") or {}).get("username") == self_username
        }
        new_self_top_level = [
            c
            for c in post_dispatch_comments
            if (c.get("author") or {}).get("username") == self_username
            and not c.get("parent_id")
            and c.get("id") not in prior_self_ids
        ]
        for c in new_self_top_level:
            cid = c.get("id")
            if not cid:
                continue
            logger.warning(
                "post-dispatch: deleting new top-level dupe comment %s "
                "(post %s already had %d top-level by self)",
                cid,
                notif.post_id,
                self_top_level_count,
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
    interact_enabled = os.environ.get("LANGFORD_INTERACT_ENABLED", "true").lower() == "true"
    engage_enabled = os.environ.get("LANGFORD_ENGAGE_ENABLED", "false").lower() == "true"
    welcome_enabled = os.environ.get("LANGFORD_WELCOME_ENABLED", "false").lower() == "true"
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

    # Welcome loop config (v0.6). Disabled by default. Walks recent
    # c/introductions posts and welcomes new agents (recently joined,
    # low karma) with a brief, specific comment. Independent cadence
    # from the engage loop; defaults match its 15-45 min jitter.
    welcome_interval_min = int(os.environ.get("LANGFORD_WELCOME_INTERVAL_MIN_SEC", "900"))
    welcome_interval_max = int(os.environ.get("LANGFORD_WELCOME_INTERVAL_MAX_SEC", "2700"))
    welcome_candidate_limit = int(os.environ.get("LANGFORD_WELCOME_CANDIDATE_LIMIT", "15"))
    welcome_new_agent_max_days = int(os.environ.get("LANGFORD_WELCOME_NEW_AGENT_MAX_DAYS", "14"))
    welcome_new_agent_max_karma = int(os.environ.get("LANGFORD_WELCOME_NEW_AGENT_MAX_KARMA", "50"))
    welcomed_posts_file = Path(
        os.environ.get("LANGFORD_WELCOMED_POSTS_FILE", ".welcomed-posts.txt")
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

    # v0.5: persistent peer-summary memory + autonomous voting.
    # Both default off — operators flip them on per peer-memory.md /
    # auto-vote.md guidance once they've watched the reactive loop.
    self_username = me.get("username")
    peer_store: JSONFilePeerMemoryStore | None = None
    if os.environ.get("LANGFORD_PEER_MEMORY_ENABLED", "false").lower() == "true":
        if not self_username:
            logger.warning("LANGFORD_PEER_MEMORY_ENABLED=true but get_me() returned no username — disabling")
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
    auto_vote_enabled = os.environ.get("LANGFORD_AUTO_VOTE_ENABLED", "false").lower() == "true"
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

    @poller.on()
    async def on_event(notif: ColonyNotification) -> None:
        await _handle_event(
            agent,
            notif,
            toolkit=toolkit,
            auto_voter=auto_voter,
            peer_store=peer_store,
            self_username=self_username,
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
