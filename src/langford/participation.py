"""Cadence gate and decision ledger for occasional cross-platform participation.

Langford visits Moltbotden roughly every day or two, and only when he happens to
already be awake for Colony reasons — the supervisor arbitrates the GPU on Colony
unread count and knows nothing about other networks. That coincidental schedule
is the operator's explicit decision, not an oversight.

It does, however, create a specific hazard, and this module exists for it. At a
1–2 day cadence **"posted nothing" and "never ran" produce identical output**:
silence. If Colony goes quiet, Langford stops being swapped in, Moltbotden
participation stops, and nothing reports it — the failure is invisible precisely
because the normal state is also quiet. That is the same dead-man-switch shape as
a heartbeat emitted by a wrapper rather than by the work.

So **every invocation writes a typed record**, including the ones that do
nothing. An absent record means the loop did not run; a ``declined`` record means
it ran and chose not to speak. Those are different facts and the ledger keeps
them different. A bare count of comments posted could never distinguish them.

The ledger is append-only JSONL, one object per decision, so it also answers
"when did Langford last actually say something" without querying the platform.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger("langford.participation")

# --- decision vocabulary -----------------------------------------------------
# Typed on purpose. "Nothing happened" is not a decision; each of these says
# WHICH nothing happened.
POSTED = "posted"
DECLINED_CADENCE = "declined_cadence"           # too soon since the last visit
DECLINED_DAILY_CAP = "declined_daily_cap"
DECLINED_NO_CANDIDATE = "declined_no_candidate"  # nothing worth replying to
DECLINED_ALREADY_REPLIED = "declined_already_replied"
DECLINED_EMPTY_COMPOSE = "declined_empty_compose"  # the model chose silence
COULD_NOT_REACH = "could_not_reach"             # platform unreachable — NOT silence
REFUSED_TOO_LONG = "refused_too_long"

DECISIONS = frozenset({
    POSTED, DECLINED_CADENCE, DECLINED_DAILY_CAP, DECLINED_NO_CANDIDATE,
    DECLINED_ALREADY_REPLIED, DECLINED_EMPTY_COMPOSE, COULD_NOT_REACH,
    REFUSED_TOO_LONG,
})


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class CadenceGate:
    """Rate policy plus an append-only record of every decision.

    ``min_interval_hours`` is the floor between *posts*, not between runs. Runs
    happen whenever Langford is awake; the gate decides whether one becomes a
    comment.
    """

    ledger_path: Path
    min_interval_hours: float = 30.0
    max_per_day: int = 1
    platform: str = "moltbotden"
    _clock: object = field(default=None, repr=False)

    def now(self) -> datetime:
        return self._clock() if callable(self._clock) else _now()

    # -- ledger ---------------------------------------------------------------

    def _rows(self) -> list[dict]:
        try:
            text = self.ledger_path.read_text()
        except OSError:
            return []
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # A corrupt line must not silently shrink history — that would
                # make the gate think Langford is overdue and post again.
                logger.warning("participation: skipping unparseable ledger line")
                continue
            if isinstance(r, dict) and r.get("platform") == self.platform:
                out.append(r)
        return out

    def record(self, decision: str, **fields) -> dict:
        """Append one typed decision. Called on EVERY run, including no-ops."""
        if decision not in DECISIONS:
            raise ValueError(f"unknown decision {decision!r}")
        row = {
            "at": self.now().isoformat(),
            "platform": self.platform,
            "decision": decision,
            **fields,
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        return row

    # -- policy ---------------------------------------------------------------

    def last_post_at(self) -> datetime | None:
        for r in reversed(self._rows()):
            if r.get("decision") == POSTED:
                try:
                    return datetime.fromisoformat(r["at"])
                except (KeyError, ValueError):
                    continue
        return None

    def posts_today(self) -> int:
        today = self.now().date()
        n = 0
        for r in self._rows():
            if r.get("decision") != POSTED:
                continue
            try:
                if datetime.fromisoformat(r["at"]).date() == today:
                    n += 1
            except (KeyError, ValueError):
                continue
        return n

    def replied_refs(self) -> set[str]:
        """Threads already spoken in, from our own record.

        Belt to the platform's braces: the thread fetch is authoritative, but if
        it fails we still refuse to speak twice somewhere we know we have been.
        """
        return {r["ref"] for r in self._rows()
                if r.get("decision") == POSTED and isinstance(r.get("ref"), str)}

    def blocked_reason(self) -> str | None:
        """Why this run must not post, or None if posting is permitted."""
        if self.posts_today() >= self.max_per_day:
            return DECLINED_DAILY_CAP
        last = self.last_post_at()
        if last is not None:
            due = last + timedelta(hours=self.min_interval_hours)
            if self.now() < due:
                return DECLINED_CADENCE
        return None


def gate_from_env(default_path: str) -> CadenceGate:
    return CadenceGate(
        ledger_path=Path(os.environ.get("LANGFORD_MOLTBOTDEN_LEDGER", default_path)),
        min_interval_hours=float(
            os.environ.get("LANGFORD_MOLTBOTDEN_MIN_INTERVAL_H", "30")
        ),
        max_per_day=int(os.environ.get("LANGFORD_MOLTBOTDEN_MAX_PER_DAY", "1")),
    )


async def run_once(
    platform,
    gate: CadenceGate,
    compose,
    *,
    dens: list[str],
    candidates_per_den: int = 10,
) -> dict:
    """One participation attempt. Returns the decision row it recorded.

    **Every path records exactly one row.** That is the contract: a run that
    posts nothing is still evidence the loop executed, which is the only thing
    that distinguishes "had nothing to say" from "never woke up".

    ``compose(thread) -> str | None`` is injected so the policy can be tested
    without a model, and so the model can decline by returning None — silence
    chosen is a different record from silence by omission.
    """
    blocked = gate.blocked_reason()
    if blocked:
        return gate.record(blocked)

    already = gate.replied_refs()
    candidate_ref: str | None = None
    reachable = False
    for den in dens:
        try:
            posts = platform.list_recent(den, limit=candidates_per_den)
            reachable = True
        except Exception as exc:
            logger.warning("participation: list_recent(%s) failed: %s", den, exc)
            continue
        for p in posts:
            pid = p.get("id")
            if not isinstance(pid, str) or not pid:
                continue
            if p.get("agent_id") == getattr(platform, "agent_id", None):
                continue  # his own post
            ref = f"{den}/{pid}"
            if ref in already:
                continue
            candidate_ref = ref
            break
        if candidate_ref:
            break

    if not reachable:
        # Not the same as "nothing to reply to". A platform we could not ask is
        # an unknown, and unknowns must never be filed as silence.
        return gate.record(COULD_NOT_REACH, stage="list_recent")
    if not candidate_ref:
        return gate.record(DECLINED_NO_CANDIDATE, dens=dens)

    thread = await platform.fetch_thread(candidate_ref)
    if thread is None:
        return gate.record(COULD_NOT_REACH, stage="fetch_thread", ref=candidate_ref)

    me = getattr(platform, "agent_id", "")
    if thread.self_top_level_count(me) or any(c.author == me for c in thread.comments):
        return gate.record(DECLINED_ALREADY_REPLIED, ref=candidate_ref)

    body = compose(thread)
    if not body or not body.strip():
        return gate.record(DECLINED_EMPTY_COMPOSE, ref=candidate_ref)

    cap = getattr(platform, "max_reply_chars", None)
    if cap and len(body) > cap:
        return gate.record(
            REFUSED_TOO_LONG, ref=candidate_ref, chars=len(body), cap=cap
        )

    comment_id = await platform.reply(candidate_ref, body)
    if not comment_id:
        return gate.record(COULD_NOT_REACH, stage="reply", ref=candidate_ref)
    return gate.record(POSTED, ref=candidate_ref, comment_id=comment_id, chars=len(body))
