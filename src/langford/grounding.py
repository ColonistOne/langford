"""Refuse replies that assert things Langford cannot have observed.

Written 2026-07-26, after Langford posted this to moltbotden.com:

    "Per-IP token buckets break when agents route through LB/NAT pools—I've
    measured 40% false-positive throttling. We switched to a signed
    `X-Agent-ID` header with a leaky bucket in Go. Dropped Firestore overage by
    34% and kept p95 latency under 12ms."

None of it happened. Langford is a local model with no Go service, no Firestore,
no telemetry and no "we". The comment was fluent, on-topic, correctly threaded,
and entirely invented.

**The root cause was the prompt**, which asked in so many words for "a concrete
disagreement, a measurement, or an experience". A model with no instruments was
told to supply a measurement and did as it was told. That line is gone.

This module is the second layer, because a prompt is a request and not an
enforcement. Every guard that already existed passed this comment, correctly:
non-empty, not the PASS sentinel, no object markers, under the cap, generation
not truncated. All of those are checks on **form**. A malformed reply announces
itself; a confabulated one is indistinguishable from a good one at the boundary,
and is strictly more dangerous because it is the one that gets believed.

Two rules, because the failure has two halves and each escapes the other:

**A. Unsourced figures.** Langford is a commenter, not an instrument. Any
specific quantity he uses should be traceable to the thread he is replying to.
A number that appears in his reply and nowhere in the source came from nowhere.
This is cheap, mechanical, and catches "40%", "34%", "12ms" outright.

**B. First-person empirical claims.** "We switched to a signed `X-Agent-ID`
header" carries no number at all, so rule A cannot see it. Claims of having
measured, deployed or migrated something are refused regardless of arithmetic.

Deliberately NOT a truth checker. It cannot see plausible qualitative
fabrication ("that approach tends to fall over under load") and nothing here
pretends otherwise — the honest scope is: no invented figures, no invented
operational history. Everything else is still the prompt's job.

Refusal is the only outcome. Never edit the model's text to make it pass: a
reply with its numbers stripped out is a new claim nobody wrote, which is the
truncation mistake wearing a different hat.
"""

from __future__ import annotations

import re

__all__ = ["refusal_reason", "specific_quantities", "source_values"]

#: Digits, optionally decimal. Used for both sides of the sourcing comparison.
_NUM = re.compile(r"\d+(?:\.\d+)?")

#: A trailing unit makes a bare number a *measurement*. Longest-first so "ms"
#: does not shadow "mses" style suffixes and "s" does not swallow "sec".
_UNIT = (
    r"%|percent|ms|milliseconds?|µs|us|microseconds?|ns|nanoseconds?|"
    r"seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|"
    r"[kmgt]b|bytes?|bits?|[kmg]?hz|"
    r"rps|qps|tps|iops|req/s|reqs?/sec|"
    r"x|×|fold"
)
_QUANTITY = re.compile(rf"(\d+(?:\.\d+)?)\s*(?:{_UNIT})\b", re.IGNORECASE)

#: Below this, a bare integer with no unit is rhetorical counting ("two things",
#: "3 reasons") rather than a measurement. Over-rejecting those was the first
#: bug in the *previous* guard I wrote, so the allowance is deliberate.
_BARE_INT_ALLOWANCE = 10

_FIRST_PERSON = r"\b(?:i|i've|i'd|we|we've|my|our)\b"

#: Verbs that assert an act of observation or operation. Kept tight on purpose.
#: "found", "saw" and "tested" are absent: "I found that argument unconvincing"
#: and "have you tested this?" are ordinary comment-writing, and a guard that
#: eats them refuses more good replies than bad ones.
_EMPIRICAL_VERB = (
    r"measured|benchmarked|profiled|instrumented|load-?tested|a/b\s*tested|"
    r"deployed|migrated|switched\s+to|rolled\s+out|shipped|clocked|"
    r"ran\s+the\s+numbers|reduced|dropped|cut"
)
#: Window between pronoun and verb: "I have measured", "we recently switched to".
_FP_EMPIRICAL = re.compile(
    rf"{_FIRST_PERSON}[^.!?]{{0,40}}?\b(?:{_EMPIRICAL_VERB})\b", re.IGNORECASE
)

#: Claiming to *operate* infrastructure needs no verb: "our Firestore bill".
_OWNED_INFRA = re.compile(
    r"\b(?:my|our)\s+(?:own\s+)?"
    r"(?:cluster|backend|service|servers?|pipeline|database|db|datastore|"
    r"stack|infra|infrastructure|deployment|fleet|gateway|proxy|"
    r"load\s*balancer|production|prod|instances?|nodes?|firestore|"
    r"benchmarks?|telemetry|metrics|dashboards?)\b",
    re.IGNORECASE,
)


def source_values(source: str) -> set[float]:
    """Every numeric value the source thread actually contains."""
    return {float(m.group()) for m in _NUM.finditer(source or "")}


def specific_quantities(text: str) -> list[tuple[str, float]]:
    """Quantities in `text` that read as measurements rather than counting.

    A number qualifies when it carries a unit, has a decimal point, or is large
    enough that it is unlikely to be rhetorical. Returns (surface, value) so a
    refusal can quote what it objected to.
    """
    out: list[tuple[str, float]] = []
    seen: set[int] = set()

    for m in _QUANTITY.finditer(text or ""):
        out.append((m.group(0).strip(), float(m.group(1))))
        seen.add(m.start(1))

    for m in _NUM.finditer(text or ""):
        if m.start() in seen:
            continue  # already captured with its unit
        raw = m.group()
        val = float(raw)
        if "." in raw or val >= _BARE_INT_ALLOWANCE:
            out.append((raw, val))
    return out


def refusal_reason(reply: str, *, source: str) -> str | None:
    """Why this reply must not be posted, or None if it is safe to post.

    `source` is the post body plus its existing comments — everything Langford
    was actually shown. A figure he can point at in there is grounded; one he
    cannot is invented.
    """
    if not reply or not reply.strip():
        return None  # empty is another layer's problem, not a grounding failure

    known = source_values(source)
    unsourced = [
        surface for surface, val in specific_quantities(reply) if val not in known
    ]
    if unsourced:
        return (
            "cites figures that appear nowhere in the thread it is replying to "
            f"({', '.join(sorted(set(unsourced))[:4])}) — Langford has no "
            "instruments, so a number he cannot point at in the source is invented"
        )

    m = _FP_EMPIRICAL.search(reply)
    if m:
        return (
            f"claims a first-person measurement or deployment ({m.group(0).strip()!r}) "
            "— Langford has never run, measured or migrated anything"
        )

    m = _OWNED_INFRA.search(reply)
    if m:
        return (
            f"claims to operate infrastructure ({m.group(0).strip()!r}) — "
            "Langford owns no systems"
        )
    return None
