"""Refuse a post that repeats one Langford has already made.

Written 2026-07-27, after the original-post A/B came back safe and boring. All
four generations were truthful and three of four carried effectively the same
title — "The Illusion of Continuity in Stateless Inference", "…in Stateless
Exchange", "The Illusion of Continuity". The prompt had bought truthfulness by
narrowing him to one subject: telling a model what it *can* honestly write about
turns out to tell it what to write about.

At one post per 72 hours that is not cosmetic. It is an agent that visibly says
the same thing forever, which is what a broken bot looks like from outside — and
it is invisible to every other guard, because each post is individually
well-formed, individually truthful, and individually fine.

**Why a guard and not a better prompt.** The collapse happened *inside* a careful
prompt. Asking more nicely for variety has exactly the standing that asking for a
measurement had: it is a request, and the thing it addresses has no obligation.

**The first measure I wrote did not work, and the only reason I know is that I
tested it against the real failure.** Word-3-shingle Jaccard scored the collapsed
posts at 0.008 — indistinguishable from unrelated posts. Shingles measure
*phrasing*; these four shared a *thesis* in different words. A guard that cannot
separate the incident that motivated it is decoration, and it would have shipped
looking reasonable.

What does separate them, measured on that output:

    content-word Jaccard on the BODY   collapsed 0.075-0.116   distinct 0.007-0.072
    content-word Jaccard on the TITLE  collapsed 0.00-0.60     distinct 0.00-0.10

So the body measure has recall and a thin margin; the title measure has a wide
margin and poor recall (rewordings score zero). Neither alone is good enough, so
both are checked and either can refuse. The thresholds sit where the measured
populations separate, not on round numbers, and the fixtures that justify them
are in tests/test_novelty.py so a future change has to argue with the data.

Stated limits, because they are real: this is lexical, not semantic. A genuine
paraphrase with fresh vocabulary passes. Two posts about the same subject from
different angles may score close to the line. It catches the failure that
actually happened, which is a narrower claim than "detects repetition".
"""

from __future__ import annotations

import re

__all__ = [
    "content_words",
    "jaccard",
    "repetition_reason",
    "BODY_THRESHOLD",
    "TITLE_THRESHOLD",
]

#: Body content-word overlap at or above this reads as the same post. Measured
#: distinct pairs topped out at 0.072 and collapsed pairs started at 0.075; the
#: threshold sits above the noise rather than in the 3-point gap, trading some
#: recall for not refusing genuinely different posts.
BODY_THRESHOLD = 0.10

#: Titles are the loud signal when they fire: collapsed titles scored 0.50-0.60
#: while no distinct pair exceeded 0.10. Wide margin, poor recall.
TITLE_THRESHOLD = 0.40

_WORD = re.compile(r"[a-z0-9']+")

#: Function words carry no subject. Kept short and boring on purpose — a long
#: hand-tuned list is a place for the threshold to hide.
_STOP = frozenset("""
the a an of in on to and or is are was were be been being it its i you we they
this that these those as at by for with from not no there have has had my our
your their what when which who how if then than so but can could will would
should may might do does did me them he she his her about into over under more
most much many any all each other some such only own same too very just now
""".split())


def content_words(text: str) -> set[str]:
    """Lower-cased words that carry subject: no stopwords, nothing under 3 chars."""
    return {w for w in _WORD.findall((text or "").lower())
            if w not in _STOP and len(w) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def repetition_reason(
    title: str,
    body: str,
    previous: list[tuple[str, str]],
    *,
    body_threshold: float = BODY_THRESHOLD,
    title_threshold: float = TITLE_THRESHOLD,
) -> str | None:
    """Why this post repeats an earlier one, or None if it is new enough.

    `previous` is his own prior (title, body) pairs. Empty means he has not
    posted — which must return None, or the first post could never be made.
    """
    if not previous or not (body or "").strip():
        return None
    tw, bw = content_words(title), content_words(body)
    for i, (ptitle, pbody) in enumerate(previous, 1):
        bs = jaccard(bw, content_words(pbody))
        ts = jaccard(tw, content_words(ptitle))
        if bs >= body_threshold or ts >= title_threshold:
            which = "body" if bs >= body_threshold else "title"
            return (
                f"repeats an earlier post ({which} content-word overlap "
                f"body={bs:.2f} title={ts:.2f} vs prior post #{i}: "
                f"{ptitle[:60]!r}) — truthful and well-formed is not enough; "
                f"saying the same thing again is its own failure"
            )
    return None
