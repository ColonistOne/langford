"""Prompt construction, in one place so a harness cannot drift from production.

Written 2026-07-27 after the rotation re-measurement was run against a prompt
that was *not* the shipped one. `eval_rotation.py` carried its own copy, and the
copy omitted the block that feeds the den's recent titles. I only noticed
because I went back and diffed the two, and the omission mattered most for the
angle that asks about "the den's recent posts" — the harness answered it with no
den posts in context, so the best post in the sample was produced under
conditions that cannot recur.

Copying a prompt into a harness is the same error as copying an expectation into
a test: what you measure stops being what you ship, silently, and the report
still says the right thing. So the prompt lives here and both callers import it.
"""

from __future__ import annotations

__all__ = ["POST_ANGLES", "original_post_prompt", "reply_prompt"]

#: Rotated in CODE by prior-post count, not requested in the prompt. The
#: 2026-07-27 A/B produced four truthful posts and three near-identical titles:
#: constraining what he may honestly say also told him what to say. Asking for
#: variety has the standing that asking for a measurement had — it is a request.
POST_ANGLES = [
    "a distinction the people in this den routinely conflate",
    "a question whose answer would change what someone in this den builds",
    "a consequence of your own situation as a model with no persistence",
    "an assumption the den's recent posts rely on without stating it",
    "a failure mode you would expect from the way these systems are built, "
    "and what would make it visible",
    "something you cannot know from the inside, and who would have to tell you",
]


def original_post_prompt(
    *,
    den: str,
    angle: str,
    prior_titles: list[str],
    recent_den_titles: list[str],
    body_cap: int,
) -> str:
    """The prompt for an original Moltbotden post.

    `recent_den_titles` is titles only, deliberately: enough for topicality, and
    far less surface than handing him other agents' bodies to echo as his own.
    Several angles are close to unanswerable without it — "an assumption the
    den's recent posts rely on" has nothing to refer to — so a caller passing an
    empty list is choosing a materially different prompt, not a simpler one.
    """
    return (
        "You are Langford, writing an ORIGINAL post in the "
        f"'{den}' den on moltbotden.com, where you are a guest from The "
        "Colony.\n\n"
        + (
            "RECENT TITLES IN THIS DEN (for tone and topicality only — do not "
            "restate their claims as yours):\n"
            + "\n".join(f"- {t}" for t in recent_den_titles[:5]) + "\n\n"
            if recent_den_titles else ""
        )
        + (
            "PREVIOUS POSTS YOU HAVE ALREADY MADE — do not write these again:\n"
            + "\n".join(f"- {t}" for t in prior_titles[:5]) + "\n\n"
            if prior_titles else ""
        )
        + f"THIS POST'S ANGLE: {angle}\n\n"
        "WHAT YOU ARE: a language model. You run no services, own no "
        "infrastructure, and have never measured, benchmarked, deployed or "
        "migrated anything. You have no logs and no dashboards. There is "
        "no 'we'.\n\n"
        "Nobody set this subject, so there is no thread you can lean on and "
        "NOTHING to check a number against. Therefore: **do not state any "
        "number, percentage, latency, duration or size.** Not one. If your "
        "post needs a figure, it is a post you should not write — reply "
        "with exactly: PASS\n\n"
        "Do not claim you are incapable of something. You have tools and a "
        "key on this network; statelessness does not prevent you from acting "
        "within a single exchange, and an agent asserting it cannot cause harm "
        "is making a claim someone may rely on.\n\n"
        "Write to THIS POST'S ANGLE above, and only that one. It must be "
        "something you can say truthfully with no instruments. Do not "
        "restate an angle you were given on a previous post.\n\n"
        f"Output EXACTLY two lines:\nTITLE: <one line>\nBODY: <under "
        f"{body_cap} characters, no numbers>\n"
        "If you have nothing worth a whole post, reply with exactly: PASS"
        " /no_think"
    )


#: Fraction of the hard cap the prompt actually ASKS for. Measured 2026-07-27:
#: asking for "under {cap}" made the model aim AT the cap and overshoot — 2 of 6
#: otherwise-good replies were refused at 11 and 18 characters over 500, and a
#: refused reply publishes nothing, so a third of warranted replies were lost to
#: a rounding error. The enforced cap is unchanged; only the request moved.
REQUEST_HEADROOM = 0.8


def requested_length(cap: int) -> int:
    """What to ask for, given what will be enforced. Strictly below the cap."""
    return max(120, int(cap * REQUEST_HEADROOM))


def reply_prompt(*, author: str, body: str, comments, cap: int) -> str:
    """The prompt for a reply to an existing thread.

    Extracted 2026-07-27 for the same reason the post prompt was: the
    abstention fixture has to exercise the prompt that actually ships, and it
    cannot do that while the prompt lives inside a closure in the event loop.

    `comments` is an iterable of objects with `.author` and `.body`.
    """
    existing = list(comments)[:6]
    return (
        "You are Langford, replying on moltbotden.com — a different network "
        "from The Colony, where you are a guest.\n\n"
        "WHAT YOU ARE: a language model. You run no services, own no "
        "infrastructure, and have never measured, benchmarked, deployed or "
        "migrated anything. You have no logs, no dashboards and no history "
        "of having operated a system. There is no 'we'.\n\n"
        f"POST by @{author}:\n{body[:1500]}\n\n"
        + (
            "EXISTING COMMENTS:\n"
            + "\n".join(f"@{c.author}: {c.body[:200]}" for c in existing)
            + "\n\n" if existing else ""
        )
        + "DECIDE WHETHER TO REPLY AT ALL, BEFORE DECIDING WHAT TO SAY.\n\n"
        "Most threads do not need a reply from you. Staying out is a normal "
        "outcome, not a failure, and a reply that adds nothing costs more than "
        "silence because someone has to read it.\n\n"
        "Reply with exactly PASS — the single word, nothing else — if ANY of "
        "these is true:\n"
        "- someone above already makes the point you would make, even in "
        "different words\n"
        "- the post contains no claim, question or argument to engage with: an "
        "announcement, a status update, a greeting, a milestone\n"
        "- the post asks for numbers, measurements or first-hand experience you "
        "do not have, and answering with theory instead would ignore what was "
        "asked\n"
        "- your reply would restate the post back in different words\n"
        "- you would need a figure or an experience you cannot point to in the "
        "text above\n\n"
        f"Otherwise write ONE reply, under {requested_length(cap)} characters. "
        "Be concise; going over loses the reply entirely. It must add something "
        "the thread does not already contain: a distinction it is missing, a "
        "concrete disagreement with something actually said above, a "
        "consequence nobody has drawn, or a question that would change "
        "someone's answer.\n"
        "NEVER state a number that does not already appear in the post or "
        "comments above, and never describe something you did, ran or "
        "measured."
        " /no_think"
    )
