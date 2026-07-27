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

__all__ = ["POST_ANGLES", "original_post_prompt"]

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
