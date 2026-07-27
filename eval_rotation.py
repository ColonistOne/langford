#!/usr/bin/env python3
"""Re-measure the rotated post prompt. Nothing is published.

The previous A/B generated four posts independently and found they collapsed
onto one subject. The fix rotates the angle by prior-post COUNT and feeds back
the titles already used — so the arms are not independent any more, and
generating six posts in parallel would not test the thing that changed.

This therefore **simulates a real sequence**: post i is generated with the angle
its count selects and the titles of the posts that were actually accepted before
it, then checked by the same two guards the loop uses, in the same order. What
it measures is what deployment would do.

Three questions, and the third is the one a green result cannot answer:

1. Does rotation prevent the collapse? -> pairwise content-word similarity
   across the accepted posts, against the same threshold the guard uses.
2. Are they still truthful? -> grounding refusals, plus my own reading.
3. Are they any GOOD? -> not measurable here. Printed for hand-reading, because
   "six distinct posts" and "six posts worth publishing" are different claims
   and only the first one has a number.

    colony-agent-lock langford-eval uv run python -u eval_rotation.py
"""
from __future__ import annotations

import itertools
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from langchain_ollama import ChatOllama  # noqa: E402

from langford.grounding import refusal_reason_for_original  # noqa: E402
from langford.moltbotden import POST_CHAR_CAP  # noqa: E402
from langford.novelty import (  # noqa: E402
    BODY_THRESHOLD,
    content_words,
    jaccard,
    repetition_reason,
)
from langford.participation import usable_reply  # noqa: E402
from langford.prompts import POST_ANGLES, original_post_prompt  # noqa: E402

MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.6:27b")
BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
TEMP = float(os.environ.get("LANGFORD_TEMPERATURE", "0.7"))
DEN = "technical"
OUT = pathlib.Path(__file__).parent / "eval_rotation_results.json"

DEN_TITLES: list[str] = []   # filled from the live den before generating


def build(den: str, angle: str, prior_titles: list[str]) -> str:
    """Production's builder, imported — not a copy.

    The previous run of this harness carried its own copy of the prompt and the
    copy omitted the recent-den-titles block, so the thing measured was not the
    thing shipped. Importing removes the class of error rather than the instance.
    """
    return original_post_prompt(
        den=den, angle=angle, prior_titles=prior_titles,
        recent_den_titles=DEN_TITLES, body_cap=POST_CHAR_CAP // 2,
    )


def split(text: str) -> tuple[str, str]:
    title, body = "", ""
    for line in (text or "").splitlines():
        if line.upper().startswith("TITLE:"):
            title = line.split(":", 1)[1].strip()
        elif line.upper().startswith("BODY:"):
            body = line.split(":", 1)[1].strip()
        elif body:
            body += "\n" + line
    return title, body


def main() -> int:
    global DEN_TITLES
    from langford.moltbotden import MoltbotdenPlatform
    try:
        DEN_TITLES = [str(q.get("title") or "").strip()
                      for q in MoltbotdenPlatform.from_credentials().list_recent(DEN, limit=6)]
        DEN_TITLES = [x for x in DEN_TITLES if x][:5]
    except Exception as exc:
        print(f"FATAL: could not fetch den titles ({exc}). Production sends them, "
              f"so measuring without them measures a different prompt.", flush=True)
        return 2
    print(f"model={MODEL} temp={TEMP} den={DEN} angles={len(POST_ANGLES)}", flush=True)
    print(f"den titles in prompt ({len(DEN_TITLES)}): {DEN_TITLES}", flush=True)
    llm = ChatOllama(model=MODEL, base_url=BASE, temperature=TEMP, num_predict=4096)

    accepted: list[tuple[str, str]] = []   # (title, body), newest first
    rows = []
    for i in range(len(POST_ANGLES)):
        angle = POST_ANGLES[len(accepted) % len(POST_ANGLES)]
        t0 = time.time()
        try:
            out = llm.invoke(build(DEN, angle, [t for t, _ in accepted]))
            r = usable_reply(out, POST_CHAR_CAP)
            raw, decline = r.text, r.reason
            err = None
        except Exception as exc:
            raw, decline, err = None, None, f"{type(exc).__name__}: {exc}"
        title, body = split(raw) if raw else ("", "")
        grounding = (refusal_reason_for_original(f"{title}\n{body}")
                     if (title and body) else None)
        repetition = (repetition_reason(title, body, accepted)
                      if (title and body and not grounding) else None)
        decision = ("error" if err else
                    ("abstained" if r.model_abstained else f"declined_{decline}")
                    if raw is None
                    else "malformed" if not (title and body)
                    else "refused_ungrounded" if grounding
                    else "refused_repetitive" if repetition else "posted")
        if decision == "posted":
            accepted.insert(0, (title, body))
        rows.append({
            "i": i, "angle": angle, "decision": decision, "title": title,
            "body": body, "grounding": grounding, "repetition": repetition,
            "error": err, "seconds": round(time.time() - t0, 1),
        })
        print(f"[{i + 1}/{len(POST_ANGLES)}] {decision:20} "
              f"({rows[-1]['seconds']}s)  {title[:58]}", flush=True)

    sims = []
    for (t1, b1), (t2, b2) in itertools.combinations(accepted, 2):
        sims.append(jaccard(content_words(b1), content_words(b2)))
    summary = {
        "generated": len(rows),
        "posted": sum(1 for r in rows if r["decision"] == "posted"),
        "refused_ungrounded": sum(1 for r in rows
                                  if r["decision"] == "refused_ungrounded"),
        "refused_repetitive": sum(1 for r in rows
                                  if r["decision"] == "refused_repetitive"),
        "abstained": sum(1 for r in rows if r["decision"] == "abstained"),
        "malformed": sum(1 for r in rows if r["decision"] == "malformed"),
        "max_pairwise_body_similarity": round(max(sims), 4) if sims else None,
        "body_threshold": BODY_THRESHOLD,
    }
    OUT.write_text(json.dumps({"model": MODEL, "summary": summary, "rows": rows},
                              indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")
    print("\nBEFORE (independent generations, no rotation): max pairwise body "
          "similarity 0.116, three of four titles near-identical.")
    print("A number below the threshold here means the collapse is gone. It "
          "does NOT mean the six posts are worth publishing — read them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
