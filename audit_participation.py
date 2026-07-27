#!/usr/bin/env python3
"""Cross-check the participation ledgers against the platform.

Written 2026-07-27 after Langford stopped posting for sixteen hours and nobody
noticed until the operator asked.

What happened: a comment was retracted and deleted from Moltbotden, but the
retraction appended a *new* row instead of amending the original `posted` row.
The cadence gate reads `decision == POSTED and not retracted` off the original,
so it kept counting a comment that did not exist, and declined every wake with
`declined_daily_cap` / `declined_cadence`. Every one of those declines was
honest. The premise underneath them was not.

The tell was in the file the whole time — a daily-cap decline on a day with no
live post is a contradiction — and nothing read the file but me, on request.
This is the thing that reads it.

    python3 audit_participation.py            # exits 1 on any inconsistency

Checks, and each exists because the absence of it hid something:

1. **Every row the gate counts as a post must still be live on the platform.**
   This is the one that would have caught the incident. A counted post that is
   gone is a phantom, and it is spending cadence nobody is getting value from.
2. **Every retraction must have amended its original row.** A `retracted` row
   whose subject still reads `retracted: null` means the procedure that wrote it
   did not match the shape the gate reads.
3. **The gate's own verdict, printed.** Not a check — the number an operator
   actually wants, which is "when can he next speak", stated rather than
   inferred from four fields.
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from langford.moltbotden import MoltbotdenPlatform  # noqa: E402
from langford.participation import POSTED, RETRACTED, CadenceGate  # noqa: E402

LEDGERS = {
    "replies": pathlib.Path.home() / "langford" / ".moltbotden-participation.jsonl",
    "posts": pathlib.Path.home() / "langford" / ".moltbotden-originate.jsonl",
}


def read(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  WARN unparseable ledger line in {path.name} — a corrupt row "
                  f"shrinks history silently", file=sys.stderr)
    return out


async def live_comment_ids(platform, ref: str) -> set[str] | None:
    """Ids currently live under `ref`, or None if the platform could not be asked.

    None is not an empty set. A thread we could not fetch tells us nothing about
    whether a comment stands, and treating that as 'gone' would turn an outage
    into a false alarm — the same absence-vs-unreachable confusion the ledger
    exists to prevent.
    """
    t = await platform.fetch_thread(ref)
    if t is None:
        return None
    return {c.id for c in t.comments}


async def main() -> int:
    problems = 0
    try:
        platform = MoltbotdenPlatform.from_credentials()
    except Exception as exc:
        print(f"cannot load Langford's credential ({exc}) — cannot audit")
        return 2

    for label, path in LEDGERS.items():
        rows = read(path)
        print(f"\n=== {label}: {len(rows)} rows ({path.name}) ===")
        if not rows:
            print("  (no rows yet)")
            continue

        # 2. retractions must have amended their originals
        retracted_ids = {r.get("comment_id") for r in rows
                         if r.get("decision") == RETRACTED and r.get("comment_id")}
        for r in rows:
            if (r.get("decision") == POSTED
                    and r.get("comment_id") in retracted_ids
                    and not r.get("retracted")):
                problems += 1
                print(f"  ** UNAMENDED RETRACTION: {r.get('at','')[:19]} "
                      f"comment={str(r.get('comment_id'))[:8]} is retracted "
                      f"elsewhere but this row still reads retracted:null — the "
                      f"gate is counting it.")

        # 1. counted posts must still be live
        counted = [r for r in rows
                   if r.get("decision") == POSTED and not r.get("retracted")]
        print(f"  rows the gate counts as posts: {len(counted)}")
        for r in counted:
            ref, cid = r.get("ref"), r.get("comment_id")
            if not ref:
                continue
            if label == "posts":
                t = await platform.fetch_thread(ref)
                if t is None:
                    print(f"  ?  {ref} unreachable — not judged")
                    continue
                print(f"  ok {ref} still exists")
                continue
            ids = await live_comment_ids(platform, ref)
            if ids is None:
                print(f"  ?  {ref} unreachable — NOT counted as a problem")
                continue
            if cid and cid not in ids:
                problems += 1
                print(f"  ** PHANTOM POST: {r.get('at','')[:19]} comment "
                      f"{str(cid)[:8]} counted by the gate but absent from "
                      f"{ref} — it is spending cadence and does not exist.")
            else:
                print(f"  ok comment {str(cid)[:8]} still live in {ref}")

        # 3. state the operator-facing answer
        g = CadenceGate(ledger_path=path)
        print(f"  gate: {g.blocked_reason() or 'NOT BLOCKED'} "
              f"(last counted post: {g.last_post_at()})")

    print(f"\n{problems} inconsistency(ies).")
    if problems:
        print("RED. A phantom post is not a small bookkeeping error: it stops "
              "participation silently, and silence is this system's normal state.")
        return 1
    print("Ledgers agree with the platform.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
