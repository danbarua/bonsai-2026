#!/usr/bin/env python3
"""Draft dispositions for gate candidates using a headless model pass.

Turns a cold read of ~138 candidate sentences into a review pass. The model
answers one question per candidate — *is this sentence a binding
requirement on the system, or prose about something else?* — and the output
is a DRAFT `gates.toml` that a human then corrects and completes.

**The scope restriction is the whole design, and it is not negotiable
here.** The model may write:

  - `[not_binding]` rows, with a one-line reason
  - the NOMINATION that a clause is binding

The model may not write, and this tool will not emit:

  - `enforcement`, `production_reachability`, `input_wiring`,
    `decision_consequence`, `break_demonstrated`

Every field in the second list is a claim about code. Asked "does this code
enforce this clause?", a model finds a way to say yes — and unlike a bad
disposition, a confabulated `enforcement = "driver.py::check_x"` has no
tell. It names a real file, a plausible function, and reads exactly like
the true rows around it. The reviewer would be checking prose written to be
agreeable. Five of the seven fields are already unverifiable by machine; if
those five were drafted too, the inventory would become a document
asserting enforcement nobody checked, dressed as a gate registry — which is
worse than no inventory, because it arrives with a green light.

A bad disposition, by contrast, is cheap and visible: it is answerable from
the document alone, and a wrong `not_binding` on a sentence the owner
recognises as a freeze jumps out immediately.

Every emitted file carries `reviewed = false`, which makes
`gate_inventory.py` fail regardless of contents. A draft cannot be mistaken
for a finished inventory even if every row looks complete.

Usage:
    uv run python tools/gates/triage_candidates.py \\
        --doc experiments/.../DESIGN.md --out gates.draft.toml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_inventory import REPO_ROOT, Clause, derive_clauses  # noqa: E402

MODEL = "claude-haiku-4-5-20251001"
BATCH = 12

_PROMPT = """\
You are triaging sentences from a frozen scientific protocol document to \
decide, for each, whether it states a BINDING REQUIREMENT on the software \
system, or is prose of some other kind.

BINDING means: the sentence commits the system to do, refuse, halt, or \
guarantee something that code could be written to enforce. Examples: an \
input must be verified before a run; a driver refuses to launch on a \
condition; an artifact is never overwritten.

NOT BINDING includes: prose describing a failure mode or a past incident; \
narration of what a result showed; commentary on why a decision was made; \
descriptions of what a human does; forward-looking notes about possible \
future work.

For each numbered item, reply with one JSON object per line, nothing else:
{{"n": <number>, "binding": true|false, "reason": "<one short clause>"}}

The reason matters most when binding is false — say what the sentence is \
instead. Keep it under 15 words. Do not comment on implementation, do not \
name files or functions, and do not speculate about what enforces anything.

Items:
{items}
"""


def ask(batch: list[tuple[int, Clause]], timeout: int) -> dict[int, dict]:
    """One headless pass over a batch. Returns {index: verdict}."""
    items = "\n".join(f"{n}. {clause.text}" for n, clause in batch)
    proc = subprocess.run(
        ["claude", "-p", _PROMPT.format(items=items), "--model", MODEL],
        capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        print(f"  model call failed: {proc.stderr.strip()[:200]}", file=sys.stderr)
        return {}
    verdicts: dict[int, dict] = {}
    for line in proc.stdout.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
            verdicts[int(record["n"])] = record
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return verdicts


def render(clauses: list[Clause], verdicts: dict[int, dict]) -> str:
    """The draft. Binding rows are NOMINATED and left empty on purpose."""
    lines = [
        "# DRAFT -- machine-triaged, not reviewed.",
        "#",
        "# `reviewed = false` makes gate_inventory.py fail regardless of what",
        "# follows, so this file cannot be mistaken for a finished inventory.",
        "# Set it true only after a human has read every row.",
        "#",
        "# Dispositions below were drafted by a model reading the documents",
        "# ALONE. Nothing here claims anything about code: every [binding]",
        "# row is deliberately empty, because a drafted `enforcement` field",
        "# would name a real file and a plausible function and read exactly",
        "# like a true one. Those fields are for a human who checked.",
        "reviewed = false",
        "",
    ]
    nominated, dispositioned = [], []
    for n, clause in enumerate(clauses, start=1):
        verdict = verdicts.get(n)
        if verdict is None:
            continue  # unanswered stays undispositioned, and so a finding
        (nominated if verdict.get("binding") else dispositioned).append(
            (clause, verdict))

    lines.append(f"# {len(nominated)} nominated binding, "
                 f"{len(dispositioned)} nominated not-binding, "
                 f"{len(clauses) - len(nominated) - len(dispositioned)} "
                 f"unanswered (left undispositioned deliberately).")
    lines.append("")

    for clause, _ in nominated:
        lines += [
            f"# {clause.doc}:{clause.line}",
            f"# {clause.text[:150]}",
            f'[binding."{clause.clause_id}"]',
            "# TO BE COMPLETED BY A HUMAN -- every field below is a claim",
            "# about code that a model must not make.",
            '# enforcement             = ""',
            '# production_reachability = ""',
            '# input_wiring            = ""',
            '# decision_consequence    = ""',
            '# test                    = ""',
            '# break_demonstrated      = ""',
            '# trigger                 = ""',
            "",
        ]
    for clause, verdict in dispositioned:
        reason = str(verdict.get("reason", "")).replace('"', "'")[:120]
        lines += [
            f"# {clause.doc}:{clause.line}",
            f"# {clause.text[:150]}",
            f'[not_binding."{clause.clause_id}"]',
            f'reason = "DRAFT: {reason}"',
            "",
        ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)

    clauses = derive_clauses(args.doc, args.root)
    if not clauses:
        print("no candidates derived -- nothing to triage", file=sys.stderr)
        return 2
    print(f"triaging {len(clauses)} candidates in batches of {BATCH}")

    verdicts: dict[int, dict] = {}
    numbered = list(enumerate(clauses, start=1))
    for start in range(0, len(numbered), BATCH):
        batch = numbered[start:start + BATCH]
        got = ask(batch, args.timeout)
        verdicts.update(got)
        print(f"  {start + len(batch)}/{len(numbered)} "
              f"({len(got)}/{len(batch)} answered)")

    args.out.write_text(render(clauses, verdicts))
    unanswered = len(clauses) - len(verdicts)
    print(f"\nwrote {args.out} -- {len(verdicts)} drafted, "
          f"{unanswered} unanswered")
    print("The draft carries `reviewed = false`; the reconciler will fail "
          "against it until a human sets it true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
