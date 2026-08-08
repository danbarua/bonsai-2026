# Patch-in-waiting: give principle 24's forward reference a path

**Target:** `CLAUDE.md`, principle 24, final sentence.
**Status: APPLIED** (2026-08-08), on Dan's instruction. Principle 24's
closing sentence now names `docs/proposals/PROVENANCE_CONTRACT.md`. Kept as
the reasoning record.

If the contract later moves out of `docs/proposals/` into `docs/`, the path
in principle 24 moves with it — the amendment fixed a dangling citation, so
leaving a new one behind would be its own punchline.

## Why

Principle 24 currently ends:

> Mechanics: see the provenance contract (capture-at-birth,
> citation-at-use).

There is no such document, and no path. A reader following that pointer
finds nothing. It is a dangling citation in the principle that exists to
stop dangling citations — which is not a gotcha so much as evidence that
the citation check in `docs/proposals/PROVENANCE_CONTRACT.md` §5 has real
work to do: this instance was written by a careful author, reviewed, and
committed, and the gap still shipped.

`docs/proposals/PROVENANCE_CONTRACT.md` is now that document.

## Proposed change

Replace the final sentence of principle 24:

```diff
-    The remedy for a captured scratch that turns out to matter is
-    promotion to committed code — never citing the capture. Mechanics:
-    see the provenance contract (capture-at-birth, citation-at-use).
+    The remedy for a captured scratch that turns out to matter is
+    promotion to committed code — never citing the capture. Mechanics:
+    `docs/proposals/PROVENANCE_CONTRACT.md` (capture-at-birth,
+    citation-at-use), which also records the measured hook-payload
+    limits any capture mechanism has to work within.
```

If and when the proposal is accepted and the document moves out of
`docs/proposals/` into `docs/`, the path in this amendment moves with it.
The amendment should not be applied ahead of that decision, or it will
itself become a stale citation.

## One substantive note for whoever applies it

Issue #22 §2 proposed that a claim could cite "a repo path **or a capture
ID**." That conflicts with principle 24's "never citing the capture." The
design resolves it in favour of principle 24 and the infrastructure brief
(`PROVENANCE_CONTRACT.md` §1.1), and the verifier is specified to reject
capture IDs outright, with a break-test pinning that behaviour.

No change to principle 24's wording is needed for that — it already says
the right thing. Noted here only so the resolution is visible to someone
reading the issue and the principle side by side and wondering which won.
