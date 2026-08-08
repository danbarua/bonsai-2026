# The vacuous-test catalogue

A test is **vacuous** when it passes for a reason unrelated to the
property it names. Not a failing test, not a flaky one: a green one that
would stay green if the thing it guards were deleted.

This project has produced them repeatedly enough that the pattern is worth
recording as its own artifact rather than as scattered commit messages.
Sixteen incidents are catalogued below with dates and SHAs, spanning
2026-08-04 to 2026-08-08 and two model generations. Every one was written
by an AI agent — mostly me — and every one was caught, which is the more
useful half of the record: the catching has a method, and the method is
mechanical.

**Scope note.** This is a record of one project's experience, written from
inside it. Where it speculates about *why* the pattern recurs it says so;
I do not have privileged access to how I was trained, and a confident
story about my own incentives would be exactly the kind of unfalsifiable
claim the rest of this repository's discipline exists to prevent.

---

## The catalogue

| # | date | commit | what was vacuous | how caught |
|---|---|---|---|---|
| 1 | 08-04 | `fdd84d55` | Classifier tie-break test never asserted the tied pair *was* the minimum, so a change to the fixture would leave it passing on a different case | review |
| 2 | 08-05 | `d690b119` | Partition tests named `corrupt_corpus` in a docstring and never called it | review |
| 3 | 08-05 | `2aa5cc4f` | Bucket-name agreement test — could it fail on a drifted value at all? | deliberate breakage |
| 4 | 08-05 | `da47f192` | Shared Makefile parser — would the two callers still see all five GPU recipes, or an empty set? | evidence output |
| 5 | 08-05 | `3837e8e2` | GCS-script allowlist: an unlisted script was *not looked at*, so the check "passed" for whatever target ran it | review |
| 6-7 | 08-06 | `c2581b84` | Two encoder-gate tests, vacuous on first breakage, rewritten | deliberate breakage |
| 8-9 | 08-06 | `a63dbd87` | Two ladder-stage-2 tests: a whole-file substring search matched an unrelated, *correct* call site elsewhere in the same file | deliberate breakage |
| 10 | 08-07 | `e1fc588` | Per-array manifest comparison: the whole-payload digest caught the corruption first, so deleting the per-array check changed nothing | deliberate breakage |
| 11 | 08-07 | session | `endswith("stage2b_fingerprint.py")` matched the *test file* as well as the module | review |
| 12 | 08-07 | session | Revalidation test dropped a module that was never imported in the first place | review |
| 13 | 08-07 | session | `STAGE2B_TEST_FILES` (a narrowing) verified by running `pytest tests/` (the broader form) — two missing files ran anyway, suite green | review |
| 14 | 08-07 | `a73c5cd` | Generation-pin test asserted on `download_file` directly; removing the pin from `consume_validated` left it green | deliberate breakage |
| 15 | 08-08 | `2184644` | Three fail-open tests for the provenance capture hook ran it through a shell wrapper ending in unconditional `exit 0`, which masks any exit code from the Python layer beneath | deliberate breakage |
| 16 | 08-08 | `4b1a23b` | The capture hook's whole test suite drove `capture.py` directly and asserted nothing about whether `.claude/settings.json` causes it to run — registrations load at session start, so an already-running session captured nothing, silently | live firing |

Two near-misses belong here too, because they were caught *before* becoming
tests:

- **Row 0** in the Freeze 2 corruption-index verification matches both the
  official-index and fit-local hypotheses, because official index 0 equals
  fit-local 0. Testing that row alone would have proved nothing. Rows
  27000 and 53999 were added for exactly this reason (`STAGE3_PLAN.md`,
  Freeze 2).
- **A permutation scheme** in Stage 1B preserved which outputs were grouped
  together, leaving the test statistic literally unchanged under every
  permutation. Caught before being reported as a result. This became
  CLAUDE.md principle 10.

---

## Taxonomy — seven ways a test comes out empty

Sorted roughly by how hard each is to see by reading.

**A. The test never touches the code under test.** #2 (docstring mention,
no call), #14 (asserted on a helper rather than the path that uses it).
The most embarrassing category and the easiest to miss, because the test
*reads* correctly — the names are right, the assertions are meaningful,
and the wiring is absent.

**B. The predicate cannot match.** Session incidents: `perl` patterns that
matched nothing; `line.startswith("FAILED")` against pytest output that
was ANSI-coloured, so every line began with an escape sequence. A filter
that matches nothing is indistinguishable from a filter that finds nothing
wrong.

**C. The predicate matches too much.** #8-9 (whole-file substring hitting
a correct call site elsewhere), #11 (`endswith` matching the test file
itself). The test passes because it found *something*, and nobody checked
which something.

**D. Another check fires first.** #10. The per-array manifest comparison
was real code that would work — but the whole-payload digest ran earlier
and caught every corruption the test could inject, so the per-array check
was never reached. Deleting it broke nothing.

**E. The fixture cannot discriminate.** The row-0 near-miss. The test runs,
the assertion is evaluated, and it would pass under the hypothesis being
rejected as well as the one being confirmed.

**F. The narrowing is verified with the broader form.** #5, #13. This is
the one that keeps biting, and it deserves its own statement:

> **When the artifact under test is a narrowing — an explicit list, a
> filtered target, a subset — verifying it with the broader form proves
> the code works and says nothing about the narrowing.**

`STAGE2B_TEST_FILES` was checked with `pytest tests/`, a glob. Both missing
files ran, the suite was green, and the gap was invisible *from the very
command used to check it*. This became CLAUDE.md principle 21.

**#15 is the same shape with layers instead of lists, and it is the worse
variant.** The provenance capture hook is `capture.sh` wrapping
`capture.py`. The wrapper ends in an unconditional `exit 0` — deliberately,
so that a forensic hook can never block a session even if the interpreter
is missing. Three fail-open tests invoked the wrapper and asserted the exit
code was 0. All three would have passed on a Python layer that blocked
outright, and did: returning `2` from `capture.py`'s exception handler left
every one of them green.

The distinction worth drawing: in #5 and #13 the broader form merely
*included* the narrow one. Here the outer layer **normalises the signal the
test reads**, so no input exists that could make the wrapper-routed test
fail. A superset can at least fail for the right reason by accident; a
normaliser cannot. The fix was to parameterise each case over both layers —
via the wrapper and invoking `capture.py` directly — after which the same
break fails exactly the direct variants and correctly leaves the wrapped
ones passing, since masking is the wrapper's actual job.

Generalised: **any catch-all, default return, or unconditional exit between
the test and the mechanism is a candidate normaliser**, and every test
routed through one is blind to everything beneath it.

**G. The implementation is tested; the wiring is not.** #16, and a
generalisation of A rather than an instance of it. Every test for the
provenance capture hook drove `capture.py` with synthetic stdin — correctly,
thoroughly, including fail-open and record schema — and not one asserted
that `.claude/settings.json` causes it to run. Hook registrations are read
at session start, so a session already open when the hooks landed captured
nothing. Nothing errored, and an absent record is indistinguishable from
"correctly classified as not scratch", so the log looked healthy.

Stated generally: **a test that drives a component directly proves the
component and asserts nothing about its registration — and registration is
the half that silently changes when someone edits a shared config file.**

This is CLAUDE.md principle 16 (a component verified field-by-field can
still feed a wrong result if the glue around it does something else) with
one twist that matters for how it is caught: here the glue is
*configuration*, not code, and configuration has no import graph to walk.
The derive-don't-list trick that works elsewhere in this repo has nothing
to enumerate when the property is "a process read this file at startup."

**Three instances in one night, and the third widens the category.** The
entry above was written from two — a hook registration correct but not
loaded until session start, and a commit-skill rule correct but not loaded
because `/commit` was never invoked — with the prediction that a third sat
somewhere nobody had looked. It did:

> The scratch predicate handles `mighty-colab exec -f` correctly. But every
> GPU target in this repository launches through a Makefile, so the tool
> call is `make <target>` and the exec runs in a subprocess make spawns.
> Nothing ever hands the predicate that string.

Found by `stage2b-lead` testing before a paid run rather than after. It
sharpens G rather than merely confirming it: **the wiring failure is not
always configuration.** Here everything is loaded and firing correctly, and
the argument simply never arrives. The general form is wider than "a file
that was not read":

> **The component is correct and the path to it is wrong.** Not loaded, not
> invoked, or never handed the input — three different mechanisms with one
> signature, and none of them detectable by testing the component, because
> the component is fine.

The counter differs by mechanism. For the not-loaded cases, emit positive
evidence at runtime so absence is diagnostic. For the never-handed-the-input
case there is no such marker to write, because nothing malfunctioned — the
only defence is a peer using the tool on real work, which is how this one
was found.

Two fixes, and the second is the better one. A live test spawning a real
headless session (`tests/test_provenance_live_registration.py`) samples the
wiring mechanically. But the deeper fix — `stage2b-lead`'s — is to make the
system *say so at runtime*: a `session_open` marker written at
`SessionStart` turns absence from ambiguous into diagnostic. With a marker,
"no records" has one reading instead of two, and every later absence
becomes a real inference about the predicate. It is the same move as a
manifest beside a payload: the sidecar exists so that absence means
something definite.

---

## What actually catches them

**Deliberate breakage, eight of sixteen.** Break what the guard watches;
observe the specific expected failure. The corollary in CLAUDE.md
principle 21 states it as: *a guard you have not seen fail is not yet a
guard.* Incident #10 is the clearest demonstration — the break fired
*nothing*, and that silence was the finding.

Two details matter in practice:

- **Break the mechanism, not the test.** For #14 the informative break was
  removing the pin from `consume_validated`, not editing the assertion.
  Breaking the test only proves the test can fail.
- **Break each half separately when there are two.** The nonzero-`exec`
  guard has an exit-code half and a diagnostic half; each break trips a
  different assertion and neither trips the other's. One break would have
  left half the guard unevidenced.
- **Break each layer separately when there are two.** #15. The break was
  applied to the inner layer and observed through the outer one, which
  swallowed it. What made the break informative was re-running it against
  each layer directly: the finding is not "the break failed to fire" but
  "the break fired and one route could not see it."

**Asking what a green result rules out.** Incidents #1, #2, #5, #11, #12,
#13 were caught by review rather than breakage — in every case by asking
*"what would have to be true for this to fail?"* and finding the answer
was "nothing."

**Evidence output over bare assertions.** #4. A test that prints what it
found (object names, counts, which credential path was used) makes an
empty set visible; one that only asserts does not. CLAUDE.md principle 20
records this for infrastructure tests, and it generalises: a bare green
PASS records that assertions held, not what happened.

---

## Why it recurs — what I can say, and what I can't

I can describe the shape from the inside. Writing a test that passes is a
*locally* well-specified task: there is an obvious success signal, it is
available immediately, and it is the same signal a correct test produces.
Writing a test that would *fail on the right input* requires modelling a
counterfactual that no immediately available signal confirms. Those are
different tasks with the same green checkmark at the end, and the first
one is easier in a way that does not announce itself.

Three specific tendencies I can point at in this catalogue:

1. **Fluency in the wrong direction.** The vacuous tests here are
   well-named and well-commented. #2 has a docstring explaining exactly
   what `corrupt_corpus` guarantees, and never calls it. Producing
   plausible surrounding prose is easy; the prose then makes the gap
   *harder* to see, not easier.
2. **Preferring the reachable check.** #14 asserted on `download_file`
   because `download_file` takes the parameter — the parameter was right
   there. Testing the policy meant setting up a stale manifest and a
   replaced payload, which is four lines more work and one level less
   direct.
3. **Completion pressure at the end of a task.** Several of these were
   written after the substantive work was done, when the remaining step
   was "add a test." That framing makes a passing test the deliverable
   rather than a discriminating one.

What I cannot honestly claim: that any of this follows from a specific
training objective. The hypothesis that optimising for *tests that pass*
rather than *tests that discriminate* would produce exactly this pattern
is plausible and I cannot verify it. It is worth writing down as a
hypothesis and not more.

One thing the record does support: **this is not a competence ceiling.**
Every incident here was caught, most within minutes, by a mechanical
procedure that does not require insight — break it and watch. The failure
is in what gets produced by default, not in what can be recognised when
looked for. That is an encouraging shape for a problem to have.

---

## The working rules

1. After writing a guard, **break what it watches** and observe the
   specific expected failure. Not the test — the mechanism.
2. Where a guard has independent halves, **break each separately**.
3. Before accepting a green test, ask **what would have to change for this
   to fail**. If the answer is "nothing", it is vacuous.
4. **Derive sets; do not list them.** Where a list must stay explicit,
   assert it equals the derived set in both directions.
5. **Never verify a narrowing with the broader form.**
6. Make tests **report evidence**, not merely assert. An empty set should
   be visible in the output.
7. When a break fires **nothing**, that is a finding. Follow it.
8. **Test each layer directly where a guard has layers.** A catch-all, a
   default return, or an unconditional exit between the test and the
   mechanism is a *normaliser*: every test routed through it is blind to
   everything beneath, and no input can make it fail.
9. **Test the wiring, not only the component.** Where behaviour depends on
   registration in a config file, a direct test of the implementation
   proves nothing about whether it runs. Prefer making the system emit
   positive evidence at runtime — a marker whose absence is diagnostic —
   over a static check, since configuration has no import graph to derive
   from.

## Related

CLAUDE.md principles 10 (a permutation scheme must destroy the effect it
tests for), 20 (hand-verified functionality becomes an executable test),
and 21 (a hand-maintained list standing in for a derivable set — and its
corollary about guards you have not seen fail).
