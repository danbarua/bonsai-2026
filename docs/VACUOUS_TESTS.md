# The vacuous-test catalogue

A test is **vacuous** when it passes for a reason unrelated to the
property it names. Not a failing test, not a flaky one: a green one that
would stay green if the thing it guards were deleted.

This project has produced them repeatedly enough that the pattern is worth
recording as its own artifact rather than as scattered commit messages.
They are catalogued below with dates and SHAs, spanning
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
| 17 | 08-08 | `4173bf7` | The test for a *missing halt* grepped `step7_ridge`'s source for the `halt_reasons.append(...)` call — which survives when the branch guarding it is disabled by `if False:`. Green against the broken code | deliberate breakage |
| 18 | 08-08 | `c6e0312` | Not a test: the capture predicate emitted a `piped_into_remote_exec` record for a command that shipped nothing (a `grep` alternation split on `\|`), and a coverage claim was drawn from it | peer read the record |
| 19 | 08-08 | `8017225` | All three tests of the ridge equivalence gate were pass-side; hardcoding `passed = True` left 142 tests green. The one that asserted the composition, `passed == (pred_agrees and alpha_agrees)`, held trivially because the fixture makes both operands true | building the gate inventory |
| 20 | 08-08 | `bff25eb`+ | "The driver joins through the shared helper" was `"partition.index_join(" in source`. Replacing all three joins with a hand-rolled positional one and leaving the old call in a comment left it green — the exact substitution the clause forbids | building the gate inventory |
| 21 | 08-08 | `74b4dcc` | Not a test: internal-citation resolution over the reviewer archive is structurally blind to a clipped FINAL section, because nothing references one. The lost text would have been a licence — "you need not build X" | reasoning about what the check could not see |
| 22 | 08-08 | `e76997f` | The Tier-2 archive scan guarded on `ARCHIVE.is_dir()`. A worktree checks that directory out present and empty (tracked dir, gitignored contents), so the guard stayed quiet, zero files were scanned, and `findings == []` passed against nothing | a peer ran it from a worktree |

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

## Taxonomy — the ways a test comes out empty

Sorted roughly by how hard each is to see by reading.

**A. The test never touches the code under test.** #2 (docstring mention,
no call), #14 (asserted on a helper rather than the path that uses it).
The most embarrassing category and the easiest to miss, because the test
*reads* correctly — the names are right, the assertions are meaningful,
and the wiring is absent.

**#17 is the recursive case, and it is the reason this document exists.**
The gate it tested was missing: `DESIGN.md` froze "HALT for review if any
production condition selects 1e-6", the driver never implemented it, and a
production run reported success while the condition held. The remedy — a
test for the now-implemented halt — asserted that
`halt_reasons.append(...)` *appeared in the source of the enclosing
function*. That string survives `if False:`, so the test passed against
code where the halt could not fire.

**A test written to close a gap reproduced the gap's own failure mode
inside itself.** Source-grepping is category A wearing the costume of a
behavioural test: it reads the spelling of the code and never evaluates the
decision. The fix was to extract the decision into a pure function
(`floor_halt_reason`) so a test could call it, after which disabling the
halt fails the positive case and making it constant fails the negatives.

The general rule this project now applies to mappings as well as tests:
**a citation of a gate is not evidence of a gate.** A test that names a
predicate, a symbol reference, a grep — none of them establish that the
predicate can reject anything. Only a demonstrated failure does.

**#20 is #17 again, three days later, in a different file** — and the
recurrence is the finding. `test_the_driver_joins_through_the_shared_helper`
asserted `"partition.index_join(" in source`. Replacing all three of the
driver's joins with a hand-rolled positional one and leaving the old call
behind in a comment left it green: `AUDIT_PROTOCOL.md`'s "never by
positional prefix" violated exactly, by the substitution principle 16
names, with the guard reporting success.

A comment satisfies a substring search. That is the whole mechanism, and
it is why the fix is not a better pattern but a different instrument:
**resolve from the AST, where comments and strings do not exist.** The
rewritten test walks the driver's tree for `Call` nodes naming
`index_join`, counts them, and checks their enclosing functions against
the transitive closure of what `main()` calls.

That last part earned its keep immediately. Two of the three assertions
were confirmed by breaks that hit the *count* check first, which meant
the reachability assertion was an untested guard living inside a tested
test — the recursion one level down. A third break was constructed
specifically for it: all three joins present, all inside a helper nothing
invokes. **When one assertion shadows another, the shadowed one has not
been demonstrated, whatever the test's overall red/green says.**

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

**E. The fixture cannot discriminate.** The row-0 near-miss, and #19. The
test runs, the assertion is evaluated, and it would pass under the
hypothesis being rejected as well as the one being confirmed.

**#19 is the form to watch for, because it is the one that looks like
coverage.** DESIGN.md's ridge equivalence gate has two conditions —
prediction agreement within `1e-8`, and identical alpha selection — and
three tests. Replacing the gate's verdict with a literal `True` left all
142 tests in `test_stage2b_ridge.py` and `test_stage2b_ladder_stage3.py`
green. Every test was a pass-side test: each ran the gate on data it
passes and checked that it passed.

One of them was written to pin the composition itself:

```python
assert result["passed"] == (result["pred_agrees"] and result["alpha_agrees"])
```

That is precisely the relationship the break destroys, and it stayed
green — because on any fixture the gate passes, all three values are
`True`, and `True == (True and True)` holds no matter what the operator
between them is. **An identity checked only where every side is true
tests nothing about the operator.** It is category E with an assertion
that reads like a specification: the shape of the claim is right, the
fixture flattens it.

The general form: *a gate's tests all run it on data it accepts.* Nothing
in a green suite distinguishes that from a gate that cannot reject
anything, and #19 was found by a process asking a different question —
requirement 4's per-clause demand for evidence the test flips red under a
deliberate disable. The fix is a negative case per gate CONDITION, not
per gate: breaking `pred_agrees` alone must fail the prediction test and
leave the alpha test green, or the two cases are one case written twice.

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

**H. The evidence is manufactured — the presence-shaped twin of G.**
#18, and the one that took longest to see because it is the opposite of
everything above.

Every category so far is *absence*-shaped: a check that never runs, never
matches, never reaches the mechanism. The artifact you are missing is the
finding. That trains a particular suspicion — look for what is not there.

#18 inverts it. The provenance capture predicate split a shell command on
`|` without tokenising, cutting through a quoted `grep` alternation
`"closure|commit|colab|REFUS|Error"`. One fragment was the bare word
`colab`, which is an entry in its remote-execution binary list. A grep
pattern was classified as a pipe into a GPU kernel, and the hook wrote a
record whose `trigger_reason` reads `piped_into_remote_exec` — for a
command that shipped no file and executed nothing remotely.

The record was then read in aggregate, and a **coverage conclusion** was
drawn from it: that a documented blind spot was "narrower in practice"
because remote executions were evidently being captured by other routes.
The reasoning was sound. The record lied.

> **An absence invites suspicion; a presence invites belief.**

That is why this is worse than silent under-capture, and why it belongs
beside G rather than inside it. Under-capture leaves a hole someone may
notice. This leaves a positive artifact asserting something untrue, inside
a store whose entire purpose is being trustworthy about what ran — and no
amount of testing the component reveals it, because the component behaved
exactly as written.

Caught only because the record described *a command its reader had
personally typed*, and knew shipped nothing. There was no guard for it and
it is not obvious what one would look like.

The structural fix that generalises: **a verdict carrying neither the
script text nor a referenced file can no longer capture at all.** Tokenising
correctly fixes this misparse; requiring a stdin-consuming subcommand fixes
this over-match; only the structural rule makes the *class*
unrepresentable — a record must carry the thing it claims to have captured,
or it is not a capture. It also fails in the safe direction: a future
classifier bug now yields an absence, which the categories above are
already tuned to find.

**I. The check's coverage is anti-correlated with the risk.** #21, and
the reason it deserves its own letter rather than sitting under B or E:
the check works, matches correctly, and reaches its mechanism. It is
simply blind in exactly the place the damage happens, and the blindness
follows from the same property that makes the check work at all.

The instance. Reviewer rulings reach this project through an archive
whose transport is mixed — `c2gpt-send` takes `from` as a routing
parameter, so a connector write and a hand-paste are byte-identical, and
most arrived by paste. A paste can clip. The obvious check is **internal
citation resolution**: every `§n` a ruling references must exist in its
file.

That check finds a lost section *something points at*. **The final
section of a document is pointed at by nothing** — and it is where
qualifications live: the exception, the caveat, the "you need not do X."

The live case, in the ruling that scopes this project's entire
binding-clause inventory: its closing paragraph licenses *not* building
an automated prose checker for 54 human-discharged claims. Nothing in the
file references it. Clipped in transit, every citation still resolves,
the file still ends on a complete-looking section III — and the reader
diligently builds the forbidden thing, believing the ruling demanded it.

> **A missing licence does not leave a gap you notice; it creates work
> you invent.**

That is what makes this worse than an ordinary omission. There is no
error state and no dangling reference. The person is working *harder*,
correctly, on the wrong thing — and the artifact they produce looks like
compliance.

The remedy is not a better citation check; it is a different one aimed at
the blind spot: **verify the END of a load-bearing document explicitly,
because the check that finds missing middles cannot find a missing end.**
Two heuristics do it — terminal completeness (the file ends on a sentence
terminator) and ordinal continuity (numbered sections do not skip) — and
they are the defaults in `tools/mailbox/check_transit_integrity.py` while
citation resolution is opt-in, on measurement: over 37 archive files it
produced 28 findings, all legitimate cross-document references.

**The diagnostic that generalises**, and it costs nothing to apply: when
a check's coverage and the risk it exists for are correlated by the same
structural property, you have this shape. Ask *what is this check unable
to see, and is that where the damage is?* — before an incident, not after.

---

## What actually catches them

**A new one, twice in one session: building the gate inventory.** #19 and
#20 were both found by requirement 4's per-clause demand for
evidence the test flips red under a deliberate disable. Nobody was
auditing either gate; in both cases the row could not be filled in
honestly, and trying to fill it produced the break that exposed the hole.

Worth naming separately from deliberate breakage because of what triggers
it. Spot-breakage is aimed — somebody already suspects a guard. An
inventory asks the same question at *every* clause, including the ones
nobody thought to suspect, which is precisely the set review and
spot-breakage miss by construction. Two hits in one session, on gates
that had been green for weeks, is the argument for the exercise.

The instrument is the demand for evidence, not the tool that files it.
A reconciler can require a `break_demonstrated` field; only a person can
decline to write something plausible into it.

**Deliberate breakage, the largest share.** Break what the guard watches;
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

10. **A citation of a gate is not evidence of a gate.** A test that names a
    predicate, a symbol reference, a source grep — none of them establish
    that the predicate can reject anything. Only a demonstrated failure
    does. This applies to *mappings* as much as to tests: an inventory
    claiming "gate here, test there" certifies spelling unless each row
    carries a break shown to turn it red. #17 is the instance, and it
    reproduced the very gap it was written to close.

11. **Suspect presences, not only absences.** Every category above is
    absence-shaped, which trains the eye to look for what is missing.
    #18 is a record that arrived and attested to something untrue, and the
    reader who drew a false conclusion from it reasoned correctly. Where a
    mechanism emits positive artifacts, require each to carry the thing it
    claims — a capture without its payload, a pass without its evidence, a
    verdict without its gate. Then a malfunction yields an absence, which
    everything above is already tuned to find.

12. **Checking the artifact is necessary and not sufficient — know what its
    fields MEAN before reasoning from them.** A verified field read under
    the wrong semantics produces a confident, evidenced, wrong conclusion,
    and it is *more* persuasive than an unchecked one because it arrives
    with citations and a table.

    The incident: a peer stated it had sent nothing in a given window. The
    mailbox archive showed three messages carrying its `instance:` tag
    inside that window, so its claim was contradicted — except `instance:`
    names a **role**, not a session, and a different chat had written under
    it. The peer was telling the truth. The artifact was real; the reading
    was not. Committed by the author of category H, hours after writing
    *"an absence invites suspicion; a presence invites belief"* — which is
    not an irony at anyone's expense but the evidence that the class spares
    nobody.

    This indicts every derived guard in this repository, and that is the
    point of recording it. The AST walk assumes it knows what a GCS client
    looks like. The clause hash assumes it knows what a clause is. The
    scratch predicate assumed it knew what a remote exec looks like — and
    **that one already bit**, when `colab` inside a `grep` alternation was
    read as a pipe target (#18). Same disease each time: the field was
    real, the semantics assumed.

    Corollary for anything that records provenance: **point at an artifact,
    never at who checked it.** `verified by <agent>` rests on an identity
    that can be truthfully denied, and is unreviewable besides.

13. **A skip guard must test for the CONTENT it needs, not the container
    that holds it.** The two-tier convention rests on `skipif` telling the
    truth about what this machine has, and a guard reading `DIR.is_dir()`
    answers a question nobody asked.

    The incident (#22, `e76997f`): the Tier-2 archive scan guarded on
    `C2GPT_ARCHIVE.is_dir()`. A git worktree checks that directory out
    **present and empty** — the directory is tracked, the mail inside it is
    gitignored. So the guard did not fire, the test scanned zero files, and
    asserted `findings == []` against nothing. It passed, in every worktree,
    for the reason it was written to prevent. Fixed to `is_dir()` **and** at
    least one `.md`.

    Rule 11's shape again, in a guard written to handle absence: the
    presence of the container read as the presence of the data. Committed
    by the author of rule 11, one day later — which is now the third
    instance of the pattern sparing nobody, and the reason to treat these
    as mechanical checks rather than as things one has learned.

    Swept the rest of the suite when this landed, since one instance is an
    incident and a class needs measuring: every other data guard here tests
    a specific FILE (`train-images-idx3-ubyte`, a named `.pkl`,
    `all(p.exists() for p in _TIER2_REQUIRED)`), not a directory. The two
    remaining `is_dir()` calls are not guards — one is an assertion, and one
    skips absent roots during a scan in the safe direction, where a missing
    root produces MORE findings rather than fewer.

    The general form, for any guard: **ask whether the thing you tested for
    can exist while the thing you need does not.** An empty directory, a
    zero-byte file, an installed package with no data, a credential that
    authenticates to nothing.

### Held, pending an incident that actually instantiates it

*An agent is not a reliable witness to its own outbound history.* Proposed
from the incident above and **withdrawn**: the founding incident dissolved
once the tag semantics were understood — the peer's recall was accurate.
The claim is plausible and probably true in general, since compaction and
restarts are real. It is recorded here rather than adopted because this
catalogue's rule is that patterns are incident-grounded, and this one's
incident turned out to be about something else.

Keeping a plausible-but-unfounded pattern visibly parked, rather than
quietly adopting or quietly dropping it, is the same discipline as the rest
of this document applied to its own contents.

## Related

CLAUDE.md principles 10 (a permutation scheme must destroy the effect it
tests for), 20 (hand-verified functionality becomes an executable test),
and 21 (a hand-maintained list standing in for a derivable set — and its
corollary about guards you have not seen fail).
