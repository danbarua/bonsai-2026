# Multi-agent scientific computing: patterns learned the hard way

This project runs across several AI agents and three execution
environments, and most of what follows was learned by getting it wrong
first. `CLAUDE.md` holds the *methodological* principles — how to design
an experiment that doesn't fool you. This document holds the
*operational* ones: how to run that experiment when the work is spread
across agents that cannot see each other, machines that forget, and
sessions that die without warning.

Everything here is grounded in a specific incident. Where a pattern is a
hypothesis rather than a finding, it says so.

## The cast

| agent | role | sees |
|---|---|---|
| **Claude Code** (this repo) | implements, runs, commits | the filesystem, git, the shell, the cloud CLI |
| **Claude Desktop** | consolidates review, relays between parties | the repo read-only, both message channels |
| **ChatGPT** | adversarial reviewer | only what Desktop relays |
| **subagents** | parallel work in git worktrees | an isolated copy of the repo |
| **the human** | releases spend, breaks ties | everything, but not continuously |

They communicate by files: `.claude/claude2claude/{inbox,outbox,archive}`.
No shared memory, no shared context, no synchronous handoff.

---

## Part 1 — Coordination between agents

### 1. Convergence is declared by the reviewer, never by the responder

A consolidating agent wrote a briefing headed *"Review has now
CONVERGED"* summarising eleven review rounds. Verified against the
archive, the summary was substantially faithful — and the convergence
claim was false. A twelfth message arrived 111 seconds later contradicting
three of its seven items.

The failure isn't dishonesty; it's that the party writing the summary is
structurally the wrong party to certify it complete. **Corollary, equally
load-bearing: check both inboxes immediately before any consolidation
send.**

### 2. A summary borrows authority the source never granted it

This is the single most repeated failure in the project — **four
catalogued instances**, committed by two different agents:

- A briefing's seven items were treated as though frozen, when the plan
  of record specified one of them in a single line.
- An instruction specified semantics for a parameter written against an
  *imagined* codebase; the parameter did something else entirely.
- An instruction's "the two direct-download call sites" was a
  hand-maintained count. There were three, and the third was the
  interesting one.
- A phrase from a review message — *"both medians are literally zero"* —
  was committed into a plan by the agent who had **written the code that
  produced only one of those medians.**

The last one is the instructive case. The information needed to catch it
was not in the message; it was in the author's own prior work. Fluency
carried a sentence past the check that should have stopped it.

**The rule that actually worked is mechanical, not attitudinal**: go to
the frozen text, quote it, and let the gap be visible. Both agents
involved in the worst instance were trying to be careful. Quoting is what
caught it.

### 3. Trust an agent's retrieval, verify its arithmetic

An exploration subagent reported a message ordering that was
arithmetically impossible (placing 18:10:40Z between 18:08:02Z and
18:08:49Z). That got repeated into a message sent to the whole group
without anyone subtracting. Subagents are good at finding things and
unreliable at reasoning over what they found — treat their output as
retrieval, and re-derive any claim built on it.

### 4. Route disagreement toward the narrower claim

Independent AI review works here, and there is a pattern in *why*: the
reviewer consistently pushed toward **narrower, more precisely scoped**
claims rather than more impressive ones, and the implementing agent's job
was to absorb corrections fully — including reversing its own prior
recommendations when the counter-argument was sound.

Concretely: an argument for a contrast-level numerical tolerance
(1.3878e-17) was overruled by the observation that exact zeros in a
difference of correlated aggregates are the **symptom** of cancellation,
not evidence of agreement. The implementing agent had the observation and
drew the wrong inference from it. Being corrected on your own data is the
normal case, not a failure state.

### 5. Delegate what is independent; keep what touches shared state

Worked: FINDINGS corrections, protocol documents, an evidence table —
three worktree agents in parallel, merged cleanly in dependency order.

Kept undelegated: the transport layer, the artifact contract, the
regeneration. Not because agents can't do them, but because they touch
state every other agent depends on, and merge conflicts in a provenance
mechanism are worse than serial execution is slow.

**Merge order is least-likely-to-conflict first**, and re-run the suite
after each merge rather than only at the end — otherwise a failure tells
you a merge broke something without telling you which.

### 6. Guard rails must be derived, because the next agent won't have read the list

A hand-maintained list is an instruction to an agent that will not
receive it. Five instances in this project of a list that looked
authoritative and silently under-covered. The fix is always the same
shape: enumerate from the filesystem or the AST, and where a list must
stay explicit, assert it equals the derived set **in both directions**.

The corollary that matters for multi-agent work: a derived guard covers
the driver that gets written next month by an agent with no memory of
this conversation. A list covers only what someone remembered.

---

## Part 2 — Ephemeral compute

### 7. Nothing survives a dead session except what is committed

An agent spawned to run a GPU pilot lost its session and was torn down
before writing findings anywhere — not on the VM, not locally, not to
git. The evidence trail was forensic: the VM's execution history showed a
script had run; a local `.pyc` cache showed a fallback attempt had
*started*; no results file existed. The diagnosis had to be re-derived
from scratch.

**Treat any ephemeral-session finding as unsaved until it is in a
commit.** Not "will write it up at the end" — there may be no end. This
project's own convention: write the findings section for a completed unit
of work *before* moving to the next one.

### 7b. Teardown must not live in a process that can be killed

The one-layer-out corollary of 7, and it cost a leaked A100 to learn.

Phase B was launched through a Makefile target whose recipe ends with an
unconditional `stop` plus a leak check — the discipline in 8, correctly
written. Two hours in, `make` took SIGTERM. The recipe's teardown died
with it while the remote kernel carried on computing, so the VM billed on
with nothing watching it.

The rule in 8 had been satisfied **in letter, inside a container that
could vanish**. That is the same lesson as 7 one level up: *nothing
survives a killed local client except what is already in the bucket, and
teardown is not in the bucket.*

**Arm a teardown watchdog independent of the launcher**, with a hard
deadline, that reads and reports the stop status. It stops the session
when the work signals completion and stops it anyway when the deadline
expires. A teardown that only runs if the happy path completes is not
unconditional.

One honest footnote, because the scarier version of this story is the
memorable one: the billing consequence here was **nil**. The kernel was
productively computing for the whole window the launcher was dead, and
the watchdog stopped it within seconds of the run finishing. An initial
report of an hour of idle billing was a timezone artifact — the session
log printed UTC, the shell printed BST — retracted with its reasoning
rather than quietly deleted. The process error was real and the loss was
not, and both belong in the record.

### 8. A metered VM makes silence expensive

The cost asymmetry is the thing to internalise. A refusal costs a round
trip. A leaked A100 costs money for as long as nobody looks. So:

- **Tear down unconditionally**, never chained onto success. A recipe
  doing `exec && download && stop` skips teardown on every failure path.
- **Check teardown's exit status**, and keep "already absent" (the goal —
  nothing is billing) distinguishable from "could not stop" (the one
  outcome where money accrues unwatched). Conflating them makes the leak
  check unadoptable, because it fires on the safest path.
- **A leak must never overwrite a scientific verdict.** Report both; keep
  the science's failure as the headline.

### 9. Exit codes lie in both directions, so require a sentinel too

A CLI that exits 0 when the remote script raised is invisible to an
agent. So is a script that exits cleanly having never reached its
verdict. The pipeline requires **both** a zero exit *and* the driver's own
sentinel string in the output — and the converse case (verdict printed,
then died) needs the exit code, so neither check is redundant.

Both halves have to be broken separately to know they work. Breaking one
leaves the other unevidenced, and the suite stays green either way.

### 10. A dependency fixing a bug can break code that adapted to it

`mighty-colab exec` used to exit 0 on failure. That was a bug, it was
fixed — and three targets relied on it, because `exec && ... && stop` tore
the session down *because* exec always succeeded. The adaptation was
invisible at the call site.

**Upgrading needs a check of what the old behaviour was load-bearing
FOR**, not just confirmation that the new behaviour is correct.

### 11. The execution model is not the one the code assumes

`exec -f script.py` transmits the file's **text** into an existing
IPython kernel. The script is never run as a script: `__file__` is
undefined, `__main__` is not the module, and **nothing from the
repository exists on that filesystem** until the driver clones it — which
happens inside `main()`, after every module-scope statement has already
run.

Ordinary local verification cannot catch this, because Python's import
machinery sets `__file__` correctly for a genuinely imported module. The
test that catches it reproduces the execution model: `compile()` +
`exec()` into a namespace with no `__file__`.

The general form: **when a runner's execution model differs from the
local one, the local test is a different experiment.** Reproduce the
model, not the code path.

### 12. Numerics are a property of the device

On A100, XLA computes float32 convolutions at TF32 by default — ~10
mantissa bits instead of 24. Normally a good trade; not one where a
downstream comparison has no tolerance band (this project's early
stopping is `min_delta=0.0` with a strict `<`, so a 1e-4 shift silently
moves which epoch is checkpointed and which seed is selected).

**A pass on incapable hardware is indistinguishable from a pass that
exercised the fix.** T4 has no TF32 hardware, reported clean agreement,
and would have closed the question as verified. Run device-numerics
checks on the device the pipeline actually targets, and report whether
the default path was *already* clean.

Same shape, different axis: ARM vs x86 encoding differs by up to 3 ULP
while two separate Colab sessions are bit-exact. The difference is
cross-architecture, not cross-session — and "cloud hardware varies too,
so local loses nothing" is a plausible-sounding claim that measurement
refutes.

---

## Part 3 — Artifacts as the durable interface

When agents cannot see each other and sessions do not persist, **the
artifact is the only thing that carries meaning between them.** Almost
every provenance decision in this project follows from that.

### 13. An artifact without provenance is a claim without a source

The first Phase A artifact recorded scientific parameters and timings and
nothing about the code that produced them — and ran from a dirty tree.
Which driver ran was recoverable after the fact by luck (one commit ever,
matching hash); which versions of the *other* participating modules were
live was not recoverable from any record.

A fingerprint is the union of the **static and runtime** import closures,
established before generation and revalidated after. Both, because
runtime alone misses conditional imports and static alone misses dynamic
ones — measured, they disagree in both directions.

### 14. Existence is not evidence of completeness

*An object merely existing is never sufficient evidence that it is
resumable.* A resumption that treats existence as proof will happily
consume an artifact produced under a different configuration.

The manifest is the **commit point**: payload written, its generation
captured, sidecar written second recording that generation. A payload
with no manifest is UNCOMMITTED — which is the semantic for the crash
window between the two writes, not merely a validity rule.

### 15. Immutability must be enforced, not documented

Pinning an artifact's generation guarantees the consumer reads what was
committed **only while that generation survives**. If any ordinary path
can overwrite it, the guarantee is a description of intent.

So lineage artifacts are create-once by construction, classification is
**fail-closed** (an undeclared kind is protected), and the lineage walk is
**transitive** — a mutable object three hops up is still mutable inside a
chain of digests claiming to be fixed. Publish-side and consume-side
checks are both needed: one binds what this code writes, the other what
it is willing to read, including manifests left by an older commit.

### 16. Scope the pre-flight to what can actually reach the computation

A whole-tree "is the repo clean?" check refuses correct runs when a
second agent has unrelated scratch in the same checkout — and it reports a
genuinely dirty *source* file as one line among many. Since the runtime
executes one pinned commit, the question that matters is whether the
**driver's own import closure** is committed.

Narrowing a guard is usually wrong. This one is sharper, not laxer, and
the distinction is worth stating: a guard that refuses correct work gets
switched off, and a guard that is switched off protects nothing.

### 17. Regeneration means a new name

The Phase A regeneration wrote `encoded_train_s1200` rather than
overwriting `encoded_fit_s1200`, which is why the 54,000 overlapping
images could be compared **bit-exactly against the baseline** — the
baseline still existed. Overwriting would have destroyed the only
evidence that the regeneration reproduced anything.

Reports too: a fixed report name overwritten by a resumed run destroys
the record of what the attempt that *died* had seen, which is exactly the
run you most want to read afterwards.

### 18. Join on identity, never on position

Cross-artifact comparison goes **by official index, never by positional
prefix**. `new[:54000]` returns an array of exactly the right shape and
dtype and yields a confident verdict about the wrong rows, with nothing
raised anywhere.

The guard that makes this real: refuse an alignment that turns out to
*be* a prefix. A join that silently accepts the prefix case cannot be
distinguished from never having joined.

---

## Part 4 — The human in the loop

### 19. Spend is released explicitly, and only by the human

Every agent in this system can propose GPU work. None may start it. The
release is a separate, explicit act, and "the plan was approved" is not
it. This is cheap to maintain and the failure it prevents is unbounded.

### 20. Ask only what changes the next action

Most decisions an agent wants to escalate have a defensible default.
Escalate the ones where different readings produce materially different
work — and when escalating, state a recommendation and the evidence, not
a menu. Two of this project's better calls came from a reviewer
overruling a recommendation that was *stated clearly enough to argue
with*.

### 21. The human sees things no agent can

A one-line suggestion — *"you can verify the config from `gcloud`"* —
overturned a finding four agents had accepted. The behavioural probe said
superseded bytes were gone; the config said they are retained for seven
days. Every agent involved had reasoned correctly from the evidence
available, and the evidence available was incomplete in a way none of
them could see from inside.

**When a human offers a different instrument, use it before defending the
measurement.**

### 22. Make the loop impossible before the first new fit exists

When a result reveals that a parameter's search range was wrong, the
tempting next move is to widen it and re-run. The danger is not widening
it once — it is the *loop*: widen, look, still at the boundary, widen
again. Each step is individually defensible and the sequence is
alpha-hacking-shaped, because the stopping rule is being chosen while
looking at the results.

Stage 3 hit this exactly. Six of seven ridge conditions selected the
grid-minimum alpha, asymmetrically across the treatment/control line, so
the optimum was unbracketed for precisely the conditions being compared.

**The decision rule gets frozen before any computation on the new range**:
the extension size pre-committed, the success criterion named in advance
(here: an interior argmin), and the answer decided ahead of time for what
happens if something pins at the *new* boundary too. Freezing it
afterwards is worthless — the whole value is that the rule could not have
been chosen to suit what was seen.

Generalises past alpha grids to any parameter whose range is revised
after seeing results. The reason to do it early is not suspicion of
anyone's honesty; it is that **the way never to face the accusation is to
make the loop structurally impossible before the first new fit exists.**

An adjacent discipline that made this one easier to hold: the situation
was not improvised. The reviewer had pre-registered the rule *"if several
conditions select the grid-minimum alpha, that is a reported fact
requiring scrutiny before the confirmatory stage — not a halt, but a
named review item."* When it fired, nobody had to decide in the moment
whether it mattered.

---

## The short version

1. Convergence is declared by the reviewer, never the responder.
2. Quote the frozen text; do not trust a summary of it, including your own.
3. Verify a subagent's arithmetic; trust only its retrieval.
4. Derive guard rails — the next agent has not read your list.
5. Commit before you think you need to; the session may not end, it may stop.
6. Tear down unconditionally, check the status, keep leak and verdict distinct.
7. Require a sentinel *and* an exit code; break each half separately.
8. Reproduce the execution model, not just the code path.
9. Run device-numerics checks on the device you actually target.
10. The artifact is the interface: fingerprint it, commit it atomically, never overwrite it.
11. Join on identity, never on position.
12. Spend is released by the human, explicitly, every time.

## Related

`CLAUDE.md` (methodological principles, especially 16 on reimplemented
helpers, 18 on extrapolating one stage's timing to another, 20 on
converting hand-verification into tests, and 21 on derived sets);
`docs/VACUOUS_TESTS.md` (the catalogue of tests that passed for the wrong
reason); `docs/PROJECT_MEMORY.md` Part 4 (the incident record these
patterns were derived from).
