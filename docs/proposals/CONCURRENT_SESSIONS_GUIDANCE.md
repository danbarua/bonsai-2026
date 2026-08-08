# Patch-in-waiting: concurrent-session discipline

**Targets:** `CLAUDE.md` (short, always-loaded rule) and
`docs/MULTI_AGENT_PRACTICE.md` (the pattern with its rationale).
**Status: APPLIED** (2026-08-08). Landed in `CLAUDE.md` under "Concurrent
sessions" and in `docs/MULTI_AGENT_PRACTICE.md` as patterns 6b and 6c, on
Dan's instruction, after both peer sessions reviewed and said land it. This
file is kept as the reasoning record — what was proposed, what review
changed, and what was deliberately dropped — none of which belongs in the
target documents themselves.
**Origin:** Dan's draft, after a night with three Claude Code sessions
working the same repository concurrently.

---

## 1. Dropped from this proposal: the `git add -A` rule

Dan's draft opened with "stage files explicitly by path (never `git add -A`
/ `git commit -a`)". **It is not proposed here**, and the reasoning is worth
recording because it nearly went in.

Worktrees have their own working directory *and their own index*. A
`git add -A` inside `.claude/worktrees/infra-tooling` cannot stage another
session's changes, because no other session works in that directory. Agents
in this repository now run in worktrees by default, so the *concurrency*
justification for the rule — the only justification a concurrent-sessions
section could offer — does not hold where the readers actually are.

Including it anyway would have contradicted this document's own closing
argument (§4): a guard that is wrong in the common case trains
route-around. A prose rule is not a guard, but the mechanism is the same —
a reader who sees a rule that is plainly inapplicable to their situation
discounts the section, including the two rules that *were* paid for.

**The commit skill's version of the rule is untouched and should stay.**
`.claude/skills/commit/SKILL.md` requires staging "exactly the files in this
logical unit", and its rationale is *scoped commits*, not collision
avoidance. That is a different claim, it survives worktrees intact, and it
belongs where it is.

One observation survives the cut, and it is the useful half: the INFRA
session violated the commit skill's rule on roughly a dozen commits on
2026-08-07/08 — not from disagreement, but because it committed directly
with `git commit -m` and never invoked `/commit`, so the file carrying the
rule was never loaded. **Guidance that lives only in a skill is invisible
unless the skill is invoked**, and the moment such a rule needs to fire is
exactly the moment someone is committing without ceremony. That is a real
placement problem, but it is the commit skill's to solve, not this
section's.

## 2. Corrections to the draft

### 2b. The observed collision was at PUSH time, not commit time

The draft checks `git status` and `git log -1` before committing. Nothing
tonight failed that way. What did fail, twice, was:

```
git push origin infra-tooling:stage2b
! [rejected]  (non-fast-forward)
```

`stage2b` had moved under the branch — once from the science track's alpha
freeze, once from the c2c track. A local `git log -1` cannot see that; only
`git fetch` can. The evidence-backed rule is therefore **fetch before
pushing to a shared branch**, and the commit-time check is the weaker half.

Review added incidents from the other two tracks: `c2c-implementation` hit
the same rejection twice in one session, and `stage2b-lead` once with origin
having moved **three** commits underneath. So this is at least five
occurrences in a night, spanning all three tracks — not one track's
clumsiness.

Both other tracks resolved it with `git merge origin/stage2b`, which works
but leaves merge commits on a shared release branch. Rebase is safe here for
a specific reason worth stating rather than asserting: **a rejected push
means those commits were never visible on origin**, so rewriting them cannot
invalidate anyone's view of the remote. One honest caveat, `c2c-implementation`'s:
worktrees share a single local object store, so another *local* session
could in principle have inspected those commits before the rebase rewrites
them. Below the bar for mechanising, above the bar for a half-sentence — so
the rule should not read "always safe, full stop."

### 2c. Tying the mail check to "before committing" is arbitrary

Commits are private until pushed. The moment that matters to other sessions
is when work *reaches* them. Tonight's two most valuable messages —
`stage2b-lead`'s field reports of a predicate under-capturing and
over-capturing — arrived mid-iteration and were acted on immediately; that
timing had nothing to do with commit boundaries.

Better triggers: **before starting a new increment** (so a peer's finding
redirects the work before it is built on) and **before pushing to shared
surface** (so consent and conflicts surface before, not after).

## 3. The part the draft is missing, and it is the load-bearing one

Dan's fourth question — "notify other agents when committed/pushed, where
work overlaps?" — is the highest-value item, and the record supports it
directly.

The provenance capture hooks shipped with two defects that a green test
suite did not catch. Both were found within the hour, in the field, by the
session they had been deployed into.

**The mechanism is not what the first draft of this section claimed.** That
draft said the reporting instruction "converted a silent defect into a bug
report." `stage2b-lead` reviewed it and cut the claim, reconstructing what
actually happened: they ran a canary because *they* wanted capture working
before their amendment audit — their own stake, not an invitation — and
would very likely have found and reported both defects regardless.

The true mechanism is better, and more transferable:

> Announcing did not open a channel so much as **publish a specification**.
> The announcement said what should and should not be captured — `git`
> excluded, heredocs-to-interpreters targeted. Without it, a commit message
> sitting in a provenance blob looks like the tool working as designed.
> There is no baseline against which to call it a defect. **The return on
> announcing is that a peer can recognise a deviation; a peer who only
> knows a tool exists can only recognise noise.**

That mechanism survives the case the incident did not test — a peer with no
independent motive. And the reviewer noted the direction of their own bias:
they were an unusually motivated reporter, so the channel's value is
probably *higher* for uninvested peers than this incident demonstrates.

So: **a tool that changes behaviour inside someone else's session must
publish what it is supposed to do, before it lands.** Not so the peer knows
it exists — so the peer can tell correct behaviour from a defect.

### 3a. Two kinds of shared surface, and only one propagates by itself

`c2c-implementation`'s refinement, from living it the same night: **some
shared-surface changes reach a reader automatically and some cannot.**

A git-tracked file — `CLAUDE.md`, `.claude/settings.json`, a hook script —
reaches every session the next time it reads that file. Announcing there
speeds things up but is not structurally required.

An MCP server's tool schema is not like that. A session's attached schema is
fixed at connection time and does not update on its own, confirmed
independently by two sessions on 2026-08-07/08 via different methods.
Deprecating `c2c-send`/`c2c-inbox` in the server source was a shared-surface
change that reached **nobody** until each peer reconnected; the code2code
mail announcing it was not a courtesy on top of a self-propagating change,
it was the only channel by which anyone would learn.

The same holds for hook registrations, which load once per session (see the
provenance capture README).

So distinguish **announcing because it is considerate** from **announcing
because there is no other channel by which the reader learns this.** In the
second case, skipping the announcement is not impolite — it is silently
wrong.

### 3b. What the notification bought that a commit message could not
`stage2b-lead` answered with a *better sequencing argument* than the one it
was given (their critical path was idle and the next compute was queued, so
waiting would have landed the change nearer a run, not further from it), and
with a *better design* than the one proposed (the `session_open` marker that
makes an absent record diagnostic rather than ambiguous). Announcing was not
courtesy; it was the mechanism by which the work got better.

Broadcast on every commit would be noise. The trigger is **shared surface**:
a config file two tracks write, a shared branch, or a change that alters
behaviour inside another session.

## 4. Proposed text

### 4a. `CLAUDE.md` — new short section, after "Running things"

```markdown
## Concurrent sessions

Other Claude Code sessions — including headless ones — may be working in
this repository at the same time. Two rules, both of which have been paid
for:

- **`git fetch` before pushing to a shared branch**, and rebase rather than
  merge. `stage2b` moved under in-flight branches at least five times in one
  night, across all three tracks. Rebasing is safe because a rejected push
  means those commits were never on origin — though worktrees share a local
  object store, so another local session could in principle have seen them.
- **Announce before you change shared surface** — a config file another
  track writes, a shared branch, or anything that alters behaviour inside
  another session — and publish what the change is *supposed* to do. Not so
  peers know it exists: so they can tell correct behaviour from a defect. A
  peer who only knows a tool exists can only recognise noise.
- **Some shared surface does not propagate.** A git-tracked file reaches
  readers when they next read it; an MCP tool schema and a hook registration
  are fixed at connection or session start and reach nobody until they
  reconnect or restart. For those, announcing is not courtesy — it is the
  only channel, and skipping it is silently wrong.

Full rationale and the incidents behind each: `docs/MULTI_AGENT_PRACTICE.md`
Part 1.
```

### 4b. `docs/MULTI_AGENT_PRACTICE.md` — new pattern in Part 1

```markdown
### 6b. Concurrent sessions share a repository, not a view of it

Several Claude Code sessions may hold the same repository open, in the main
checkout and in worktrees, with no way to see each other's state.

Notably absent: a rule against `git add -A`. Worktrees have private
indexes, so an agent working in one cannot stage another session's changes,
and agents here run in worktrees by default. The concurrency case for that
rule does not hold where the readers are. (`.claude/skills/commit/SKILL.md`
still requires staging by logical unit — a different claim, about scoped
commits rather than collisions, and unaffected by worktrees.)

**Fetch before pushing to a shared branch.** A local `git log` cannot see a
branch that moved on the remote. `stage2b` moved under an in-flight branch
twice on 2026-08-07/08 — once from the science track, once from the c2c
track — and both were caught only by a rejected push. Rebase onto the moved
branch and re-run the suite; a merge commit on a shared release branch is
harder to reason about later.

**Check mail before starting an increment, not before committing.** Commits
are private until pushed. A peer's finding is worth having before work is
built on it, not after it is recorded.

**Announce before changing shared surface, and open a reporting channel.**
Shared surface means a config file another track writes, a shared branch, or
any change that alters behaviour inside another session. Two returns on
this, both observed:

- The provenance capture hooks shipped with two defects a green suite did
  not catch. Both were found within the hour because the affected session
  had been told explicitly how to report a failure rather than route around
  it. A tool that changes behaviour inside someone else's session needs that
  channel opened *before* it lands, or its failures are absorbed as that
  session's problem instead of returning as evidence.
- Announcing produced better work, not just consent. The peer replied with a
  sharper sequencing argument than the one it was given, and with a better
  design — a session-start marker making an absent record diagnostic rather
  than ambiguous — than the one proposed to it.

Broadcast on every commit is noise. The trigger is shared surface.
```

### 4c. `docs/MULTI_AGENT_PRACTICE.md` — one line in "The short version"

```markdown
6b. Concurrent sessions share a repo, not a view of it: fetch before
    pushing shared branches, announce before changing shared surface —
    and say how to report it going wrong.
```

## 4d. `docs/MULTI_AGENT_PRACTICE.md` — second new pattern, authorized separately

Authorized by `claude-desktop-orchestrator` on 2026-08-08 as a standard step
for infrastructure deliverables.

```markdown
### 6c. A peer-use round before an infra deliverable is called done

Have a second instance use the tool on its own real work before declaring
it finished. Not review of the code — use of the tool.

The provenance capture hooks passed a suite of 967 tests and shipped with
four defects. Three were found by `stage2b-lead` using the hooks on real
work within an hour of them landing; the fourth (a version probe whose
command was never valid) was found by running it against the real binary
for the first time. None were found by testing.

This is not a testing-effort failure, and treating it as one predicts the
wrong fix. Every one of the four sat in the gap between "component correct"
and "wired correctly" — the same gap as principle 16, and now taxonomy
entry G in `docs/VACUOUS_TESTS.md`. **The builder cannot see wiring gaps
from inside the build**, for the same reason this project already runs
external review on its scientific claims: the author's model of what the
artifact does is the thing under test, and it cannot audit itself.

The peer-use round is that discipline applied to tooling rather than to
findings. It requires the peer to be told what the tool is *supposed* to
do (see 6b), or they can only report noise.
```

## 4e. The pattern behind two of the night's incidents

Proposed for `docs/VACUOUS_TESTS.md`, alongside taxonomy G, on
`stage2b-lead`'s observation:

> **The artifact is correct and never loaded.** A hook registration is
> invisible until session start; a rule that lives in a skill is invisible
> until the skill is invoked. Both are cases of the thing existing, being
> correct, and not reaching the moment it had to fire — and neither is
> detectable by testing the artifact, because the artifact is fine.
>
> Two independent instances in one night, from different mechanisms, is a
> pattern rather than a coincidence, and it predicts a third somewhere
> nobody has looked yet. The counter is the same in both cases: make the
> system emit positive evidence that it loaded, so absence is diagnostic.

## 5. What this does not propose

No hook enforcing any of it, and no `git add -A` rule (§1).

A `PreToolUse` guard on staging was considered and rejected on the same
ground that removed the prose rule: it would fire in worktrees where the
practice is safe, and a guard that is wrong in the common case trains
route-around — the argument that also keeps the provenance capture hooks
non-blocking.

Neither remaining rule is proposed for enforcement yet, for the reason this
repository applies to guards generally: a guard whose failure mode has not
been observed is a guess about what will go wrong.

But the two rules do not have the same future, and `stage2b-lead`'s
asymmetry is worth stating because it changes what "if broken again,
mechanise" means for each:

- **Rule 1's failure is loud.** A rejected push announces itself, and a
  pre-push check requiring a recent fetch would be cheap. Mechanising it is
  a live path if the rule keeps getting broken.
- **Rule 2's failure is silent, and cannot be mechanised at all.** Nothing
  can detect an announcement that was never made. There is no artifact whose
  absence is checkable, because the absent thing is a message nobody sent.

So rule 2 will always rest on practice. That is an argument for stating its
rationale well rather than tersely — a rule that can only be followed
voluntarily has to earn compliance by being understood, which is why §3's
mechanism was worth getting right rather than merely stated.
