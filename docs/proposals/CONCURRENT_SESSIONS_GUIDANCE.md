# Patch-in-waiting: concurrent-session discipline

**Targets:** `CLAUDE.md` (short, always-loaded rule) and
`docs/MULTI_AGENT_PRACTICE.md` (the pattern with its rationale).
**Status:** proposal. Both are shared documents the science and c2c tracks
also edit, so the infrastructure track proposes rather than edits.
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
suite did not catch. They were found within the hour, in the field, because
the hooks were deployed into another session's real work *and that session
had been told, explicitly, how to report a failure*: "if `.provenance/runs/`
populates from `make`, `pytest`, or a committed driver, that is a bug in my
predicate — send me the command rather than working around it."

That instruction is what converted a silent defect into a bug report. Both
defects were then fixed and merged in a single iteration, and the peer was
never blocked. The lesson is not "testing is futile" — it is that **a tool
that changes behaviour inside someone else's session needs a deliberate
reporting channel, opened before it lands, or its failures are absorbed as
that session's problem instead of returning as evidence.**

Note also what the notification bought that a commit message could not:
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
  merge. `stage2b` moved under an in-flight branch twice in one night.
- **Announce before you change shared surface** — a config file another
  track writes, a shared branch, or anything that alters behaviour inside
  another session — and say how to report it going wrong. This is not
  courtesy: it is how defects come back as bug reports instead of being
  absorbed silently by whoever they hit.

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

## 5. What this does not propose

No hook enforcing any of it, and no `git add -A` rule (§1).

A `PreToolUse` guard on staging was considered and rejected on the same
ground that removed the prose rule: it would fire in worktrees where the
practice is safe, and a guard that is wrong in the common case trains
route-around — the argument that also keeps the provenance capture hooks
non-blocking.

The two rules that remain are both enforceable in principle — a pre-push
check could require a recent fetch — and neither is proposed for
enforcement yet, for the reason this repository applies to guards
generally: a guard whose failure mode has not been observed is a guess
about what will go wrong. Both rules were written from incidents that
*did* happen; if they are broken again after being documented, that
recurrence is the evidence that would justify mechanising them.
