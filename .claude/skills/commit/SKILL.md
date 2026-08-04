---
name: commit
description: >-
  Safely create scoped git commits in this repo. Use whenever the user asks to
  "commit", "commit this", "commit my changes", or similar, or before any
  commit you're about to make on your own initiative mid-task.
  Merges two suggestions from Claude Code `insights` (run once from this
  repo, once from the `mighty-colab` repo) after finding real problems with
  each in isolation -- see rationale below.
---

# Skill: Safe Scoped Commit

## Why this exists

Two things went wrong in the same session that led to this skill:

1. **A staged deletion from an unrelated source (IDE save/rename) got swept
   into a commit** because `git add <one file>` followed by a plain
   `git commit -m "..."` commits the *entire* index, not just what was just
   added. The diff check that would have caught it
   (`git diff --stat <that one file>`) was scoped too narrowly to see it.
2. **This project has a documented history of concurrent-session collisions**
   -- `CLAUDE.md` Part 4 records a Claude Code session dying silently
   mid-task with nothing recoverable, and this same session separately hit a
   `stagea2a`/`stage2a` branch mixup from two sessions editing in parallel.
   Committing without first checking whether *another* session already
   changed things risks compounding that.

## Steps

1. **Check for collisions first.** Run `git status --short` (full, not
   scoped to any one file) and `git log --oneline -5`. If either shows
   changes or commits this session did not make -- an unfamiliar staged
   file, a HEAD that's moved since you last checked, a branch that isn't
   what you expect -- **stop and report a possible concurrent-session
   collision** rather than committing over it. Investigate (whose change is
   it, is it in-progress work) before proceeding.

2. **Group changes into logical units.** One commit per logical change, not
   one giant commit for an entire session's work. If unsure where a
   boundary falls, prefer more/smaller commits over fewer/larger ones --
   easier to review, easier to revert independently.

3. **Stage explicit paths, never a blanket add.** `git add <path> <path>
   ...` for exactly the files in this logical unit -- never `git add -A` or
   `git add .`. This is what would have caught problem #1 above: staging
   explicit paths never accidentally includes a file you didn't intend.
   After staging, run a full `git status --short` (not a diff scoped to one
   file) to see everything that's about to be committed, including
   anything already sitting in the index from before this skill ran.

4. **Run tests conditionally, not always.** If the staged changes include
   `.py` files, run `uv run pytest tests/ -m "not slow"` (this project's
   established convention -- see `CLAUDE.md`) and don't commit if it fails.
   Skip entirely for doc-only/config-only changes -- running the suite on
   every commit regardless of content is wasted time on a research
   codebase with no CI gate forcing it.

5. **Run lint only if the project actually has it configured.** Check for
   a `ruff` config (`pyproject.toml`'s `[tool.ruff]`, or a `ruff.toml`)
   before running `ruff check --fix . && ruff format .`. This repo
   (`bonsai-2026`) currently has **no ruff config and no `ruff` on PATH** --
   running it unconditionally here would either error out or, if some
   global install got picked up, silently reformat the entire repo as a
   side effect of an unrelated commit. Skip this step outright when there's
   nothing configured to run against, rather than forcing it in.

6. **Write imperative-mood commit messages**, one per logical change,
   following this repo's existing convention (see `git log` for examples --
   summary line stating what changed and why, not what files were touched).
   Follow the global Claude Code git instructions already in force for this
   session (new commits over amends, no `--no-verify`, the `Co-Authored-By`
   footer, etc.) -- this skill doesn't repeat those, only adds the
   repo-specific steps above.

7. **Never push automatically.** Show `git log --oneline` for the new
   commit(s) and ask before pushing -- matches the standing instruction to
   never push without an explicit ask. Do not report commits as "pushed"
   unless a push actually happened after being asked for.
