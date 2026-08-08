#!/usr/bin/env python3
"""Assemble a briefing: what the mesh said, joined to what actually landed.

Mail carries intent; git carries what shipped. Reading either alone gives a
systematically wrong picture, and the failure has a direction: a mail-only
digest OVER-reports open loops, because a commit can close an item minutes
before a digest is written and go unmentioned simply because nobody has
mailed about it yet. That is not hypothetical -- it produced a wrong claim in
this project, where a finished remediation item was called half-done off a
digest whose window closed eleven minutes after the commit that closed it.

The asymmetry is structural, not accidental: the mailbox is gitignored, so
agents deliberately route durable content into commit messages. Git holds the
half the mailbox never sees.

Read-only. This reads digests, counts files in mailbox/ and archive/, and
runs `git log`. It writes nothing, moves nothing, and never calls
code2code-inbox -- that is a CONSUMING read which would archive mail
addressed to a live session.

Usage:
    assemble_briefing.py                        # cold: "I have just arrived"
    assemble_briefing.py --since-commit <sha>
    assemble_briefing.py --since-digest MAILBOX_SUMMARY_....md

All arguments are optional and caller-supplied. There is deliberately NO
server-side per-caller watermark: `instance:` names a role, not a session, so
per-instance state would fuse two sessions sharing a role and under-report for
one of them. The assembler is stateless; the caller says what it has seen.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Channels this assembles over. Derived from the directory layout rather than
# hand-listed capabilities -- see CLAUDE.md principle 21. A channel is a
# directory under .claude/ with a mailbox-summaries/ subdirectory.
CHANNEL_DIRS = ("code2code", "claude2gpt")

DIGEST_GLOB = "MAILBOX_SUMMARY_*.md"
CHECKPOINT = "checkpoint.txt"

DEFAULT_REF = "origin/stage2b"

# A digest filename carries 2026-08-08T14-48-02Z; git wants 14:48:02.
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})-(\d{2})-(\d{2})Z")

# Identifiers worth joining on: backticked code spans, and issue/PR numbers.
# Prose is not joinable -- an open loop phrased entirely in words gets
# reported as NOT CHECKED rather than as "no match found". Those are
# different claims, and collapsing them is the exact failure this tool
# exists to correct one layer down.
_BACKTICK_RE = re.compile(r"`([^`]+)`")
_ISSUE_RE = re.compile(r"(?:^|\s)(#\d+)\b")

# Below this length a token matches everything and discriminates nothing.
MIN_TOKEN_LEN = 4

# RARITY IS THE EVIDENCE. A loop is normally closed by one commit, so a token
# shared with many commits in the window is not a signal -- it is a common
# word that happens to sit in backticks. Measured on the first real run:
# `main` hit 11 of 31 commits, `stage2b-lead` (a session name) hit 7, and
# `gates.toml` hit 10, all pure noise, while the genuinely useful matches
# (`ci_targets.py`, `publish_review.sh`) hit exactly one each.
#
# This is deliberately ONE derived rule rather than a hand-maintained
# stopword list of common tokens and session names -- see CLAUDE.md principle
# 21. A list would silently under-cover the next noisy token; rarity cannot.
MAX_ABS_HITS = 3
# The absolute cap alone is meaningless on a short window (3 of 4 commits is
# not rare), so a fraction guards the small-n end.
LOW_SIGNAL_FRACTION = 0.25

# A backticked span is joinable if it is command-like -- bare words, paths,
# flags, make targets -- and not a prose quotation. `make stage2b-gate-inventory`
# must join; a quoted sentence must not.
_COMMANDLIKE_RE = re.compile(r"^[\w./#-]+(?: [\w./#-]+){0,3}$")

# A bullet already marked closed is not an open loop; reporting it under
# "no candidate found" would state the exact opposite of what it records.
_CLOSED_RE = re.compile(r"~~|\*\*CLOSED\*\*", re.IGNORECASE)

# Session names are never join evidence. A commit and a loop both mentioning
# a session is not evidence the commit closed the loop -- peers name each
# other constantly, in commit bodies and in digests alike.
#
# Rarity alone cannot catch this. On a long window a session name is common
# and gets suppressed; on a SHORT window it hits once, looks rare, and is
# promoted. Found by a cold consumer within an hour of the tool shipping:
# in a 3-commit window `stage2b-lead` matched a commit whose message merely
# mentioned them, and was offered as a candidate on two unrelated loops --
# "the join only matched it by name coincidence, not by tracing the fix".
#
# The names are DERIVED from the archive's own filenames, never hand-listed
# (CLAUDE.md principle 21): a list would miss the next session to join the
# mesh, and miss it silently.
_FROM_RE = re.compile(r"--from-([a-z0-9][a-z0-9-]*?)(?=--to-|\.md$)")
_TO_RE = re.compile(r"--to-([a-z0-9][a-z0-9-]*)\.md$")


def run_git(args: list[str], cwd: Path) -> str:
    """Run git, returning stdout. Raises on failure -- a silent git error
    would produce a confident briefing over an empty window."""
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


def repo_root(start: Path) -> Path:
    """Resolve the MAIN checkout, not the worktree.

    `--show-toplevel` returns the worktree when run from one, and a worktree
    has no mailbox. `--git-common-dir` returns the main checkout's .git, whose
    parent is the real root. `${CLAUDE_PROJECT_DIR:-.}` has the same failure:
    the `.` fallback is the worktree.
    """
    common = run_git(["rev-parse", "--git-common-dir"], cwd=start).strip()
    return (start / common).resolve().parent if not Path(common).is_absolute() \
        else Path(common).parent


def digest_ts_to_iso(name: str) -> str | None:
    """MAILBOX_SUMMARY_2026-08-08T14-48-02Z.md -> 2026-08-08T14:48:02Z."""
    m = _TS_RE.search(name)
    if not m:
        return None
    date, hh, mm, ss = m.groups()
    return f"{date}T{hh}:{mm}:{ss}Z"


@dataclass
class ChannelState:
    """Everything known about one channel's digest coverage."""

    name: str
    root: Path
    exists: bool = False
    digests: list[Path] = field(default_factory=list)
    unread: int = 0
    archived: int = 0
    undigested: int = 0
    undigested_names: list[str] = field(default_factory=list)

    @property
    def latest_digest(self) -> Path | None:
        return self.digests[-1] if self.digests else None


def _count_md(d: Path) -> int:
    return len(list(d.glob("*.md"))) if d.is_dir() else 0


def read_channel(root: Path, name: str) -> ChannelState:
    """Measure one channel. Counts only -- nothing is read destructively."""
    base = root / ".claude" / name
    st = ChannelState(name=name, root=base)
    if not base.is_dir():
        return st
    st.exists = True

    st.unread = _count_md(base / "mailbox")
    archive = base / "archive"
    st.archived = _count_md(archive)

    summaries = base / "mailbox-summaries"
    if summaries.is_dir():
        st.digests = sorted(summaries.glob(DIGEST_GLOB))

    # Derive the undigested set rather than trusting a count. This is the
    # same `comm -13` the summarise-mailbox skill uses, and for the same
    # reason: a message is archived under its ORIGINAL send-time filename, so
    # it can enter archive/ with a timestamp older than files already there.
    # A watermark silently drops exactly the messages that sat unread
    # longest -- which are the ones most likely to matter.
    ckpt = summaries / CHECKPOINT
    done: set[str] = set()
    if ckpt.is_file():
        done = {ln.strip() for ln in ckpt.read_text().splitlines() if ln.strip()}
    if archive.is_dir():
        present = {p.name for p in archive.glob("*.md")}
        st.undigested_names = sorted(present - done)
        st.undigested = len(st.undigested_names)
    return st


@dataclass
class Commit:
    sha: str
    short: str
    date: str
    author: str
    subject: str
    body: str
    files: list[str]

    @property
    def haystack(self) -> str:
        return "\n".join([self.subject, self.body, *self.files]).lower()


_REC = "\x1e"  # record separator -- safe against any character in a body
_FLD = "\x1f"


def read_commits(cwd: Path, rev_range: str, since: str | None) -> list[Commit]:
    """Read commits with bodies AND changed paths.

    Bodies matter more than usual here: this project routes durable reasoning
    into commit messages precisely because the mailbox is not committed.
    """
    # The separator LEADS the format. `--name-only` appends its file list
    # after the formatted fields, so a trailing separator would push each
    # commit's files into the NEXT record -- silently attributing every
    # changed path to the wrong commit, and corrupting the join haystack
    # rather than merely the display.
    fmt = _REC + _FLD.join(["%H", "%h", "%ad", "%an", "%s", "%b"])
    args = ["log", f"--format={fmt}", "--date=short", "--name-only"]
    if since:
        args.append(f"--since={since}")
    if rev_range:
        args.append(rev_range)
    out = run_git(args, cwd=cwd)

    commits: list[Commit] = []
    for chunk in out.split(_REC):
        chunk = chunk.strip("\n")
        if not chunk.strip():
            continue
        parts = chunk.split(_FLD)
        if len(parts) < 6:
            continue
        sha, short, date, author, subject, rest = parts[:6]
        # `--name-only` appends paths after the body, blank-line separated.
        body_lines: list[str] = []
        files: list[str] = []
        seen_blank = False
        for ln in rest.split("\n"):
            if not ln.strip():
                seen_blank = True
                continue
            # A path has no spaces and contains a separator or a known suffix.
            if seen_blank and ("/" in ln or "." in ln) and " " not in ln:
                files.append(ln)
            else:
                body_lines.append(ln)
        commits.append(
            Commit(
                sha=sha.strip(),
                short=short.strip(),
                date=date.strip(),
                author=author.strip(),
                subject=subject.strip(),
                body="\n".join(body_lines).strip(),
                files=files,
            )
        )
    return commits


@dataclass
class Loop:
    """One open-loop bullet from a digest, plus the advisory join result."""

    text: str
    tokens: list[str]
    matches: list[tuple[str, Commit]] = field(default_factory=list)
    low_signal: list[str] = field(default_factory=list)
    present: list[tuple[str, str]] = field(default_factory=list)
    name_tokens: list[str] = field(default_factory=list)

    @property
    def checked(self) -> bool:
        # A loop whose only identifiers were session names was not searched
        # for anything, so it belongs under NOT CHECKED rather than under
        # "searched, nothing matched" -- the same distinction this tool keeps
        # everywhere else.
        return len(self.tokens) > len(self.name_tokens)

    @property
    def one_line(self) -> str:
        flat = " ".join(self.text.split())
        return flat


def section(text: str, heading: str) -> str:
    """Return the body of a `## heading` section, up to the next `## `."""
    lines = text.split("\n")
    out: list[str] = []
    inside = False
    for ln in lines:
        if ln.startswith("## "):
            if inside:
                break
            inside = ln[3:].strip().lower().startswith(heading.lower())
            continue
        if inside:
            out.append(ln)
    return "\n".join(out)


def bullets(body: str) -> list[str]:
    """Split a section body into top-level `- ` bullets, keeping wrapped
    continuation lines with their bullet."""
    items: list[str] = []
    cur: list[str] = []
    for ln in body.split("\n"):
        if ln.startswith("- "):
            if cur:
                items.append("\n".join(cur))
            cur = [ln[2:]]
        elif cur and (ln.startswith("  ") or ln.startswith("\t")):
            cur.append(ln.strip())
        elif cur and not ln.strip():
            items.append("\n".join(cur))
            cur = []
    if cur:
        items.append("\n".join(cur))
    return [i for i in items if i.strip()]


def tokens_of(text: str) -> list[str]:
    """Joinable identifiers in a loop: backticked spans and issue numbers."""
    found: list[str] = []
    for raw in _BACKTICK_RE.findall(text):
        tok = raw.strip()
        if len(tok) < MIN_TOKEN_LEN:
            continue
        # A backticked quotation is prose, not an identifier -- but a make
        # target is two words and must still join. Rejecting every span with
        # a space put `make stage2b-gate-inventory` in NOT CHECKED on the
        # first real run, which reads as "unsearchable" for one of the most
        # searchable strings in the digest.
        if not _COMMANDLIKE_RE.match(tok):
            continue
        found.append(tok)
    found.extend(_ISSUE_RE.findall(text))
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    return [t for t in found if not (t.lower() in seen or seen.add(t.lower()))]


def session_names(root: Path) -> set[str]:
    """Every session name the mesh has ever used, derived from filenames.

    Message filenames carry `--from-<name>` and optionally `--to-<name>`, so
    the roster is a property of the corpus rather than a list somebody has to
    remember to update when a session joins.
    """
    names: set[str] = set()
    for channel in CHANNEL_DIRS:
        base = root / ".claude" / channel
        for sub in ("archive", "mailbox"):
            d = base / sub
            if not d.is_dir():
                continue
            for p in d.glob("*.md"):
                for rx in (_FROM_RE, _TO_RE):
                    m = rx.search(p.name)
                    if m:
                        names.add(m.group(1).lower())
    return names


def join_loops(
    loops: list[Loop], commits: list[Commit], names: set[str] | None = None
) -> None:
    """Attach candidate closing commits to each loop. ADVISORY ONLY.

    Matching a commit to a loop is a judgement, not a derivation. A commit
    that mentions an identifier may well be unrelated to the loop that also
    mentions it. This narrows a reader's search; it does not decide anything.
    """
    n = len(commits) or 1
    names = names or set()
    for loop in loops:
        for tok in loop.tokens:
            # A session name is never evidence, at any window size.
            if tok.lower() in names:
                loop.name_tokens.append(tok)
                continue
            hits = [c for c in commits if tok.lower() in c.haystack]
            if not hits:
                continue
            # Rarity is the evidence -- see MAX_ABS_HITS. Both bounds apply:
            # the absolute one carries long windows, the fractional one short.
            #
            # A SINGLE hit is always discriminating and is never suppressed.
            # Without that clause the fraction punished small windows for
            # being small: one hit in three commits is 0.33, over the 0.25
            # bound, so a unique genuine match was discarded and the loop
            # filed under "no candidate found". That fires exactly when the
            # digests catch up and the cold window shrinks to a handful of
            # commits -- the normal steady state, not an edge case.
            crowded = len(hits) > 1 and len(hits) / n > LOW_SIGNAL_FRACTION
            if len(hits) > MAX_ABS_HITS or crowded:
                loop.low_signal.append(tok)
                continue
            for c in hits:
                loop.matches.append((tok, c))


_MAKE_TARGET_RE = re.compile(r"^make ([\w./-]+)$")
_PATHISH_RE = re.compile(r"[/]|\.(py|sh|ts|md|toml|ya?ml|json|txt)$")


def tree_evidence(root: Path, loop: Loop) -> None:
    """Ask the CURRENT TREE whether a loop's named artifact now exists.

    The git join is window-bounded, so a loop closed by an older commit reads
    as open -- measured, not hypothetical: `make stage2b-gate-inventory` was
    reported "no candidate found, most likely genuinely open" while the target
    sat at Makefile:655, added before the window opened.

    This check has no window. It is also the right authority for the question:
    a commit date says when a file was last touched, not whether its content
    is current -- `git log` arbitrates WRONGLY between two disagreeing
    documents when the staler claim happens to sit in the later commit.

    Still ADVISORY. That a named target exists does not prove the loop that
    named it is closed; it proves the artifact is no longer absent.
    """
    for tok in loop.tokens:
        m = _MAKE_TARGET_RE.match(tok)
        if m:
            mk = root / "Makefile"
            if mk.is_file():
                target = m.group(1)
                for i, ln in enumerate(mk.read_text().splitlines(), 1):
                    if ln.startswith(f"{target}:"):
                        loop.present.append((tok, f"Makefile:{i}"))
                        break
            continue
        if _PATHISH_RE.search(tok):
            cand = root / tok
            if cand.exists():
                loop.present.append((tok, tok))


def fmt_coverage(
    channels: list[ChannelState],
    ref: str,
    head_sha: str,
    window_desc: str,
    n_commits: int,
) -> str:
    """The coverage boundary. FIRST, never buried.

    Without this the briefing silently inherits digest lag and reproduces, one
    layer up, the exact failure it was built to correct.
    """
    out = ["## Coverage boundary", ""]
    out.append("What this can and cannot see. Read before trusting anything below.")
    out.append("")
    out.append("| Channel | Latest digest | Unread | Archived | **Undigested** |")
    out.append("|---|---|---|---|---|")
    for ch in channels:
        if not ch.exists:
            out.append(f"| `{ch.name}` | *channel not present* | — | — | — |")
            continue
        latest = ch.latest_digest.name if ch.latest_digest else "*none*"
        flag = f"**{ch.undigested}**" if ch.undigested else "0"
        out.append(
            f"| `{ch.name}` | `{latest}` | {ch.unread} | {ch.archived} | {flag} |"
        )
    out.append("")

    stale = [c for c in channels if c.exists and c.undigested]
    if stale:
        for ch in stale:
            out.append(
                f"> **`{ch.name}` is {ch.undigested} messages behind.** Nothing in "
                f"them appears below. `/summarise-mailbox` closes the gap."
            )
        out.append("")
    else:
        out.append("Every archived message is covered by a digest.")
        out.append("")

    unread = [c for c in channels if c.exists and c.unread]
    if unread:
        for ch in unread:
            out.append(
                f"> `{ch.name}` has {ch.unread} message(s) still in `mailbox/` — not yet "
                f"received by anyone, and deliberately not summarised."
            )
        out.append("")

    out.append(f"**Git:** `{ref}` at `{head_sha[:12]}`, window {window_desc} "
               f"({n_commits} commit{'s' if n_commits != 1 else ''}).")
    out.append("")
    return "\n".join(out)


def fmt_rulings(gpt: ChannelState) -> str:
    out = ["## Rulings in force", ""]
    if not gpt.exists or not gpt.latest_digest:
        out.append("*No c2gpt digest available.*")
        out.append("")
        return "\n".join(out)

    text = gpt.latest_digest.read_text()
    out.append(
        "> **Provenance grade.** These record what the file SAYS, not how it "
        "arrived. **Authority: intact** — Dan is the release gate. "
        "**Fidelity: unverified** — `from:` is a routing parameter, not an "
        "attestation, and a hand-paste can clip while reading complete."
    )
    out.append("")
    out.append(f"From `{gpt.latest_digest.name}`:")
    out.append("")
    body = section(text, "Decisions reached")
    items = bullets(body)
    if items:
        for it in items:
            out.append(f"- {' '.join(it.split())}")
    else:
        out.append("*No decisions recorded in the latest c2gpt digest.*")
    out.append("")
    return "\n".join(out)


def fmt_landed(commits: list[Commit], window_desc: str) -> str:
    out = ["## What landed", ""]
    if not commits:
        out.append(
            f"**Nothing landed in this window** ({window_desc}). This is a real "
            f"measured empty result, not a failure to look."
        )
        out.append("")
        return "\n".join(out)

    out.append(f"{len(commits)} commits, newest first.")
    out.append("")
    for c in commits:
        out.append(f"- **`{c.short}`** {c.subject}  ")
        out.append(f"  <sub>{c.date} · {c.author} · {len(c.files)} file(s)</sub>")
    out.append("")
    return "\n".join(out)


def fmt_loops(loops: list[Loop], src: str | None, window_desc: str) -> str:
    out = ["## Open loops", ""]
    if src is None:
        out.append("*No code2code digest available — no open loops to report.*")
        out.append("")
        return "\n".join(out)

    out.append(f"Carried from `{src}`, each checked against git. ")
    out.append("")
    # A cold reader adopted a role it saw named here and reported another
    # session's blocker as its own. The briefing is not addressed to anyone,
    # and has no way to know who is reading it -- so it says so, rather than
    # guessing ownership from a name it happens to match. Deriving "yours"
    # from a mention would be the same weak inference just removed from the
    # join: a loop naming two sessions names neither as its owner.
    # Rules, not paragraphs. Measured finding from a cold-read audit: a
    # reader retained one-to-three-sentence principles and re-derived the
    # essay-length ones from source instead, twenty minutes after reading
    # them. Every claim below is load-bearing and kept; only the prose around
    # them is gone. In particular "not proof it is open" stays -- that line
    # is what stopped a resolved red build being escalated as an emergency.
    out.append("> **Addressed to nobody.** A name here is not an assignment to you.")
    out.append(">")
    out.append(
        f"> **The join is ADVISORY.** A match is a judgement, not a "
        f"derivation. Candidates narrow your search; they close nothing."
    )
    out.append(">")
    out.append(
        f"> **Window: {window_desc}.** A loop closed by an older commit still "
        f"shows as open."
    )
    out.append("")

    if not loops:
        out.append("*The latest digest lists no open loops.*")
        out.append("")
        return "\n".join(out)

    likely = [l for l in loops if l.matches]
    in_tree = [l for l in loops if not l.matches and l.present]
    unmatched = [l for l in loops if l.checked and not l.matches and not l.present]
    unchecked = [l for l in loops if not l.checked]

    if likely:
        out.append(f"### Possibly closed — {len(likely)} to verify")
        out.append("")
        for l in likely:
            out.append(f"- {l.one_line}")
            seen: set[str] = set()
            for tok, c in l.matches:
                key = c.sha
                if key in seen:
                    continue
                seen.add(key)
                out.append(f"  - candidate `{c.short}` — *{c.subject}* (matched `{tok}`)")
            if l.low_signal:
                out.append(
                    f"  - <sub>ignored as non-discriminating: "
                    f"{', '.join('`' + t + '`' for t in l.low_signal)}</sub>"
                )
            if l.name_tokens:
                out.append(
                    f"  - <sub>ignored as session names: "
                    f"{', '.join('`' + t + '`' for t in l.name_tokens)}</sub>"
                )
        out.append("")

    if in_tree:
        out.append(f"### Named artifact now EXISTS — {len(in_tree)} to verify")
        out.append("")
        out.append(
            "Nothing matched in the window, but the named thing is in the tree now. "
            "This check has no window: it asks the tree, not the log."
        )
        out.append("")
        for l in in_tree:
            out.append(f"- {l.one_line}")
            for tok, where in l.present:
                out.append(f"  - `{tok}` is present at `{where}`")
        out.append("")

    if unmatched:
        out.append(f"### No candidate found — {len(unmatched)}")
        out.append("")
        out.append(
            "Searched the window and the tree; nothing matched. "
            "**This is not proof the loop is open** (principle 2)."
        )
        out.append("")
        for l in unmatched:
            out.append(f"- {l.one_line}")
        out.append("")

    if unchecked:
        out.append(f"### NOT CHECKED — {len(unchecked)}")
        out.append("")
        out.append(
            "No joinable identifier. **Not searched at all** — a different "
            "status from searched-and-found-nothing above."
        )
        out.append("")
        for l in unchecked:
            out.append(f"- {l.one_line}")
        out.append("")

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Assemble a mesh briefing from mail digests joined to git."
    )
    ap.add_argument("--since-commit", metavar="SHA",
                    help="Only report commits after this SHA.")
    ap.add_argument("--since-digest", metavar="FILENAME",
                    help="Treat this digest as already seen; window starts at its timestamp.")
    ap.add_argument("--ref", default=DEFAULT_REF,
                    help=f"Git ref to read (default: {DEFAULT_REF}).")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="Override repo root (default: derived via --git-common-dir).")
    args = ap.parse_args()

    here = Path.cwd()
    try:
        root = args.repo_root.resolve() if args.repo_root else repo_root(here)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    channels = [read_channel(root, n) for n in CHANNEL_DIRS]
    by_name = {c.name: c for c in channels}
    c2c = by_name["code2code"]
    gpt = by_name["claude2gpt"]

    # Resolve the git window.
    try:
        head_sha = run_git(["rev-parse", args.ref], cwd=root).strip()
    except RuntimeError as e:
        print(f"error: cannot resolve --ref {args.ref}: {e}", file=sys.stderr)
        return 2

    rev_range = args.ref
    since: str | None = None
    if args.since_commit:
        rev_range = f"{args.since_commit}..{args.ref}"
        # Only abbreviate an actual SHA -- truncating a ref name printed
        # `origin/stage..origin/stage2b`, which names no such ref.
        shown = args.since_commit
        if re.fullmatch(r"[0-9a-f]{7,40}", shown):
            shown = shown[:12]
        window_desc = f"`{shown}..{args.ref}`"
    else:
        # Cold default: since the latest digest. This directly answers the
        # question the digests cannot -- "what landed that no digest knows
        # about" -- which is the over-reporting failure this tool corrects.
        anchor = args.since_digest
        if not anchor and c2c.latest_digest:
            anchor = c2c.latest_digest.name
        since = digest_ts_to_iso(anchor) if anchor else None
        if since:
            window_desc = f"since `{anchor}` ({since})"
        else:
            window_desc = f"all of `{args.ref}` (no digest to anchor on)"

    try:
        commits = read_commits(root, rev_range, since)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Open loops from the latest code2code digest.
    loops: list[Loop] = []
    loop_src: str | None = None
    if c2c.latest_digest:
        loop_src = c2c.latest_digest.name
        body = section(c2c.latest_digest.read_text(), "Open loops")
        for b in bullets(body):
            # Skip the standing caveat / carried-forward preamble bullets.
            if b.lower().startswith("standing caveat"):
                continue
            # A digest records closures inline (strikethrough, **CLOSED**).
            # Those are not open loops. Carrying one through would file it
            # under "no candidate found -- most likely genuinely open",
            # stating the exact opposite of what the digest recorded.
            if _CLOSED_RE.search(b):
                continue
            loops.append(Loop(text=b, tokens=tokens_of(b)))
        join_loops(loops, commits, session_names(root))
        for l in loops:
            tree_evidence(root, l)

    ts = run_git(["log", "-1", "--format=%ad", "--date=iso-strict", args.ref],
                 cwd=root).strip()

    parts = [
        f"# Mesh briefing",
        "",
        f"Assembled from mail digests joined to `{args.ref}` (tip dated {ts}).",
        "",
        fmt_coverage(channels, args.ref, head_sha, window_desc, len(commits)),
        fmt_rulings(gpt),
        fmt_landed(commits, window_desc),
        fmt_loops(loops, loop_src, window_desc),
    ]
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
