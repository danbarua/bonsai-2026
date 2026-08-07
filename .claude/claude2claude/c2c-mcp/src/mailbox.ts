import { promises as fs, readFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

// This file lives at <repoRoot>/.claude/claude2claude/c2c-mcp/src/mailbox.ts
// (or dist/mailbox.js once built) -- the repo root is four levels up from
// either src/ or dist/, since both sit at the same depth under c2c-mcp/.
function defaultRepoRoot(): string {
  const here = path.dirname(fileURLToPath(import.meta.url));
  return path.resolve(here, "..", "..", "..", "..");
}

export const REPO_ROOT = process.env.BONSAI_PROJECT_ROOT
  ? path.resolve(process.env.BONSAI_PROJECT_ROOT)
  : defaultRepoRoot();

// Read once from package.json (one level up from src/ or dist/, same as
// this file) rather than hardcoded in server.ts/index.ts -- found live: a
// hardcoded "0.1.0" in server.ts's McpServer constructor meant a version
// bump there silently had no effect on what a running (but not rebuilt)
// server actually advertised, which is exactly what happened here (dist/
// was stale after an src/-only edit, so both the MCP version AND the new
// code-sessions tool were invisible to a client until a real rebuild).
// Reading the single source of truth at runtime instead means a stale
// build now visibly reports its OWN (old) version rather than silently
// echoing back whatever string was last typed into server.ts.
function readPackageVersion(): string {
  const here = path.dirname(fileURLToPath(import.meta.url));
  try {
    const raw = readFileSync(path.join(here, "..", "package.json"), "utf8");
    const version = JSON.parse(raw).version;
    return typeof version === "string" ? version : "0.0.0";
  } catch {
    return "0.0.0";
  }
}

export const PKG_VERSION = readPackageVersion();

export interface Channel {
  id: string;
  inbox: string;
  outbox: string;
  archive: string;
}

function makeChannel(id: string, dirName: string): Channel {
  const root = path.join(REPO_ROOT, ".claude", dirName);
  return {
    id,
    inbox: path.join(root, "inbox"),
    outbox: path.join(root, "outbox"),
    archive: path.join(root, "archive"),
  };
}

export const CHANNELS = {
  c2c: makeChannel("c2c", "claude2claude"),
  c2gpt: makeChannel("c2gpt", "claude2gpt"),
} as const;

async function ensureDir(dir: string): Promise<void> {
  await fs.mkdir(dir, { recursive: true });
}

function pad(n: number): string {
  return String(n).padStart(2, "0");
}

// YYYY-MM-DDTHH-MM-SSZ, matching the c2c protocol's filename convention.
function timestampFilename(date: Date): string {
  return (
    `${date.getUTCFullYear()}-${pad(date.getUTCMonth() + 1)}-${pad(date.getUTCDate())}` +
    `T${pad(date.getUTCHours())}-${pad(date.getUTCMinutes())}-${pad(date.getUTCSeconds())}Z`
  );
}

// Standard ISO 8601 (with seconds precision) for the in-body header comment.
function timestampIso(date: Date): string {
  return date.toISOString().replace(/\.\d{3}Z$/, "Z");
}

// Slugifies a Claude Code session name (which may contain spaces or other
// non-filename-friendly characters -- session names aren't restricted to
// the kebab-case convention some users happen to follow) for use as a
// filename component: lowercase, non-alphanumeric runs collapsed to a
// single hyphen, leading/trailing hyphens stripped. Returns "" (not a
// bare "-") if nothing alphanumeric survives, so the caller can tell
// "no usable slug" apart from "a slug that happens to be a hyphen."
function slugify(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export interface SendResult {
  filename: string;
  path: string;
}

/**
 * Writes a new message file to `dir` (the caller picks which -- a channel's
 * outbox or inbox -- based on who's sending; see server.ts's role mapping).
 * Filename collisions (two sends in the same second) are resolved with a
 * `-2`, `-3`, ... suffix, using an atomic exclusive-create so concurrent
 * callers can't race each other onto the same filename.
 *
 * `instance` is the sending Claude Code session's `/rename`-set name (never
 * its raw session_id -- that's an opaque GUID, not useful to a reader; see
 * `.claude/hooks/c2c-mail/pre-c2c-mcp.sh`, which resolves and surfaces this
 * name to the model before every c2c-mcp call specifically so it can be
 * passed here). When set, it's appended to the header as `· instance:
 * <name>` (parsed back out by `parseInstance` below, exact/unslugified)
 * AND, slugified, to the FILENAME itself -- e.g.
 * `2026-08-07T21-30-00Z-c2c-implementation.md` -- so a reader can filter
 * by sender with a plain `ls`/glob, no file reads needed. Session names
 * aren't restricted to filename-safe characters (spaces are allowed), so
 * the filename component is always slugified (`slugify` above); the
 * header keeps the exact original name for display. Appended AFTER the
 * timestamp, never before, so filenames still sort chronologically first
 * -- `readMailbox`/`c2c_list_unread`'s oldest-first ordering depends on
 * plain lexicographic sort staying aligned with time. Distinct from `to`
 * (below): `instance` says who SENT it, `to` says who it's addressed to.
 *
 * `to` is optional addressing (a specific session's `/rename`-set name,
 * e.g. from the `code-sessions` tool): when set, it's appended to the
 * header as `· to: <name>`, parsed back out by `parseAddressee` below.
 * Omitted (the default -- every message sent before this existed has no
 * `to:` field) means broadcast: any reader may consume it, matching the
 * exact behavior before addressing existed.
 */
export async function sendMessage(
  dir: string,
  sender: string,
  content: string,
  to?: string,
  instance?: string,
): Promise<SendResult> {
  await ensureDir(dir);
  const now = new Date();
  const instanceSlug = instance ? slugify(instance) : "";
  const base = timestampFilename(now) + (instanceSlug ? `-${instanceSlug}` : "");
  const instancePart = instance ? ` · instance: ${instance}` : "";
  const toPart = to ? ` · to: ${to}` : "";
  const header = `<!-- from: ${sender} · ${timestampIso(now)}${instancePart}${toPart} -->\n\n`;
  const body = content.endsWith("\n") ? content : `${content}\n`;
  const data = header + body;

  let counter = 1;
  for (;;) {
    const filename = counter === 1 ? `${base}.md` : `${base}-${counter}.md`;
    const filePath = path.join(dir, filename);
    try {
      await fs.writeFile(filePath, data, { encoding: "utf8", flag: "wx" });
      return { filename, path: filePath };
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === "EEXIST") {
        counter += 1;
        continue;
      }
      throw err;
    }
  }
}

export interface InboxMessage {
  filename: string;
  content: string;
  to?: string;
  instance?: string;
}

// Extracts the optional "to: <name>" addressee from a message's header
// comment (its first line, as written by sendMessage above). Undefined
// means broadcast -- either no to: field at all, or a header this project
// didn't generate (e.g. a hand-typed message via the c2c skill) with
// nothing matching. Tolerant of trailing text after the name (stops at the
// first `-->` or `·`) rather than requiring an exact byte-for-byte match to
// sendMessage's own format.
//
// ANCHORED to a preceding `·` (or start of string) -- NOT a bare substring
// search. Found live: an unanchored /to:\s*(\S+)/ matches "to:" wherever it
// appears, including inside another field's VALUE (e.g. a hypothetical
// future "branch: auto-import-photo:staging" segment would misparse as
// `to: staging`, silently hijacking the addressing/instance mechanism this
// whole feature exists to make reliable). No such field exists in the
// header today, but the parser shouldn't depend on that staying true --
// anchoring on the `·` delimiter that actually separates fields is what
// makes this safe regardless of what content later fields carry.
export function parseAddressee(content: string): string | undefined {
  const firstLine = content.split("\n", 1)[0];
  const beforeClose = firstLine.split("-->", 1)[0];
  const match = /(?:^|·)\s*to:\s*(\S+)/i.exec(beforeClose);
  return match ? match[1] : undefined;
}

// Extracts the optional "instance: <name>" sender identity from a
// message's header comment, the same way parseAddressee extracts "to:".
// Undefined means the sender didn't provide one (older messages, or a
// sender role -- claude-desktop, chatgpt -- that isn't itself
// multi-instance in the way Claude Code sessions are). Anchored on `·` for
// the same reason parseAddressee is -- see its comment.
export function parseInstance(content: string): string | undefined {
  const firstLine = content.split("\n", 1)[0];
  const beforeClose = firstLine.split("-->", 1)[0];
  const match = /(?:^|·)\s*instance:\s*(\S+)/i.exec(beforeClose);
  return match ? match[1] : undefined;
}

/**
 * Reads every `.md` file in `sourceDir` (the caller picks inbox or outbox
 * based on whose messages the reader wants -- see server.ts's role
 * mapping), oldest first (filenames sort chronologically). When `archive`
 * is true (the default, matching the existing c2c protocol), each file is
 * moved to `archiveDir` after being read, so a later call only returns
 * messages nobody has processed yet.
 *
 * `asName`, if given, is this reader's own addressed identity (e.g. a
 * Claude Code session's `/rename`-set name). It only changes behavior on
 * the *consuming* path (`archive: true`): a message addressed to a
 * DIFFERENT name is skipped entirely -- neither returned nor archived --
 * so it's left untouched for its actual addressee rather than accidentally
 * consumed by the wrong reader. A peek (`archive: false`) always returns
 * everything regardless of addressing, since peeking doesn't consume
 * anything and hiding mailbox state during a peek would make the mailbox
 * harder to reason about, not safer. Omitting `asName` entirely preserves
 * the exact pre-addressing behavior (every message visible and
 * archivable) -- callers that don't know their own name (Desktop, ChatGPT,
 * or an un-updated caller) are unaffected.
 */
export interface ReadMailboxResult {
  messages: InboxMessage[];
  // Filenames left in place, unarchived, because they're addressed to a
  // different name than asName -- always empty unless archive && asName.
  skipped: string[];
}

export async function readMailbox(
  sourceDir: string,
  archiveDir: string,
  archive: boolean,
  asName?: string,
): Promise<ReadMailboxResult> {
  await ensureDir(sourceDir);
  const entries = await fs.readdir(sourceDir);
  const filenames = entries.filter((f) => f.endsWith(".md")).sort();

  const messages: InboxMessage[] = [];
  const skipped: string[] = [];
  for (const filename of filenames) {
    const filePath = path.join(sourceDir, filename);
    const content = await fs.readFile(filePath, "utf8");
    const to = parseAddressee(content);
    const instance = parseInstance(content);

    if (archive && asName && to && to !== asName) {
      skipped.push(filename); // addressed to someone else: leave it in place, unconsumed
      continue;
    }

    messages.push({ filename, content, to, instance });
    if (archive) {
      await ensureDir(archiveDir);
      await fs.rename(filePath, path.join(archiveDir, filename));
    }
  }
  return { messages, skipped };
}

// The Claude Code CLI itself (not this server) writes one JSON file per
// running/recently-run process here, keyed by PID -- confirmed by direct
// inspection on this machine, not from any documented API. Overridable so
// tests can point at a throwaway directory instead of real global state
// (this project has been bitten before by a test writing into a real,
// shared registry -- see the OAuth client-registry equivalent).
export const SESSIONS_DIR = process.env.CLAUDE_SESSIONS_DIR
  ? path.resolve(process.env.CLAUDE_SESSIONS_DIR)
  : path.join(os.homedir(), ".claude", "sessions");

export interface CodeSessionInfo {
  sessionId: string;
  name: string;
  cwd: string;
  status: string;
  pid: number;
  jobId: string;
  updatedAt: string; // ISO 8601, "" if the source file had no valid updatedAt
  alive: boolean;
}

// process.kill with signal 0 sends nothing -- it only probes whether the PID
// exists and is signalable. Checked directly rather than trusting the JSON
// file's own `status` field, since a crashed or killed session leaves its
// file behind unmodified (confirmed live on this machine: one found session
// file was hours stale with no messagingSocketPath, i.e. already dead).
function isProcessAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    // EPERM means the process exists but is owned by another user -- still
    // alive, just not signalable by us. Any other error (ESRCH, etc.) means
    // no such process.
    return (err as NodeJS.ErrnoException).code === "EPERM";
  }
}

/**
 * Lists Claude Code sessions known to this machine, filtered to those whose
 * `cwd` falls under `repoRoot` (the main checkout or any of its
 * `.claude/worktrees/*`). The filter is not cosmetic: SESSIONS_DIR is
 * global across every project on the machine, so an unfiltered list would
 * surface unrelated sessions (a different repo entirely) to any MCP client
 * connected to this per-repo server. Malformed or mid-write JSON files are
 * skipped individually rather than failing the whole call -- the CLI can be
 * writing a new one at the exact moment this reads the directory.
 */
export async function listCodeSessions(repoRoot: string): Promise<CodeSessionInfo[]> {
  let entries: string[];
  try {
    entries = await fs.readdir(SESSIONS_DIR);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw err;
  }

  const sessions: CodeSessionInfo[] = [];
  for (const entry of entries.filter((f) => f.endsWith(".json"))) {
    let raw: string;
    try {
      raw = await fs.readFile(path.join(SESSIONS_DIR, entry), "utf8");
    } catch {
      continue; // removed between readdir and readFile -- benign race
    }
    let parsed: Record<string, unknown>;
    try {
      parsed = JSON.parse(raw);
    } catch {
      continue; // not fully written yet, or not a session file at all
    }
    if (typeof parsed.cwd !== "string" || typeof parsed.pid !== "number") continue;

    const resolvedCwd = path.resolve(parsed.cwd);
    const rel = path.relative(repoRoot, resolvedCwd);
    if (rel.startsWith("..") || path.isAbsolute(rel)) continue; // outside this repo

    sessions.push({
      sessionId: typeof parsed.sessionId === "string" ? parsed.sessionId : "",
      name: typeof parsed.name === "string" ? parsed.name : "",
      cwd: resolvedCwd,
      status: typeof parsed.status === "string" ? parsed.status : "unknown",
      pid: parsed.pid,
      jobId: typeof parsed.jobId === "string" ? parsed.jobId : "",
      updatedAt: typeof parsed.updatedAt === "number" ? new Date(parsed.updatedAt).toISOString() : "",
      alive: isProcessAlive(parsed.pid),
    });
  }
  sessions.sort((a, b) => a.name.localeCompare(b.name));
  return sessions;
}
