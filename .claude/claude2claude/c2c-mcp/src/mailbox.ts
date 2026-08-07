import { promises as fs } from "node:fs";
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
 */
export async function sendMessage(
  dir: string,
  sender: string,
  content: string,
): Promise<SendResult> {
  await ensureDir(dir);
  const now = new Date();
  const base = timestampFilename(now);
  const header = `<!-- from: ${sender} · ${timestampIso(now)} -->\n\n`;
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
}

/**
 * Reads every `.md` file in `sourceDir` (the caller picks inbox or outbox
 * based on whose messages the reader wants -- see server.ts's role
 * mapping), oldest first (filenames sort chronologically). When `archive`
 * is true (the default, matching the existing c2c protocol), each file is
 * moved to `archiveDir` after being read, so a later call only returns
 * messages nobody has processed yet.
 */
export async function readMailbox(
  sourceDir: string,
  archiveDir: string,
  archive: boolean,
): Promise<InboxMessage[]> {
  await ensureDir(sourceDir);
  const entries = await fs.readdir(sourceDir);
  const filenames = entries.filter((f) => f.endsWith(".md")).sort();

  const messages: InboxMessage[] = [];
  for (const filename of filenames) {
    const filePath = path.join(sourceDir, filename);
    const content = await fs.readFile(filePath, "utf8");
    messages.push({ filename, content });
    if (archive) {
      await ensureDir(archiveDir);
      await fs.rename(filePath, path.join(archiveDir, filename));
    }
  }
  return messages;
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
