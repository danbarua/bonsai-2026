import { promises as fs } from "node:fs";
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
