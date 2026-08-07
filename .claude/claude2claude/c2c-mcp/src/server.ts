import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import {
  ListPromptsRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListResourcesRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import path from "node:path";
import { z } from "zod";
import { CHANNELS, listCodeSessions, PKG_VERSION, readMailbox, REPO_ROOT, sendMessage, type Channel } from "./mailbox.js";

interface ChannelToolConfig {
  toolPrefix: string; // e.g. "c2c" -> tools named c2c-send / c2c-inbox
  channel: Channel;
  channelLabel: string; // human-readable, for tool descriptions
  // codeRole always writes outbox/ and reads inbox/ (the roles the
  // directories are named for); peerRole is the mirror image: writes
  // inbox/, reads outbox/. Both archive what they read to the same
  // archive/, matching the existing c2c protocol (Desktop can't delete,
  // only move -- Code follows the same convention for symmetry).
  codeRole: string;
  peerRole: string;
}

const CHANNEL_TOOLS: ChannelToolConfig[] = [
  {
    toolPrefix: "c2c",
    channel: CHANNELS.c2c,
    channelLabel: "claude2claude (.claude/claude2claude/)",
    codeRole: "claude-code",
    peerRole: "claude-desktop",
  },
  {
    toolPrefix: "c2gpt",
    channel: CHANNELS.c2gpt,
    channelLabel: "claude2gpt (.claude/claude2gpt/)",
    codeRole: "claude-code",
    peerRole: "chatgpt",
  },
];

function registerChannelTools(server: McpServer, cfg: ChannelToolConfig): void {
  const { codeRole, peerRole } = cfg;

  server.registerTool(
    `${cfg.toolPrefix}-send`,
    {
      title: `Send a ${cfg.toolPrefix} message`,
      description:
        `Write a new markdown message on the ${cfg.channelLabel} mailbox for the other side to read. ` +
        `\`sender\` decides which directory it lands in: "${codeRole}" writes to outbox/ (read by ` +
        `"${peerRole}"), "${peerRole}" writes to inbox/ (read by "${codeRole}"). A leading ` +
        `"<!-- from: <sender> · <timestamp> -->" header is added automatically -- pass only the ` +
        `message body in \`content\`. Never overwrites an existing file: a same-second collision ` +
        `gets a -2, -3, ... suffix. Pass \`to\` to address this message to one specific session ` +
        `(e.g. a name from the code-sessions tool) rather than broadcasting to whichever reader ` +
        `gets there first -- an addressed message is skipped (left unread, not consumed) by any ` +
        `-inbox call passing a different \`as\` name.`,
      inputSchema: {
        sender: z
          .enum([codeRole, peerRole])
          .describe(`Who this message is from: "${codeRole}" (-> outbox/) or "${peerRole}" (-> inbox/).`),
        content: z.string().min(1).describe("The markdown message body (no header needed)."),
        to: z
          .string()
          .optional()
          .describe(
            "Address this message to one specific session name (see code-sessions). " +
              "Omit to broadcast (any reader may consume it) -- the default, and the only " +
              "behavior that existed before addressing was added.",
          ),
      },
    },
    async ({ sender, content, to }) => {
      const dir = sender === codeRole ? cfg.channel.outbox : cfg.channel.inbox;
      const dirName = sender === codeRole ? "outbox" : "inbox";
      const result = await sendMessage(dir, sender, content, to);
      const addressing = to ? ` (addressed to ${to})` : "";
      return {
        content: [
          {
            type: "text",
            text: `Wrote ${cfg.channelLabel} ${dirName} message: ${result.filename}${addressing}`,
          },
        ],
      };
    },
  );

  server.registerTool(
    `${cfg.toolPrefix}-inbox`,
    {
      title: `Read a ${cfg.toolPrefix} mailbox`,
      description:
        `Read pending messages on the ${cfg.channelLabel} mailbox, oldest first. \`reader\` decides ` +
        `which directory gets read: "${codeRole}" reads inbox/ (what "${peerRole}" sent), "${peerRole}" ` +
        `reads outbox/ (what "${codeRole}" sent). By default each message read is moved to archive/ ` +
        `(mirroring the existing c2c protocol), so a later call only returns messages nobody has ` +
        `processed yet. Pass archive=false to peek without consuming. Pass \`as\` (this reader's own ` +
        `session name, see code-sessions) to skip -- leave unarchived, not returned as consumed -- any ` +
        `message addressed to a DIFFERENT name; a peek still shows everything regardless of \`as\`, ` +
        `since peeking never consumes anything. Omit \`as\` for the pre-addressing behavior: every ` +
        `message visible and archivable, addressed or not.`,
      inputSchema: {
        reader: z
          .enum([codeRole, peerRole])
          .describe(`Whose mailbox to read: "${codeRole}" (<- inbox/) or "${peerRole}" (<- outbox/).`),
        archive: z
          .boolean()
          .default(true)
          .describe("Move each returned message to archive/ after reading (default true)."),
        as: z
          .string()
          .optional()
          .describe(
            "This reader's own session name (see code-sessions). When set, a consuming read " +
              "(archive:true) skips mail addressed to a different name instead of consuming it. " +
              "Has no effect on a peek (archive:false), which always shows everything. Omit for " +
              "the pre-addressing behavior.",
          ),
      },
    },
    async ({ reader, archive, as }) => {
      const sourceDir = reader === codeRole ? cfg.channel.inbox : cfg.channel.outbox;
      const dirName = reader === codeRole ? "inbox" : "outbox";
      const { messages, skipped } = await readMailbox(sourceDir, cfg.channel.archive, archive, as);
      if (messages.length === 0) {
        const skipNote =
          skipped.length > 0
            ? ` (${skipped.length} message(s) addressed to another session were left unread: ${skipped.join(", ")})`
            : "";
        return {
          content: [
            {
              type: "text",
              text: `${cfg.channelLabel} ${dirName} is empty -- no pending messages for ${reader}.${skipNote}`,
            },
          ],
        };
      }
      const skipNote =
        skipped.length > 0
          ? ` ${skipped.length} message(s) addressed to another session were left unread: ${skipped.join(", ")}.`
          : "";
      const summary =
        `${messages.length} message(s) read from the ${cfg.channelLabel} ${dirName} ` +
        `(as ${reader}, archived: ${archive}).${skipNote}`;
      const body = messages
        .map((m) => {
          const addressing = m.to ? ` (to: ${m.to})` : "";
          return `### ${m.filename}${addressing}\n\n${m.content.trim()}`;
        })
        .join("\n\n---\n\n");
      return {
        content: [{ type: "text", text: `${summary}\n\n${body}` }],
      };
    },
  );
}

// Not channel-scoped like registerChannelTools's tools -- this reads the
// CLI's own global session registry (~/.claude/sessions/), not a mailbox
// directory, so it gets its own standalone registration.
function registerCodeSessionsTool(server: McpServer): void {
  server.registerTool(
    "code-sessions",
    {
      title: "List Claude Code sessions",
      description:
        "Lists Claude Code sessions on this machine whose working directory is under " +
        "this repository (the main checkout or any .claude/worktrees/* inside it), " +
        "read from the CLI's own local session registry -- not this mailbox. Each " +
        "entry has the session's human-assigned name (set via /rename), its working " +
        "directory, last-known status, and whether its process is still actually " +
        "alive (checked directly, not just trusted from a possibly-stale file). " +
        "Useful for seeing which named sessions currently exist, e.g. before " +
        "addressing a mailbox message to a specific one.",
      inputSchema: {},
    },
    async () => {
      const sessions = await listCodeSessions(REPO_ROOT);
      if (sessions.length === 0) {
        return {
          content: [{ type: "text", text: "No Claude Code sessions found under this repository." }],
        };
      }
      const lines = sessions.map((s) => {
        const relCwd = path.relative(REPO_ROOT, s.cwd) || ".";
        return (
          `- **${s.name || "(unnamed)"}** -- ${s.alive ? "alive" : "not running"}, ` +
          `status=${s.status}, cwd=${relCwd}, sessionId=${s.sessionId}, pid=${s.pid}`
        );
      });
      return {
        content: [{ type: "text", text: `${sessions.length} session(s) under this repo:\n\n${lines.join("\n")}` }],
      };
    },
  );
}

export function createServer(): McpServer {
  const server = new McpServer(
    {
      name: "c2c-mcp",
      version: PKG_VERSION,
    },
    // Declared even though we have none of either: some MCP clients (found
    // debugging ChatGPT's Developer Mode connector) call resources/list and
    // prompts/list unconditionally during discovery, not gated on whether a
    // server advertised the capability first. Without a handler registered
    // for a method, the SDK's dispatcher returns a hard JSON-RPC
    // "Method not found" instead of a valid empty result -- which broke
    // discovery entirely (no tools were visible, not just resources/prompts).
    // registerCapabilities merges rather than overwrites, so this coexists
    // fine with registerTool's own separate "tools" capability registration
    // below.
    { capabilities: { resources: {}, prompts: {} } },
  );
  for (const cfg of CHANNEL_TOOLS) {
    registerChannelTools(server, cfg);
  }
  registerCodeSessionsTool(server);
  server.server.setRequestHandler(ListResourcesRequestSchema, async () => ({ resources: [] }));
  server.server.setRequestHandler(ListResourceTemplatesRequestSchema, async () => ({ resourceTemplates: [] }));
  server.server.setRequestHandler(ListPromptsRequestSchema, async () => ({ prompts: [] }));
  return server;
}
