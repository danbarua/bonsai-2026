import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import {
  ListPromptsRequestSchema,
  ListResourceTemplatesRequestSchema,
  ListResourcesRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import path from "node:path";
import { z } from "zod";
import {
  archiveMessageByFilename,
  CHANNELS,
  listCodeSessions,
  PKG_VERSION,
  readMailbox,
  REPO_ROOT,
  sendMessage,
  type Channel,
} from "./mailbox.js";

interface ChannelToolConfig {
  toolPrefix: string; // e.g. "c2c" -> tools named c2c-send / c2c-inbox
  channel: Channel;
  channelLabel: string; // human-readable, for tool descriptions
  // Any of codeRoles writes outbox/ and reads inbox/ (the roles the
  // directories are named for); peerRole is the mirror image: writes
  // inbox/, reads outbox/. Both archive what they read to the same
  // archive/, matching the existing c2c protocol (Desktop can't delete,
  // only move -- Code follows the same convention for symmetry).
  //
  // codeRoles is a list, not a single string, because a channel can have
  // more than one legitimate internal-side sender: c2gpt is read/written
  // by BOTH claude-code and claude-desktop (either may message ChatGPT
  // directly), with chatgpt as the sole external peer -- there is no
  // ambiguity about which DIRECTORY a role writes to (any codeRole ->
  // outbox/, the peer -> inbox/), only about which of possibly-several
  // internal parties actually sent a given message (recorded in the
  // message's own "from:" header, unaffected by this).
  codeRoles: string[];
  peerRole: string;
}

const CHANNEL_TOOLS: ChannelToolConfig[] = [
  {
    toolPrefix: "c2c",
    channel: CHANNELS.c2c,
    channelLabel: "claude2claude (.claude/claude2claude/)",
    codeRoles: ["claude-code"],
    peerRole: "claude-desktop",
  },
  {
    toolPrefix: "c2gpt",
    channel: CHANNELS.c2gpt,
    channelLabel: "claude2gpt (.claude/claude2gpt/)",
    // Both claude-code and claude-desktop, not just one: either may
    // message ChatGPT directly (confirmed live -- Desktop was already
    // doing so, forced to pose as "claude-code" when that was the only
    // code-side role this tool offered, compensating with a manual
    // "btw I'm Claude Desktop" disclaimer in the message body). Listing
    // both roles here makes that disclaimer structural instead of an ad
    // hoc workaround in free text, without shutting Code out of the
    // channel it was originally built to use directly.
    codeRoles: ["claude-code", "claude-desktop"],
    peerRole: "chatgpt",
  },
];

function registerChannelTools(server: McpServer, cfg: ChannelToolConfig): void {
  const { codeRoles, peerRole } = cfg;
  const allRoles = [...codeRoles, peerRole] as unknown as [string, ...string[]];
  const isCodeRole = (role: string) => codeRoles.includes(role);
  const codeRoleList = codeRoles.map((r) => `"${r}"`).join(" or ");

  server.registerTool(
    `${cfg.toolPrefix}-send`,
    {
      title: `Send a ${cfg.toolPrefix} message`,
      description:
        `Write a new markdown message on the ${cfg.channelLabel} mailbox for the other side to read. ` +
        `\`sender\` decides which directory it lands in: ${codeRoleList} write to outbox/ (read by ` +
        `"${peerRole}"), "${peerRole}" writes to inbox/ (read by ${codeRoleList}). A leading ` +
        `"<!-- from: <sender> · <timestamp> -->" header is added automatically -- pass only the ` +
        `message body in \`content\`. Never overwrites an existing file: a same-second collision ` +
        `gets a -2, -3, ... suffix. Pass \`to\` to address this message to one specific session ` +
        `(e.g. a name from the code-sessions tool) rather than broadcasting to whichever reader ` +
        `gets there first -- an addressed message is skipped (left unread, not consumed) by any ` +
        `-inbox call passing a different \`as\` name. Pass \`instance\` when sending as ` +
        `"claude-code" to say WHICH Claude Code session wrote it (its \`/rename\`-set name -- ` +
        `see code-sessions -- never a raw session_id, which is an opaque GUID) -- a PreToolUse ` +
        `hook surfaces this session's own name as additional context right before this tool is ` +
        `called, specifically so it can be passed here.`,
      inputSchema: {
        sender: z
          .enum(allRoles)
          .describe(`Who this message is from: ${codeRoleList} (-> outbox/) or "${peerRole}" (-> inbox/).`),
        content: z.string().min(1).describe("The markdown message body (no header needed)."),
        to: z
          .string()
          .optional()
          .describe(
            "Address this message to one specific session name (see code-sessions). " +
              "Omit to broadcast (any reader may consume it) -- the default, and the only " +
              "behavior that existed before addressing was added.",
          ),
        instance: z
          .string()
          .optional()
          .describe(
            "Which Claude Code session sent this (its /rename-set name, e.g. from " +
              "code-sessions or the PreToolUse hook's additionalContext) -- meaningful when " +
              "sender is \"claude-code\", since multiple Code sessions can share this mailbox. " +
              "Omit if unknown or not applicable.",
          ),
      },
      outputSchema: {
        filename: z.string().describe("The written message's filename."),
        path: z.string().describe("Absolute path to the written file."),
        directory: z.enum(["inbox", "outbox"]).describe("Which directory it landed in."),
        to: z.string().optional().describe("The addressee, if this message was addressed."),
        instance: z.string().optional().describe("The sending Claude Code session's name, if provided."),
      },
    },
    async ({ sender, content, to, instance }) => {
      const dir = isCodeRole(sender) ? cfg.channel.outbox : cfg.channel.inbox;
      const dirName = isCodeRole(sender) ? "outbox" : "inbox";
      const result = await sendMessage(dir, sender, content, to, instance);
      const addressing = to ? ` (addressed to ${to})` : "";
      const instanceNote = instance ? ` (from instance ${instance})` : "";
      const structuredContent: {
        filename: string;
        path: string;
        directory: "inbox" | "outbox";
        to?: string;
        instance?: string;
      } = {
        filename: result.filename,
        path: result.path,
        directory: dirName,
        ...(to ? { to } : {}),
        ...(instance ? { instance } : {}),
      };
      return {
        content: [
          {
            type: "text",
            text: `Wrote ${cfg.channelLabel} ${dirName} message: ${result.filename}${addressing}${instanceNote}`,
          },
        ],
        structuredContent,
      };
    },
  );

  server.registerTool(
    `${cfg.toolPrefix}-inbox`,
    {
      title: `Read a ${cfg.toolPrefix} mailbox`,
      description:
        `Read pending messages on the ${cfg.channelLabel} mailbox, oldest first. \`reader\` decides ` +
        `which directory gets read: ${codeRoleList} read inbox/ (what "${peerRole}" sent), "${peerRole}" ` +
        `reads outbox/ (what ${codeRoleList} sent). By default each message read is moved to archive/ ` +
        `(mirroring the existing c2c protocol), so a later call only returns messages nobody has ` +
        `processed yet. Pass archive=false to peek without consuming. Pass \`as\` (this reader's own ` +
        `session name, see code-sessions) to skip -- leave unarchived, not returned as consumed -- any ` +
        `message addressed to a DIFFERENT name; a peek still shows everything regardless of \`as\`, ` +
        `since peeking never consumes anything. Omit \`as\` for the pre-addressing behavior: every ` +
        `message visible and archivable, addressed or not.`,
      inputSchema: {
        reader: z
          .enum(allRoles)
          .describe(`Whose mailbox to read: ${codeRoleList} (<- inbox/) or "${peerRole}" (<- outbox/).`),
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
      outputSchema: {
        directory: z.enum(["inbox", "outbox"]).describe("Which directory was read."),
        archived: z.boolean().describe("Whether returned messages were moved to archive/."),
        messages: z
          .array(
            z.object({
              filename: z.string(),
              content: z.string(),
              to: z.string().optional().describe("The addressee, if this message was addressed."),
            }),
          )
          .describe("Messages returned, oldest first."),
        skipped: z
          .array(z.string())
          .describe("Filenames left unarchived because they're addressed to a different name than `as`."),
      },
    },
    async ({ reader, archive, as }) => {
      const sourceDir = isCodeRole(reader) ? cfg.channel.inbox : cfg.channel.outbox;
      const dirName = isCodeRole(reader) ? "inbox" : "outbox";
      const { messages, skipped } = await readMailbox(sourceDir, cfg.channel.archive, archive, as);
      const structuredContent = { directory: dirName, archived: archive, messages, skipped };
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
          structuredContent,
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
          const instanceNote = m.instance ? ` (instance: ${m.instance})` : "";
          return `### ${m.filename}${addressing}${instanceNote}\n\n${m.content.trim()}`;
        })
        .join("\n\n---\n\n");
      return {
        content: [{ type: "text", text: `${summary}\n\n${body}` }],
        structuredContent,
      };
    },
  );
}

// code2code has no fixed peer role -- every party is a Claude Code
// session (unlike c2c/c2gpt, which always have a non-Code peer:
// claude-desktop or chatgpt) -- so it doesn't fit registerChannelTools'
// two-role shape and gets its own registration instead of a third
// CHANNEL_TOOLS entry. Differences that matter, not just naming:
//
// - No `sender`/`reader` role parameter at all -- there's only one role,
//   so it'd always be the literal "claude-code", adding a field with one
//   valid value for no benefit.
// - `instance` (send) and `as` (inbox) are REQUIRED, not optional. On
//   c2c/c2gpt, omitting them degrades gracefully to "sender/reader
//   identity unknown" because a non-Code peer (Desktop, ChatGPT) doesn't
//   have multiple instances to disambiguate. Here, EVERY party is
//   Multi-instance-Code, so an unidentified sender/reader defeats the
//   channel's entire reason to exist (see code-sessions) -- and
//   pre-c2c-mcp.sh's PreToolUse hook already auto-injects both via
//   updatedInput, so requiring them costs a well-behaved caller nothing.
// - readMailbox is called with excludeSelfSent: true (always) -- see its
//   doc comment. Without this, a session's own broadcast would show up as
//   its own unread mail the moment it called -inbox after sending.
function registerCode2CodeTools(server: McpServer): void {
  const channel = CHANNELS.code2code;

  server.registerTool(
    "code2code-send",
    {
      title: "Send a message to another Claude Code session",
      description:
        "Write a new markdown message on the code2code mailbox (.claude/code2code/) for another " +
        "Claude Code session to read. Unlike c2c/c2gpt, every party here is a Claude Code session " +
        "-- there's no fixed peer role -- so `instance` (this session's own /rename-set name, see " +
        "code-sessions) is REQUIRED: a PreToolUse hook auto-injects it if omitted, so a caller " +
        "normally never has to supply it explicitly. Pass `to` (a name from code-sessions) to " +
        "address one specific recipient -- STRONGLY preferred over omitting it: an unaddressed " +
        "broadcast is consumed by whichever session calls code2code-inbox first, which is fine " +
        "for a one-off announcement but wrong as a default for anything meant for one recipient. " +
        "Never overwrites an existing file: a same-second collision gets a -2, -3, ... suffix.",
      inputSchema: {
        instance: z
          .string()
          .min(1)
          .describe(
            "This session's own /rename-set name (see code-sessions). Required -- every " +
              "code2code message needs a known sender, since there's no fixed peer role to fall " +
              "back on. A PreToolUse hook supplies this automatically; pass it explicitly only if " +
              "overriding that.",
          ),
        content: z.string().min(1).describe("The markdown message body (no header needed)."),
        to: z
          .string()
          .optional()
          .describe(
            "Address this message to one specific session name (see code-sessions). Omitting " +
              "this broadcasts to whichever session calls code2code-inbox first -- prefer setting " +
              "it for anything meant for one recipient.",
          ),
      },
      outputSchema: {
        filename: z.string().describe("The written message's filename."),
        path: z.string().describe("Absolute path to the written file."),
        to: z.string().optional().describe("The addressee, if this message was addressed."),
        instance: z.string().describe("The sending Claude Code session's name."),
      },
    },
    async ({ instance, content, to }) => {
      const result = await sendMessage(channel.outbox, "claude-code", content, to, instance);
      const addressing = to ? ` (addressed to ${to})` : " (broadcast -- first reader to call code2code-inbox consumes it)";
      return {
        content: [
          {
            type: "text",
            text: `Wrote code2code message: ${result.filename}${addressing}, from instance ${instance}`,
          },
        ],
        structuredContent: { filename: result.filename, path: result.path, ...(to ? { to } : {}), instance },
      };
    },
  );

  server.registerTool(
    "code2code-inbox",
    {
      title: "Read code2code mail addressed to this session",
      description:
        "Read pending messages on the code2code mailbox (.claude/code2code/), oldest first. `as` " +
        "(this reader's own /rename-set name, see code-sessions) is REQUIRED: a PreToolUse hook " +
        "auto-injects it if omitted, so a caller normally never has to supply it explicitly. A " +
        "consuming read (archive:true, the default) skips -- leaves unarchived, not returned -- " +
        "any message addressed to a DIFFERENT session, AND any unaddressed broadcast this same " +
        "session itself sent (so a session never consumes its own announcement before anyone else " +
        "sees it). Pass archive=false to peek without consuming -- a peek always shows everything, " +
        "self-sent or not.",
      inputSchema: {
        as: z
          .string()
          .min(1)
          .describe(
            "This reader's own /rename-set name (see code-sessions). Required. A PreToolUse " +
              "hook supplies this automatically; pass it explicitly only if overriding that.",
          ),
        archive: z.boolean().default(true).describe("Move each returned message to archive/ after reading (default true)."),
      },
      outputSchema: {
        archived: z.boolean().describe("Whether returned messages were moved to archive/."),
        messages: z
          .array(
            z.object({
              filename: z.string(),
              content: z.string(),
              to: z.string().optional().describe("The addressee, if this message was addressed."),
              instance: z.string().optional().describe("The sending session's name."),
            }),
          )
          .describe("Messages returned, oldest first."),
        skipped: z
          .array(z.string())
          .describe(
            "Filenames left unarchived: addressed to a different session, or this session's own broadcast.",
          ),
      },
    },
    async ({ as, archive }) => {
      const { messages, skipped } = await readMailbox(channel.inbox, channel.archive, archive, as, true);
      const structuredContent = { archived: archive, messages, skipped };
      const skipNote =
        skipped.length > 0
          ? ` ${skipped.length} message(s) left unread (addressed elsewhere, or your own broadcast): ${skipped.join(", ")}.`
          : "";
      if (messages.length === 0) {
        return {
          content: [
            { type: "text", text: `code2code mailbox is empty -- no pending messages for ${as}.${skipNote}` },
          ],
          structuredContent,
        };
      }
      const summary = `${messages.length} message(s) read from code2code (as ${as}, archived: ${archive}).${skipNote}`;
      const body = messages
        .map((m) => {
          const addressing = m.to ? ` (to: ${m.to})` : "";
          const instanceNote = m.instance ? ` (from: ${m.instance})` : "";
          return `### ${m.filename}${addressing}${instanceNote}\n\n${m.content.trim()}`;
        })
        .join("\n\n---\n\n");
      return {
        content: [{ type: "text", text: `${summary}\n\n${body}` }],
        structuredContent,
      };
    },
  );

  server.registerTool(
    "code2code-archive",
    {
      title: "Archive a specific code2code message by filename",
      description:
        "Move ONE specific message on the code2code mailbox (.claude/code2code/) to archive/, by " +
        "filename (see a code2code-inbox result or a peek with archive:false). Mainly for " +
        "retracting your own stale broadcast: a session's own unaddressed broadcast is never " +
        "archived by that session's own code2code-inbox calls (so it can't accidentally consume " +
        "its own announcement before anyone else sees it) -- this is the explicit, deliberate way " +
        "to clear one once it's served its purpose. Not restricted to your own messages or to " +
        "broadcasts -- it archives whatever filename it's given. No-op (found: false, not an " +
        "error) if the filename isn't present -- already archived, already gone, or never existed.",
      inputSchema: {
        filename: z.string().min(1).describe("The exact filename to archive (from code2code-inbox or a peek)."),
      },
      outputSchema: {
        found: z.boolean().describe("Whether the file was present and archived."),
      },
    },
    async ({ filename }) => {
      const found = await archiveMessageByFilename(channel.inbox, channel.archive, filename);
      return {
        content: [
          {
            type: "text",
            text: found
              ? `Archived ${filename}.`
              : `${filename} was not found in the code2code mailbox (already archived, or never existed).`,
          },
        ],
        structuredContent: { found },
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
      outputSchema: {
        sessions: z.array(
          z.object({
            sessionId: z.string(),
            name: z.string(),
            cwd: z.string().describe("Absolute path, relative-displayed in the text summary only."),
            status: z.string(),
            pid: z.number(),
            jobId: z.string(),
            updatedAt: z.string().describe("ISO 8601, empty string if unavailable."),
            alive: z.boolean().describe("Checked directly (kill -0), not trusted from the file's own status."),
          }),
        ),
      },
    },
    async () => {
      const sessions = await listCodeSessions(REPO_ROOT);
      if (sessions.length === 0) {
        return {
          content: [{ type: "text", text: "No Claude Code sessions found under this repository." }],
          structuredContent: { sessions: [] },
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
        structuredContent: { sessions },
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
  registerCode2CodeTools(server);
  registerCodeSessionsTool(server);
  server.server.setRequestHandler(ListResourcesRequestSchema, async () => ({ resources: [] }));
  server.server.setRequestHandler(ListResourceTemplatesRequestSchema, async () => ({ resourceTemplates: [] }));
  server.server.setRequestHandler(ListPromptsRequestSchema, async () => ({ prompts: [] }));
  return server;
}
