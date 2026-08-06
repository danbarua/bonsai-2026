import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { CHANNELS, readMailbox, sendMessage, type Channel } from "./mailbox.js";

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
        `gets a -2, -3, ... suffix.`,
      inputSchema: {
        sender: z
          .enum([codeRole, peerRole])
          .describe(`Who this message is from: "${codeRole}" (-> outbox/) or "${peerRole}" (-> inbox/).`),
        content: z.string().min(1).describe("The markdown message body (no header needed)."),
      },
    },
    async ({ sender, content }) => {
      const dir = sender === codeRole ? cfg.channel.outbox : cfg.channel.inbox;
      const dirName = sender === codeRole ? "outbox" : "inbox";
      const result = await sendMessage(dir, sender, content);
      return {
        content: [
          {
            type: "text",
            text: `Wrote ${cfg.channelLabel} ${dirName} message: ${result.filename}`,
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
        `processed yet. Pass archive=false to peek without consuming.`,
      inputSchema: {
        reader: z
          .enum([codeRole, peerRole])
          .describe(`Whose mailbox to read: "${codeRole}" (<- inbox/) or "${peerRole}" (<- outbox/).`),
        archive: z
          .boolean()
          .default(true)
          .describe("Move each returned message to archive/ after reading (default true)."),
      },
    },
    async ({ reader, archive }) => {
      const sourceDir = reader === codeRole ? cfg.channel.inbox : cfg.channel.outbox;
      const dirName = reader === codeRole ? "inbox" : "outbox";
      const messages = await readMailbox(sourceDir, cfg.channel.archive, archive);
      if (messages.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: `${cfg.channelLabel} ${dirName} is empty -- no pending messages for ${reader}.`,
            },
          ],
        };
      }
      const summary =
        `${messages.length} message(s) read from the ${cfg.channelLabel} ${dirName} ` +
        `(as ${reader}, archived: ${archive}).`;
      const body = messages
        .map((m) => `### ${m.filename}\n\n${m.content.trim()}`)
        .join("\n\n---\n\n");
      return {
        content: [{ type: "text", text: `${summary}\n\n${body}` }],
      };
    },
  );
}

export function createServer(): McpServer {
  const server = new McpServer({
    name: "c2c-mcp",
    version: "0.1.0",
  });
  for (const cfg of CHANNEL_TOOLS) {
    registerChannelTools(server, cfg);
  }
  return server;
}
