import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { CHANNELS, readInbox, sendMessage, type Channel } from "./mailbox.js";

interface ChannelToolConfig {
  toolPrefix: string; // e.g. "c2c" -> tools named c2c-send / c2c-inbox
  channel: Channel;
  channelLabel: string; // human-readable, for tool descriptions
  senders: [string, string]; // the two valid `sender` values for -send
}

const CHANNEL_TOOLS: ChannelToolConfig[] = [
  {
    toolPrefix: "c2c",
    channel: CHANNELS.c2c,
    channelLabel: "claude2claude (.claude/claude2claude/)",
    senders: ["claude-code", "claude-desktop"],
  },
  {
    toolPrefix: "c2gpt",
    channel: CHANNELS.c2gpt,
    channelLabel: "claude2gpt (.claude/claude2gpt/)",
    senders: ["claude-code", "chatgpt"],
  },
];

function registerChannelTools(server: McpServer, cfg: ChannelToolConfig): void {
  const [senderA, senderB] = cfg.senders;

  server.registerTool(
    `${cfg.toolPrefix}-send`,
    {
      title: `Send a ${cfg.toolPrefix} message`,
      description:
        `Write a new markdown message to the ${cfg.channelLabel} outbox for the other side to read. ` +
        `A leading "<!-- from: <sender> · <timestamp> -->" header is added automatically -- pass only ` +
        `the message body in \`content\`. Never overwrites an existing file: a same-second collision ` +
        `gets a -2, -3, ... suffix.`,
      inputSchema: {
        sender: z
          .enum([senderA, senderB])
          .describe(`Who this message is from: "${senderA}" or "${senderB}".`),
        content: z.string().min(1).describe("The markdown message body (no header needed)."),
      },
    },
    async ({ sender, content }) => {
      const result = await sendMessage(cfg.channel, sender, content);
      return {
        content: [
          {
            type: "text",
            text: `Wrote ${cfg.channelLabel} outbox message: ${result.filename}`,
          },
        ],
      };
    },
  );

  server.registerTool(
    `${cfg.toolPrefix}-inbox`,
    {
      title: `Read the ${cfg.toolPrefix} inbox`,
      description:
        `Read pending messages from the ${cfg.channelLabel} inbox, oldest first. By default each ` +
        `message read is moved to archive/ (mirroring the existing c2c protocol), so a later call ` +
        `only returns messages nobody has processed yet. Pass archive=false to peek without consuming.`,
      inputSchema: {
        archive: z
          .boolean()
          .default(true)
          .describe("Move each returned message to archive/ after reading (default true)."),
      },
    },
    async ({ archive }) => {
      const messages = await readInbox(cfg.channel, archive);
      if (messages.length === 0) {
        return {
          content: [
            { type: "text", text: `${cfg.channelLabel} inbox is empty -- no pending messages.` },
          ],
        };
      }
      const summary = `${messages.length} message(s) read from the ${cfg.channelLabel} inbox (archived: ${archive}).`;
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
