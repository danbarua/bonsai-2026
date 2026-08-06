# c2c-mcp

A local HTTP MCP server exposing the claude2claude and claude2gpt
filesystem mailboxes as MCP tools, instead of requiring a client-side
skill (`.claude/skills/c2c/SKILL.md`) or filesystem connector to
implement the protocol itself. Any MCP-capable client that can reach
`http://127.0.0.1:8765/mcp` gets `c2c-send`/`c2c-inbox` and
`c2gpt-send`/`c2gpt-inbox` as ordinary tool calls.

Message format and directory conventions (`inbox/` → `outbox/` →
`archive/`, the `<!-- from: <sender> · <timestamp> -->` header, oldest-
first ordering) match the existing protocol documented in
`.claude/claude2claude/DESKTOP_PROTOCOL.md` and
`.claude/skills/c2c/SKILL.md` -- this server is a second
implementation of the same mailbox convention, not a different one.

## Tools

| Tool | Channel dir | `sender` values | Effect |
|---|---|---|---|
| `c2c-send` | `.claude/claude2claude/` | `claude-code`, `claude-desktop` | Writes a new timestamped file to `outbox/` |
| `c2c-inbox` | `.claude/claude2claude/` | -- | Reads `inbox/*.md` oldest-first, moves each to `archive/` unless called with `archive: false` |
| `c2gpt-send` | `.claude/claude2gpt/` | `claude-code`, `chatgpt` | Same as `c2c-send`, other channel |
| `c2gpt-inbox` | `.claude/claude2gpt/` | -- | Same as `c2c-inbox`, other channel |

`-send` takes `{ sender, content }` -- `content` is just the message
body; the header comment is generated from `sender` and the current
UTC time. A same-second filename collision gets a `-2`, `-3`, ...
suffix rather than overwriting anything.

`-inbox` takes `{ archive?: boolean }` (default `true`). With
`archive: false` it returns pending messages without moving them, for
peeking without consuming.

## Build & run

```bash
npm install
npm run build   # tsc -> dist/
npm start        # node dist/index.js
```

Or skip the build step during development:

```bash
npm run dev       # tsx src/index.ts
```

The server binds to `127.0.0.1:8765` by default (`C2C_MCP_HOST`,
`C2C_MCP_PORT` to override) and serves the MCP endpoint at `/mcp` and
a plain JSON health check at `/health`. It resolves the repo root from
its own file location, so it can be started from any working
directory; `BONSAI_PROJECT_ROOT` overrides that (used by tests to point
at a throwaway mailbox instead of the real one).

The transport is stateless Streamable HTTP: each request gets its own
server/transport pair, since every tool call here is a self-contained
filesystem read or write with nothing to keep alive between calls.

## Registering with a client

The server must already be running (`npm start` in a terminal, or
under a process manager) before a client tries to connect -- nothing
here starts it automatically.

**Claude Code** -- add an HTTP entry to `.mcp.json`:

```json
{
  "mcpServers": {
    "c2c-mcp": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

**Claude Desktop** -- add the same URL as a remote/HTTP MCP server in
Desktop's connector settings.

**ChatGPT** -- add `http://127.0.0.1:8765/mcp` as a custom MCP
connector in the app's connector settings.
