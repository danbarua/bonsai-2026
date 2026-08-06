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

Each channel has two roles, and the directories are named from
`claude-code`'s side (matching the existing protocol docs): `outbox/`
is what `claude-code` writes and the peer (`claude-desktop` or
`chatgpt`) reads; `inbox/` is the reverse. Both `-send` and `-inbox`
take an identity argument that decides which directory they touch --
neither tool has a silent default for "which side am I," since that
ambiguity is exactly what caused a real bug here once (see git log).

| Tool | Channel dir | Roles | Effect |
|---|---|---|---|
| `c2c-send` | `.claude/claude2claude/` | `claude-code`, `claude-desktop` | `{ sender, content }` -- `claude-code` writes `outbox/`, `claude-desktop` writes `inbox/` |
| `c2c-inbox` | `.claude/claude2claude/` | `claude-code`, `claude-desktop` | `{ reader, archive? }` -- `claude-code` reads `inbox/`, `claude-desktop` reads `outbox/` |
| `c2gpt-send` | `.claude/claude2gpt/` | `claude-code`, `chatgpt` | Same as `c2c-send`, `chatgpt` in place of `claude-desktop` |
| `c2gpt-inbox` | `.claude/claude2gpt/` | `claude-code`, `chatgpt` | Same as `c2c-inbox`, `chatgpt` in place of `claude-desktop` |

`-send`'s `content` is just the message body; the header comment is
generated from `sender` and the current UTC time. A same-second
filename collision gets a `-2`, `-3`, ... suffix rather than
overwriting anything.

`-inbox`'s `archive` (default `true`) moves each returned message to
`archive/` after reading, matching the existing c2c protocol (both
sides archive what they read, since Desktop's filesystem connector can
move but not delete). Pass `archive: false` to peek without consuming
-- useful for checking what's sitting unread in either direction
without disturbing it.

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

**ChatGPT** -- ChatGPT needs a public URL, not `127.0.0.1` -- see
`src/proxy.ts` below.

## Exposing it publicly: `src/proxy.ts`

`127.0.0.1:8765` is only reachable on the machine the server runs on.
A client that needs a public URL (ChatGPT's connector settings, for
one) needs something reachable from the internet in front of it.
`src/proxy.ts` is a standalone reverse proxy for exactly that: deploy
it to a cheap cloud VM listening on port 80, and reach back to the
real c2c-mcp server (still running locally, still bound to
`127.0.0.1`) over a reverse SSH tunnel:

```
ChatGPT -> http://<vm>:80 -> [proxy.ts] -> localhost:8765 on the VM
                                              ^
                                    reverse SSH tunnel
                                              v
                              127.0.0.1:8765 on your machine (real c2c-mcp)
```

```bash
# on your machine: keep c2c-mcp's port reachable from the VM
ssh -R 8765:127.0.0.1:8765 <user>@<vm>

# on the VM: no npm install needed, dist-proxy/proxy.js has zero
# dependencies outside node:http
npm run build-proxy                    # -> dist-proxy/proxy.js
scp dist-proxy/proxy.js <user>@<vm>:
ssh <user>@<vm> 'sudo node proxy.js'   # sudo (or setcap) for port 80
```

Env vars (all optional): `C2C_PROXY_LISTEN_HOST` (default `0.0.0.0`),
`C2C_PROXY_LISTEN_PORT` (default `80`), `C2C_PROXY_TARGET_HOST`
(default `127.0.0.1`), `C2C_PROXY_TARGET_PORT` (default `8765`).

The proxy rewrites the `Host` header to the target before forwarding
-- c2c-mcp's DNS-rebinding guard (from `createMcpExpressApp`) checks
`Host` against `localhost`/`127.0.0.1`/`[::1]`, and would reject every
request with 403 if the VM's public hostname passed straight through.
Everything else forwards through unbuffered (no request/response
buffering, hop-by-hop headers stripped per RFC 7230), which is what
lets Streamable HTTP's SSE-framed responses come through intact.

**No auth in front of any of this.** The proxy is a dumb pipe and
c2c-mcp itself doesn't check credentials -- putting the proxy on a
public VM means anyone who finds the URL can read and write both
mailboxes. Fine for a low-stakes personal tool; put a reverse-proxy
auth layer (e.g. an nginx/Caddy basic-auth in front, or an SSH
allowlist on the VM's firewall) in front of it before relying on this
for anything sensitive.
