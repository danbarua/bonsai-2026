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

A fifth tool, `code-sessions`, isn't channel-scoped: it takes no
arguments and lists Claude Code sessions on this machine whose working
directory is under this repo (the main checkout or any
`.claude/worktrees/*`), reading the CLI's own local session registry
(`~/.claude/sessions/*.json`, not any mailbox) rather than anything
this server writes itself. Each entry has the session's `/rename`-set
name, its `cwd`, last-known status, and whether its process is still
actually alive (checked directly via a zero-signal `kill` probe, not
just trusted from a possibly-stale file) -- useful for a peer deciding
which named session a mailbox message should be addressed to.
`CLAUDE_SESSIONS_DIR` overrides the registry location (used by
`test/code-sessions.sh` to point at a throwaway directory instead of
real global state).

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

**Claude Desktop/iOS** -- these need a public URL, not `127.0.0.1`
(Desktop's config-file entry doesn't support remote servers at all;
only Settings -> Connectors does), *and* in practice require a working
OAuth handshake even for a server meant to be authless -- see
`src/proxy.ts` and the OAuth section below, then add the public
`https://.../mcp` URL via Settings -> Connectors, leaving the OAuth
Client ID/Secret fields blank (DCR registers one automatically).

**ChatGPT** -- also needs the public URL from `src/proxy.ts` below.

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

# on the VM: no npm install needed, dist-proxy/proxy.cjs has zero
# dependencies outside node:http
npm run build-proxy                     # -> dist-proxy/proxy.cjs
scp dist-proxy/proxy.cjs <user>@<vm>:
ssh <user>@<vm> 'sudo node proxy.cjs'   # sudo (or setcap) for port 80
```

The output is named `.cjs`, not `.js`: Node infers module type from the
*nearest* `package.json` above a script, and this repo's declares
`"type": "module"` -- run as plain `.js` from inside this directory
tree, Node would misread the compiled CommonJS output as ESM and
crash. `.cjs` is unambiguous regardless of what sits above it, which
matters doubly on the VM where there's no surrounding `package.json`
at all.

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

**The proxy itself is still a dumb, unauthenticated pipe.** Anyone who
can reach the VM can reach c2c-mcp's HTTP surface. What sits behind
that surface is real auth now (see below) -- `/mcp` itself rejects
public traffic with no valid token -- but the `/authorize`,
`/register`, and `/token` endpoints are necessarily reachable by
anyone too, since that's how a client bootstraps into the flow. An
attacker can still burn cycles hitting those, just not read or write
either mailbox without completing consent. A firewall allowlist or a
reverse-proxy IP restriction in front of the VM is still worth doing
if you want to shrink that surface further.

## OAuth for Claude Desktop/iOS: `src/oauth.ts`

Claude Desktop and iOS's "Add custom connector" flow requires a
working auth handshake even when a server is meant to be authless in
principle -- authless (`none`) is documented as supported, but in
practice the connector dialog forced OAuth. `src/oauth.ts` is a
minimal, spec-compliant OAuth 2.1 authorization server (Dynamic Client
Registration + PKCE) mounted on the same Express app as `/mcp`,
built for exactly one consenting user, not a multi-tenant service:

- `GET /.well-known/oauth-protected-resource` (RFC 9728)
- `GET /.well-known/oauth-authorization-server` (RFC 8414)
- `POST /register` (RFC 7591, Dynamic Client Registration -- no
  client secret, PKCE S256 required instead)
- `GET`/`POST /authorize` -- a one-button consent page, no login
  (anyone who reaches this URL can click Allow; see the security note
  above)
- `POST /token` -- authorization-code exchange and refresh

**Enable it** by setting `C2C_MCP_PUBLIC_URL` to exactly the URL
you'll paste into Claude's connector dialog (path included, e.g.
`https://c2c.framesift.ai/mcp`) when starting c2c-mcp. Leave it unset
for pure local use and `/mcp` stays fully authless, as before.

**Auth only applies to traffic that came through the proxy**, not to
local callers (Claude Code's own `.mcp.json` entry, `curl` against
`127.0.0.1` directly): `src/proxy.ts` stamps every request it forwards
with a marker header (overwriting any client-supplied copy first, so
a public caller can't fake "I'm local"), and `requireBearerAuth` only
checks for a token when that header is present. `127.0.0.1` binding is
already Claude Code's trust boundary; this doesn't add a second gate
on top of it, and it means Claude Code never needs to go through the
OAuth dance CIMD would otherwise require for its loopback-redirect
client. If you ever run c2c-mcp directly on a public interface without
the proxy in front, this scheme doesn't protect you -- it exists to
gate the proxy's ingress specifically.

**Tokens are self-verifying, not looked up in a store.** Access and
refresh tokens are `payload.signature` pairs (HMAC-SHA256 over a
JSON claims blob) signed with a key generated on first run and
persisted to the gitignored `.data/oauth-signing-key` -- restarting
the server (as `run-c2c-mcp.sh`-style dev loops do) doesn't invalidate
already-issued tokens, which matters because the whole point is
keeping a long-running claude.ai chat connected across restarts.
Authorization codes stay in-memory with a 60-second TTL and don't
survive a restart -- self-healing, since a client whose code was
mid-flight during a restart just re-authorizes. Registered DCR
clients, unlike codes, *are* persisted (the gitignored
`.data/oauth-clients.json`), so a client's `client_id` survives a
server restart without needing to re-register. One known gap from
this simplification: refresh "rotation" issues a new refresh token but
can't revoke the old one without a persisted token store. Acceptable
for a single-user tool; would need real storage to harden further.

Verified end-to-end, not just endpoint-by-endpoint, in
`test/oauth-flow.sh`: the full DCR -> consent -> code exchange -> `/mcp`
call -> refresh chain, plus the negatives that matter (wrong PKCE
verifier, a replayed code, an expired code, and -- through the real
built `proxy.cjs`, not just the server directly -- a client trying to
forge the "I'm local" marker header). Run it with `bash
test/oauth-flow.sh` after `npm install`; it builds, spins up a
throwaway server and proxy against a disposable mailbox root, and
tears both down on exit.
