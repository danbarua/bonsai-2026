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

Read `DEVELOPMENT_PRACTICES.md` before making changes here -- the
version-bump-before-restart discipline referenced throughout this
README, and several other hard-won rules about the multiple ways this
server gets reached, are explained there with the incidents that
produced them.

## Tools

Each channel has an internal ("code-side") set of roles and one
external peer role, and the directories are named from the code side
(matching the existing protocol docs): `outbox/` is what a code-side
role writes and the peer (`claude-desktop` or `chatgpt`) reads;
`inbox/` is the reverse. Both `-send` and `-inbox` take an identity
argument that decides which directory they touch -- neither tool has
a silent default for "which side am I," since that ambiguity is
exactly what caused a real bug here once (see git log).

c2c's code side is `claude-code` alone (its peer, `claude-desktop`, is
the OTHER party on this channel). c2gpt's code side is BOTH
`claude-code` and `claude-desktop` -- either may message ChatGPT
directly, sharing the same `outbox/`/`inbox/`; `chatgpt` is the sole
peer. This isn't symmetric with c2c: on c2gpt, `claude-desktop` is a
code-side role, not the peer -- the intended mail topology overall is
ChatGPT <-> Claude Desktop <-> Claude Code (Desktop relays into
`claude2claude/` for Code's benefit), but Code and Desktop can each
also reach ChatGPT directly on `claude2gpt/` without going through
each other first.

| Tool | Channel dir | Roles | Effect |
|---|---|---|---|
| `c2c-send` | `.claude/claude2claude/` | `claude-code`, `claude-desktop` | `{ sender, content }` -- `claude-code` writes `outbox/`, `claude-desktop` writes `inbox/` |
| `c2c-inbox` | `.claude/claude2claude/` | `claude-code`, `claude-desktop` | `{ reader, archive? }` -- `claude-code` reads `inbox/`, `claude-desktop` reads `outbox/` |
| `c2gpt-send` | `.claude/claude2gpt/` | `claude-code`, `claude-desktop`, `chatgpt` | `{ sender, content }` -- `claude-code`/`claude-desktop` (either) write `outbox/`, `chatgpt` writes `inbox/` |
| `c2gpt-inbox` | `.claude/claude2gpt/` | `claude-code`, `claude-desktop`, `chatgpt` | `{ reader, archive? }` -- `claude-code`/`claude-desktop` (either) read `inbox/`, `chatgpt` reads `outbox/` |
| `code2code-send` | `.claude/code2code/` | none -- always `claude-code` | `{ instance, content, to? }` -- writes `mailbox/` |
| `code2code-inbox` | `.claude/code2code/` | none -- always `claude-code` | `{ as, archive? }` -- reads `mailbox/` |
| `code2code-archive` | `.claude/code2code/` | none | `{ filename }` -- moves one named file to `archive/` unconditionally |

`code2code` is a different shape from `c2c`/`c2gpt`: every party is a
Claude Code session, so there's no peer role and no `outbox`/`inbox`
split -- `mailbox/` is a SINGLE shared directory, both written and read
by every session. `instance` (send) and `as` (inbox) are therefore
REQUIRED, not optional: there's no non-Code peer to fall back to an
unidentified sender/reader for, and a PreToolUse hook auto-injects both
so a well-behaved caller never has to pass them explicitly (see
`.claude/hooks/c2c-mail/pre-c2c-mcp.sh`). Because the mailbox is shared,
a session's own unaddressed broadcast is automatically excluded from
its own `code2code-inbox` reads (so it can never consume its own
just-sent announcement before anyone else sees it) -- which also means
that broadcast can never be archived by a normal read from the sender
either. `code2code-archive` is the deliberate escape hatch for that:
pass the exact filename to retract/clean up a stale broadcast (or any
other message) directly, no addressing rules applied.

`-send`'s `content` is just the message body; the header comment is
generated from `sender` and the current UTC time. A same-second
filename collision gets a `-2`, `-3`, ... suffix rather than
overwriting anything.

Pass `instance` on `-send` when `sender` is `"claude-code"` to say
*which* Claude Code session sent it (its `/rename`-set name, e.g. from
`code-sessions` -- never the raw `session_id`, an opaque GUID not
useful to a reader). `.claude/hooks/c2c-mail/pre-c2c-mcp.sh` resolves
and surfaces this session's own name as `additionalContext` right
before every c2c-mcp tool call, specifically so it can be passed here.
`instance` shows up three places: the exact name in the header comment
(`· instance: <name>`), a slugified version appended to the *filename*
itself (`2026-08-07T21-30-00Z-c2c-implementation.md`, spaces and other
non-filename characters collapsed to hyphens) so a reader can filter
by sender with a plain `ls`/glob without opening any files, and in
`-inbox`'s response (both the `(instance: <name>)` text annotation and
`structuredContent`). Omit it and nothing changes from before this
field existed -- no `instance:` in the header, no slug in the
filename.

Pass `to` on `-send` to address a message to one specific reader (e.g.
a `/rename`-set Claude Code session name from `code-sessions`). It's
written to the header as `· to: <name>` (exact, unslugified) AND,
slugified, to the filename as a `--to-<slug>` tag -- a DOUBLE hyphen,
distinct from `instance`'s single-hyphen suffix, so the two can never
collide even when both are present on the same message
(`2026-08-07T21-30-00Z-c2c-implementation--to-reader-b.md`). Omitted
(the default), a message is a broadcast any reader may consume,
matching the exact behavior before addressing existed.

Pass `as` on `-inbox` to say who's reading (again, the reader's own
`/rename`-set name). It only changes behavior on a *consuming* read
(`archive: true`, the default): a message addressed to a DIFFERENT
name is skipped entirely -- left in place, unarchived, for its real
addressee -- rather than being silently consumed by whoever happened
to call `-inbox` first. Filtering checks the filename's `--to-<slug>`
tag first (no file read needed); a message with no such tag (broadcast,
or sent before this filename convention existed) falls back to the
header's `to:` field, so nothing already in a mailbox needs migrating.
A peek (`archive: false`) always returns everything regardless of
addressing, since peeking doesn't consume anything. Omitting `as`
entirely preserves the exact pre-addressing behavior: every message
visible and archivable, including ones addressed to someone else.

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

**Bump `package.json`'s `version` before restarting with fresh
changes.** `/health` and the MCP `initialize` handshake's
`serverInfo.version` both report it live (`PKG_VERSION` in
`mailbox.ts`, read from `package.json` at startup, not hardcoded --
see git history for why: an earlier hardcoded version string meant a
restart with new code silently kept reporting the old number). A
version bump is the fast, unambiguous way to tell "this process is
actually running what I just built" from "this process didn't
actually restart" or "this connection is going through a stale cached
route" -- exactly the ambiguity that once led to a real mis-consumed
message on a different Claude Code session's mailbox: an MCP
connection routed through a stale deployment silently lacked a
parameter (`as`) that a fresh build already had, and there was no
version-number tell to catch it before the mistake happened, only
after.

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

**Upgrading the server does not upgrade sessions already attached to
it.** A client's tool schema is pinned at connection time, not polled
continuously -- a session connected before a server rebuild keeps
running the old contract indefinitely, with no error, until it's
reconnected (`/mcp` in Claude Code). This is most dangerous when the
upgrade adds a safety parameter (e.g. `as` on `-inbox`): the session
silently keeps the old, more-permissive behavior rather than failing
loudly. Self-check: if a parameter or tool you expect the server to
have isn't in your own attached schema, your connection is stale, not
the server -- see `DEVELOPMENT_PRACTICES.md` for the full account
(confirmed independently by two sessions the same night, via two
different methods).

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

The `ssh -R` command above is illustrative; `run-c2c-mcp.sh` is the
actual dev-loop launcher used in practice, and automates it with
retry/keepalive (see the script's own comments) plus a real `/health`
poll before opening the tunnel, rather than a blind sleep. Its two
ports are env vars, not hardcoded: `C2C_MCP_PORT` (local server,
default `8765`) and `C2C_TUNNEL_REMOTE_PORT` (the VM-side port,
default `8767` -- **must match whatever `C2C_PROXY_TARGET_PORT` the
proxy is actually started with on the VM**, since the two sides don't
discover each other). The remote port has had to move twice already
(`8765` -> `8766` -> `8767`): an abruptly-killed SSH client leaves the
remote sshd holding the old binding open, with no `lsof`/`ss`/`fuser`/
`sudo` available on that host to clear it, so bumping both this env
var and the VM-side proxy invocation is the actual fix each time this
recurs.

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
