import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import type { Express, NextFunction, Request, Response } from "express";

/**
 * A minimal single-user OAuth 2.1 authorization server (DCR + PKCE),
 * bolted onto c2c-mcp's own Express app so a public deployment (through
 * src/proxy.ts) can satisfy Claude Desktop/iOS's connector requirement
 * without a separate identity service. Scoped deliberately narrow: one
 * consenting user, no client secrets (public clients only, PKCE S256
 * required), no CIMD support (Claude Code's own loopback-redirect flow
 * isn't handled -- see requireBearerAuth below for why that's fine).
 *
 * Endpoints, per the current Claude connector auth docs
 * (claude.com/docs/connectors/building/authentication) and the
 * referenced RFCs:
 *   GET  /.well-known/oauth-protected-resource   (RFC 9728)
 *   GET  /.well-known/oauth-authorization-server (RFC 8414)
 *   POST /register                               (RFC 7591, DCR)
 *   GET/POST /authorize                          (auth code + consent)
 *   POST /token                                  (code exchange + refresh)
 */

// Set by src/proxy.ts on every request it forwards (after stripping any
// client-supplied copy first) -- distinguishes public/tunnelled traffic
// from same-machine callers (Claude Code's own .mcp.json, local curl),
// which stay authless since 127.0.0.1 binding is already their security
// boundary. See requireBearerAuth.
export const PROXY_MARKER_HEADER = "x-c2c-via-proxy";

const ACCESS_TTL_S = 60 * 60; // 1 hour
const REFRESH_TTL_S = 60 * 60 * 24 * 30; // 30 days
const CODE_TTL_MS = 60 * 1000; // 60 seconds, matches Claude's own auth-code lifetime expectations

interface TokenClaims {
  sub: string;
  client_id: string;
  resource: string;
  scope: string;
  kind: "access" | "refresh";
  exp: number; // unix seconds
}

function loadOrCreateSigningKey(keyPath: string): Buffer {
  try {
    return fs.readFileSync(keyPath);
  } catch {
    const key = crypto.randomBytes(32);
    fs.mkdirSync(path.dirname(keyPath), { recursive: true });
    fs.writeFileSync(keyPath, key, { mode: 0o600 });
    return key;
  }
}

// Persisted for the same reason tokens are (see signToken above), but
// discovered as a real gap rather than anticipated: registered DCR clients
// were originally in-memory only on the assumption that a client re-runs
// POST /register on every reconnect, so losing the registry across a
// restart would be self-healing. ChatGPT's connector doesn't do that -- it
// caches the client_id from an earlier registration and reuses it directly
// against /authorize, so a server restart orphaned a real, already-
// registered client and broke reconnection with "Invalid authorization
// request." Small JSON file, same directory as the signing key.
function loadClients(clientsPath: string): Map<string, ClientRecord> {
  try {
    const raw = JSON.parse(fs.readFileSync(clientsPath, "utf8")) as ClientRecord[];
    return new Map(raw.map((c) => [c.client_id, c]));
  } catch {
    return new Map();
  }
}

function saveClients(clientsPath: string, clients: Map<string, ClientRecord>): void {
  fs.mkdirSync(path.dirname(clientsPath), { recursive: true });
  fs.writeFileSync(clientsPath, JSON.stringify([...clients.values()], null, 2));
}

// Stateless tokens (payload + HMAC signature, both base64url) so access
// and refresh tokens survive a server restart without any persisted
// store -- the whole point of this server is keeping a long-running
// claude.ai chat connected while the local dev server gets restarted.
// Known limitation: refresh "rotation" issues a new token but can't
// revoke the old one without a store; acceptable for a single-user tool.
function signToken(claims: TokenClaims, key: Buffer): string {
  const payload = Buffer.from(JSON.stringify(claims)).toString("base64url");
  const sig = crypto.createHmac("sha256", key).update(payload).digest("base64url");
  return `${payload}.${sig}`;
}

function verifyToken(token: string, key: Buffer): TokenClaims | null {
  const parts = token.split(".");
  if (parts.length !== 2) return null;
  const [payload, sig] = parts;
  const expected = crypto.createHmac("sha256", key).update(payload).digest("base64url");
  const a = Buffer.from(sig);
  const b = Buffer.from(expected);
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
  try {
    const claims = JSON.parse(Buffer.from(payload, "base64url").toString("utf8")) as TokenClaims;
    if (typeof claims.exp !== "number" || claims.exp < Math.floor(Date.now() / 1000)) return null;
    return claims;
  } catch {
    return null;
  }
}

function randomId(): string {
  return crypto.randomBytes(24).toString("base64url");
}

function pkceMatches(verifier: string, challenge: string): boolean {
  const computed = crypto.createHash("sha256").update(verifier).digest("base64url");
  const a = Buffer.from(computed);
  const b = Buffer.from(challenge);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

function escapeHtml(s: string): string {
  const map: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
  return s.replace(/[&<>"']/g, (c) => map[c] ?? c);
}

interface ClientRecord {
  client_id: string;
  redirect_uris: string[];
}

interface AuthCodeRecord {
  client_id: string;
  redirect_uri: string;
  code_challenge: string;
  resource: string;
  scope: string;
  expiresAt: number;
}

export interface MountOAuthOptions {
  publicMcpUrl: string; // must exactly match what's pasted into Claude's connector dialog, e.g. https://c2c.framesift.ai/mcp
  signingKeyPath: string;
}

export function mountOAuth(app: Express, opts: MountOAuthOptions): { requireBearerAuth: RequireBearerAuth } {
  const key = loadOrCreateSigningKey(opts.signingKeyPath);
  const resourceUrl = new URL(opts.publicMcpUrl);
  const issuer = `${resourceUrl.protocol}//${resourceUrl.host}`;
  const protectedResourceMetadataUrl = `${issuer}/.well-known/oauth-protected-resource`;

  // Persisted (see loadClients/saveClients above) -- a restart must not
  // orphan an already-registered client. Auth codes stay in-memory: a
  // restart inside their 60s TTL is an acceptable edge case, and losing one
  // is self-healing (the client just retries /authorize from scratch).
  const clientsPath = path.join(path.dirname(opts.signingKeyPath), "oauth-clients.json");
  const clients = loadClients(clientsPath);
  const authCodes = new Map<string, AuthCodeRecord>();

  const servePrm = (_req: Request, res: Response) => {
    res.json({
      resource: opts.publicMcpUrl,
      authorization_servers: [issuer],
    });
  };
  app.get("/.well-known/oauth-protected-resource", servePrm);
  // Alias for the RFC 9728 resource-specific well-known path convention
  // (.well-known/oauth-protected-resource/<mcp-path>). Not required for our
  // own flow -- the WWW-Authenticate header's resource_metadata pointer on
  // our 401 is what clients are supposed to follow -- but ChatGPT's
  // connector was observed probing this path too (see logs/err.log from an
  // earlier session) as a fallback, so it's cheap to serve the same
  // document here rather than leave it 404.
  app.get(`/.well-known/oauth-protected-resource${resourceUrl.pathname}`, servePrm);

  // We're an OAuth 2.1 AS, not a real OpenID Connect provider -- no id_token
  // is ever issued (response_types_supported below is just "code", no
  // "openid" scope offered). Served at both the RFC 8414 path and the OIDC
  // Discovery 1.0 path anyway: found live that a client (ChatGPT's
  // connector) probes /.well-known/openid-configuration unconditionally as
  // an early discovery step and aborts hard on 404 without ever falling
  // back to plain OAuth metadata, even though the MCP spec only requires a
  // server to serve *one* of the two. subject_types_supported and
  // id_token_signing_alg_values_supported are included purely because
  // OpenID Connect Discovery 1.0 lists them as required fields for a
  // conforming document -- they don't imply we actually support signed ID
  // tokens; jwks_uri truthfully serves an empty keyset for the same reason.
  const serveAsMetadata = (_req: Request, res: Response) => {
    res.json({
      issuer,
      authorization_endpoint: `${issuer}/authorize`,
      token_endpoint: `${issuer}/token`,
      registration_endpoint: `${issuer}/register`,
      jwks_uri: `${issuer}/.well-known/jwks.json`,
      response_types_supported: ["code"],
      grant_types_supported: ["authorization_code", "refresh_token"],
      code_challenge_methods_supported: ["S256"],
      subject_types_supported: ["public"],
      id_token_signing_alg_values_supported: ["RS256"],
      // Public clients only (no client_secret) -- DCR never issues one below.
      token_endpoint_auth_methods_supported: ["none"],
    });
  };
  app.get("/.well-known/oauth-authorization-server", serveAsMetadata);
  app.get("/.well-known/openid-configuration", serveAsMetadata);
  app.get("/.well-known/jwks.json", (_req, res) => res.json({ keys: [] }));

  app.post("/register", (req, res) => {
    const redirectUris: unknown = req.body?.redirect_uris;
    if (!Array.isArray(redirectUris) || redirectUris.length === 0 || !redirectUris.every((u) => typeof u === "string")) {
      res.status(400).json({ error: "invalid_client_metadata", error_description: "redirect_uris is required" });
      return;
    }
    const clientId = randomId();
    clients.set(clientId, { client_id: clientId, redirect_uris: redirectUris });
    saveClients(clientsPath, clients);
    res.status(201).json({
      client_id: clientId,
      client_id_issued_at: Math.floor(Date.now() / 1000),
      redirect_uris: redirectUris,
      grant_types: ["authorization_code", "refresh_token"],
      response_types: ["code"],
      token_endpoint_auth_method: "none",
    });
  });

  function renderConsent(res: Response, fields: Record<string, string>) {
    const hidden = Object.entries(fields)
      .map(([k, v]) => `<input type="hidden" name="${escapeHtml(k)}" value="${escapeHtml(v)}">`)
      .join("\n");
    res.type("html").send(`<!doctype html>
<html><body style="font-family: system-ui, sans-serif; max-width: 28rem; margin: 4rem auto; text-align: center;">
<h2>Authorize c2c-mcp</h2>
<p>A client is requesting access to<br><code>${escapeHtml(opts.publicMcpUrl)}</code>.</p>
<form method="POST" action="/authorize">
${hidden}
<button type="submit" style="font-size: 1rem; padding: 0.6rem 1.8rem; cursor: pointer;">Allow</button>
</form>
</body></html>`);
  }

  function handleAuthorize(req: Request, res: Response) {
    const q: Record<string, unknown> = req.method === "GET" ? req.query : req.body;
    const responseType = String(q.response_type ?? "");
    const clientId = String(q.client_id ?? "");
    const redirectUri = String(q.redirect_uri ?? "");
    const codeChallenge = String(q.code_challenge ?? "");
    const codeChallengeMethod = String(q.code_challenge_method ?? "");
    const resource = String(q.resource ?? opts.publicMcpUrl);
    const state = q.state !== undefined ? String(q.state) : undefined;
    const scope = q.scope !== undefined ? String(q.scope) : "";

    const client = clients.get(clientId);
    if (
      responseType !== "code" ||
      !client ||
      !client.redirect_uris.includes(redirectUri) ||
      codeChallengeMethod !== "S256" ||
      !codeChallenge
    ) {
      res.status(400).send("Invalid authorization request.");
      return;
    }

    if (req.method === "GET") {
      renderConsent(res, {
        response_type: responseType,
        client_id: clientId,
        redirect_uri: redirectUri,
        code_challenge: codeChallenge,
        code_challenge_method: codeChallengeMethod,
        resource,
        scope,
        ...(state !== undefined ? { state } : {}),
      });
      return;
    }

    // POST: consent form submitted -- issue a single-use code and redirect back.
    const code = randomId();
    authCodes.set(code, {
      client_id: clientId,
      redirect_uri: redirectUri,
      code_challenge: codeChallenge,
      resource,
      scope,
      expiresAt: Date.now() + CODE_TTL_MS,
    });
    const redirect = new URL(redirectUri);
    redirect.searchParams.set("code", code);
    if (state !== undefined) redirect.searchParams.set("state", state);
    res.redirect(303, redirect.toString());
  }

  app.get("/authorize", handleAuthorize);
  app.post("/authorize", handleAuthorize);

  app.post("/token", (req, res) => {
    const grantType = req.body?.grant_type;
    const now = Math.floor(Date.now() / 1000);

    if (grantType === "authorization_code") {
      const code = String(req.body?.code ?? "");
      const record = authCodes.get(code);
      if (!record || record.expiresAt < Date.now()) {
        authCodes.delete(code);
        res.status(400).json({ error: "invalid_grant" });
        return;
      }
      const redirectUri = String(req.body?.redirect_uri ?? "");
      const clientId = String(req.body?.client_id ?? "");
      const verifier = String(req.body?.code_verifier ?? "");
      // Single-use regardless of outcome: delete before validating so a
      // replayed/racing request can't redeem the same code twice.
      authCodes.delete(code);
      if (
        redirectUri !== record.redirect_uri ||
        clientId !== record.client_id ||
        !verifier ||
        !pkceMatches(verifier, record.code_challenge)
      ) {
        res.status(400).json({ error: "invalid_grant" });
        return;
      }

      const accessToken = signToken(
        { sub: "dan", client_id: clientId, resource: record.resource, scope: record.scope, kind: "access", exp: now + ACCESS_TTL_S },
        key,
      );
      const refreshToken = signToken(
        { sub: "dan", client_id: clientId, resource: record.resource, scope: record.scope, kind: "refresh", exp: now + REFRESH_TTL_S },
        key,
      );
      res.json({
        access_token: accessToken,
        token_type: "Bearer",
        expires_in: ACCESS_TTL_S,
        refresh_token: refreshToken,
        scope: record.scope,
      });
      return;
    }

    if (grantType === "refresh_token") {
      const refreshToken = String(req.body?.refresh_token ?? "");
      const claims = verifyToken(refreshToken, key);
      if (!claims || claims.kind !== "refresh") {
        res.status(400).json({ error: "invalid_grant" });
        return;
      }
      const accessToken = signToken({ ...claims, kind: "access", exp: now + ACCESS_TTL_S }, key);
      const newRefreshToken = signToken({ ...claims, kind: "refresh", exp: now + REFRESH_TTL_S }, key);
      res.json({
        access_token: accessToken,
        token_type: "Bearer",
        expires_in: ACCESS_TTL_S,
        refresh_token: newRefreshToken,
        scope: claims.scope,
      });
      return;
    }

    res.status(400).json({ error: "unsupported_grant_type" });
  });

  const requireBearerAuth: RequireBearerAuth = (req, res, next) => {
    // Same-machine callers (Claude Code's .mcp.json, local curl) never carry
    // this header -- only src/proxy.ts sets it, and only after stripping any
    // client-supplied copy. 127.0.0.1 binding is already their trust
    // boundary, so they stay authless; only traffic that came in through the
    // public proxy gets challenged.
    const viaProxy = req.headers[PROXY_MARKER_HEADER] === "1";
    if (!viaProxy) {
      next();
      return;
    }
    const authHeader = req.headers.authorization ?? "";
    const match = /^Bearer\s+(.+)$/i.exec(authHeader);
    const claims = match ? verifyToken(match[1], key) : null;
    if (!claims || claims.kind !== "access") {
      res
        .status(401)
        .set("WWW-Authenticate", `Bearer resource_metadata="${protectedResourceMetadataUrl}"`)
        .json({ jsonrpc: "2.0", error: { code: -32001, message: "Unauthorized" }, id: null });
      return;
    }
    next();
  };

  return { requireBearerAuth };
}

type RequireBearerAuth = (req: Request, res: Response, next: NextFunction) => void;
