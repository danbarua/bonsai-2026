import path from "node:path";
import { fileURLToPath } from "node:url";
import express from "express";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { createMcpExpressApp } from "@modelcontextprotocol/sdk/server/express.js";
import { createServer } from "./server.js";
import { REPO_ROOT } from "./mailbox.js";
import { mountOAuth } from "./oauth.js";

const HOST = process.env.C2C_MCP_HOST ?? "127.0.0.1";
const PORT = Number(process.env.C2C_MCP_PORT ?? 8765);
// Set only when this server is reachable publicly (through src/proxy.ts).
// Must be exactly what gets pasted into Claude's connector dialog, e.g.
// https://c2c.framesift.ai/mcp -- when unset, /mcp stays fully authless
// (the pre-OAuth default), matching local-only usage with no exposure.
const PUBLIC_MCP_URL = process.env.C2C_MCP_PUBLIC_URL;

const app = createMcpExpressApp({ host: HOST });

// A tools/call validation failure (unknown/missing enum value, etc.) isn't
// an HTTP-level error at all -- the SDK reports it as an ordinary 200
// response with the failure embedded in the JSON-RPC result body
// (`result.isError: true`), by design (it's a tool-level outcome, not a
// transport-level one). Extracted here so it can be checked against the
// buffered response body alongside the HTTP-status check below.
function findJsonRpcError(body: string): string | undefined {
  for (const line of body.split("\n")) {
    if (!line.startsWith("data:")) continue;
    let parsed: unknown;
    try {
      parsed = JSON.parse(line.slice("data:".length).trim());
    } catch {
      continue;
    }
    const msg = parsed as { error?: { message?: string }; result?: { isError?: boolean; content?: { text?: string }[] } };
    if (msg.error) return msg.error.message ?? "JSON-RPC error";
    if (msg.result?.isError) return msg.result.content?.[0]?.text ?? "tool call reported isError";
  }
  return undefined;
}

// Logs every response that's an error to stderr, regardless of which route
// produced it -- mounted before any route (including OAuth's, registered
// later by mountOAuth), so one line here covers all of them instead of
// adding logging to each handler individually. Covers both HTTP-level
// errors (4xx/5xx) and MCP's tool-level errors (see findJsonRpcError
// above), by buffering the response body to inspect after it's sent.
// Doesn't capture rejections from createMcpExpressApp's own DNS-rebinding
// guard, which runs before this middleware is reached on the same app
// instance.
function logErrorRequests(req: express.Request, res: express.Response, next: express.NextFunction): void {
  const chunks: Buffer[] = [];
  const originalWrite = res.write.bind(res);
  const originalEnd = res.end.bind(res);
  // The transport writes raw Uint8Array chunks, not Node Buffers or strings
  // -- Buffer.isBuffer() is false for a plain Uint8Array (despite Buffer
  // extending it), and String(uint8Array) joins byte *values* with commas
  // rather than UTF-8-decoding them. Buffer.from() handles all three
  // correctly; String()/.toString() would silently corrupt the capture.
  const toBuffer = (chunk: unknown): Buffer =>
    Buffer.isBuffer(chunk) ? chunk : chunk instanceof Uint8Array ? Buffer.from(chunk) : Buffer.from(String(chunk));

  res.write = ((chunk: unknown, ...rest: unknown[]) => {
    if (chunk) chunks.push(toBuffer(chunk));
    return (originalWrite as (...a: unknown[]) => boolean)(chunk, ...rest);
  }) as typeof res.write;

  res.end = ((chunk?: unknown, ...rest: unknown[]) => {
    if (chunk && typeof chunk !== "function") chunks.push(toBuffer(chunk));
    return (originalEnd as (...a: unknown[]) => express.Response)(chunk, ...rest);
  }) as typeof res.end;

  res.on("finish", () => {
    const httpError = res.statusCode >= 400;
    const jsonRpcErrorDetail = chunks.length > 0 ? findJsonRpcError(Buffer.concat(chunks).toString("utf8")) : undefined;
    // The JSON-RPC `method` field (e.g. "resources/list") is what actually
    // identifies what a client was trying to do -- without it, a logged
    // "Method not found" only tells you *that* something failed, not what
    // to go add a handler for. req.body is already parsed here (express.json()
    // runs inside createMcpExpressApp, before this middleware).
    const method = typeof req.body?.method === "string" ? ` (method: ${req.body.method})` : "";
    if (httpError) {
      console.error(`[c2c-mcp] ${req.method} ${req.originalUrl} -> HTTP ${res.statusCode}${method}`);
    } else if (jsonRpcErrorDetail) {
      console.error(`[c2c-mcp] ${req.method} ${req.originalUrl} -> ${jsonRpcErrorDetail}${method}`);
    } else {
      // Successful requests, to stdout rather than stderr -- kept separate
      // from the error stream, but genuinely needed: "the connector says
      // Connected but shows no actions" is not distinguishable from "ChatGPT
      // never actually called tools/list" without this. A silent error log
      // only rules out failures, not silence.
      console.log(`[c2c-mcp] ${req.method} ${req.originalUrl} -> ${res.statusCode}${method}`);
    }
  });
  next();
}
app.use(logErrorRequests);

// createMcpExpressApp already applies express.json() globally; OAuth's
// /token and /authorize (form-submitted consent) both need urlencoded too.
app.use(express.urlencoded({ extended: false }));

app.get("/health", (_req, res) => {
  res.json({ ok: true, repoRoot: REPO_ROOT });
});

let requireBearerAuth: express.RequestHandler = (_req, _res, next) => next();
if (PUBLIC_MCP_URL) {
  // Overridable so tests (and any throwaway server) never share the real
  // signing key / persisted DCR client registry with a live deployment --
  // both live in this same directory, keyed off the running script's own
  // location by default, same as BONSAI_PROJECT_ROOT's role for mailbox
  // data. Without this, a test run against the real dist/index.js would
  // read and overwrite the live server's actual OAuth state.
  const here = path.dirname(fileURLToPath(import.meta.url));
  const dataDir = process.env.C2C_OAUTH_DATA_DIR ?? path.join(here, "..", ".data");
  const signingKeyPath = path.join(dataDir, "oauth-signing-key");
  ({ requireBearerAuth } = mountOAuth(app, { publicMcpUrl: PUBLIC_MCP_URL, signingKeyPath }));
  console.log(`OAuth enabled for public traffic, resource=${PUBLIC_MCP_URL}`);
} else {
  console.log("C2C_MCP_PUBLIC_URL not set -- OAuth routes not mounted, /mcp stays authless.");
}

// Stateless: every request gets a fresh server + transport pair. Each tool
// call here is a self-contained filesystem read/write, so there's no
// session state worth keeping alive between requests.
app.post("/mcp", requireBearerAuth, async (req, res) => {
  const server = createServer();
  try {
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
    });
    res.on("close", () => {
      transport.close();
      server.close();
    });
    await server.connect(transport);
    await transport.handleRequest(req, res, req.body);
  } catch (err) {
    console.error("Error handling MCP request:", err);
    if (!res.headersSent) {
      res.status(500).json({
        jsonrpc: "2.0",
        error: { code: -32603, message: "Internal server error" },
        id: null,
      });
    }
  }
});

const methodNotAllowed = (_req: import("express").Request, res: import("express").Response) => {
  res.status(405).json({
    jsonrpc: "2.0",
    error: { code: -32000, message: "Method not allowed." },
    id: null,
  });
};
app.get("/mcp", methodNotAllowed);
app.delete("/mcp", methodNotAllowed);

app.listen(PORT, HOST, () => {
  console.log(`c2c-mcp listening on http://${HOST}:${PORT}/mcp (repo root: ${REPO_ROOT})`);
});
