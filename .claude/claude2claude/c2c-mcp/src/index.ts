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
// createMcpExpressApp already applies express.json() globally; OAuth's
// /token and /authorize (form-submitted consent) both need urlencoded too.
app.use(express.urlencoded({ extended: false }));

app.get("/health", (_req, res) => {
  res.json({ ok: true, repoRoot: REPO_ROOT });
});

let requireBearerAuth: express.RequestHandler = (_req, _res, next) => next();
if (PUBLIC_MCP_URL) {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const signingKeyPath = path.join(here, "..", ".data", "oauth-signing-key");
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
