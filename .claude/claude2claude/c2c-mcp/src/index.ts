import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { createMcpExpressApp } from "@modelcontextprotocol/sdk/server/express.js";
import { createServer } from "./server.js";
import { REPO_ROOT } from "./mailbox.js";

const HOST = process.env.C2C_MCP_HOST ?? "127.0.0.1";
const PORT = Number(process.env.C2C_MCP_PORT ?? 8765);

const app = createMcpExpressApp({ host: HOST });

app.get("/health", (_req, res) => {
  res.json({ ok: true, repoRoot: REPO_ROOT });
});

// Stateless: every request gets a fresh server + transport pair. Each tool
// call here is a self-contained filesystem read/write, so there's no
// session state worth keeping alive between requests.
app.post("/mcp", async (req, res) => {
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
