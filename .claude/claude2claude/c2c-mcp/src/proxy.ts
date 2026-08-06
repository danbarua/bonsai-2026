import http from "node:http";

/**
 * Standalone reverse proxy: forwards every incoming HTTP request to the
 * c2c-mcp server. Deployed on a cloud VM (public port 80) with the c2c-mcp
 * server itself kept on the developer's machine, reachable on the VM only
 * as `localhost:8765` via a reverse SSH tunnel (`ssh -R 8765:127.0.0.1:8765
 * <vm>`). Zero dependencies outside node:http, so the compiled output
 * (`npm run build-proxy` -> dist-proxy/proxy.js, CommonJS) runs standalone
 * on the VM with no npm install step: copy the one file, `node proxy.js`.
 */

const LISTEN_HOST = process.env.C2C_PROXY_LISTEN_HOST ?? "0.0.0.0";
const LISTEN_PORT = Number(process.env.C2C_PROXY_LISTEN_PORT ?? 80);
const TARGET_HOST = process.env.C2C_PROXY_TARGET_HOST ?? "127.0.0.1";
const TARGET_PORT = Number(process.env.C2C_PROXY_TARGET_PORT ?? 8765);

// Hop-by-hop headers (RFC 7230 6.1) never get forwarded across a proxy leg.
// content-length/transfer-encoding are dropped too, deliberately: both legs
// re-derive their own framing from how bytes are actually written, which is
// what makes unbuffered piping -- including an SSE response with no known
// length -- come out correct on the other side.
const HOP_BY_HOP = new Set([
  "connection",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailers",
  "transfer-encoding",
  "upgrade",
  "content-length",
]);

function filteredHeaders(headers: http.IncomingHttpHeaders): http.OutgoingHttpHeaders {
  const out: http.OutgoingHttpHeaders = {};
  for (const [key, value] of Object.entries(headers)) {
    if (value === undefined || HOP_BY_HOP.has(key.toLowerCase())) continue;
    out[key] = value;
  }
  return out;
}

const server = http.createServer((clientReq, clientRes) => {
  const headers = filteredHeaders(clientReq.headers);
  // c2c-mcp's DNS-rebinding guard checks the Host header's hostname against
  // localhost/127.0.0.1/[::1]. Forward the *target's* host here, not
  // whatever public host/IP the client actually connected to -- otherwise
  // every request gets rejected 403 by the backend, not proxied at all.
  headers.host = `${TARGET_HOST}:${TARGET_PORT}`;

  const proxyReq = http.request(
    {
      host: TARGET_HOST,
      port: TARGET_PORT,
      method: clientReq.method,
      path: clientReq.url,
      headers,
    },
    (proxyRes) => {
      const responseHeaders = filteredHeaders(proxyRes.headers);
      clientRes.writeHead(proxyRes.statusCode ?? 502, proxyRes.statusMessage, responseHeaders);
      proxyRes.pipe(clientRes, { end: true });
    },
  );

  proxyReq.on("error", (err) => {
    console.error(`[c2c-proxy] upstream error for ${clientReq.method} ${clientReq.url}: ${err.message}`);
    if (!clientRes.headersSent) {
      clientRes.writeHead(502, { "content-type": "text/plain" });
    }
    clientRes.end("Bad gateway");
  });

  // If either side hangs up early (client navigates away mid-SSE-stream,
  // upstream drops the connection), tear down the other leg instead of
  // leaking a half-open socket.
  clientReq.on("error", () => proxyReq.destroy());
  clientRes.on("close", () => {
    if (!clientRes.writableEnded) proxyReq.destroy();
  });

  clientReq.pipe(proxyReq, { end: true });
});

server.on("clientError", (err, socket) => {
  if (socket.writable) {
    socket.end("HTTP/1.1 400 Bad Request\r\n\r\n");
  }
});

server.listen(LISTEN_PORT, LISTEN_HOST, () => {
  console.log(
    `c2c-proxy listening on http://${LISTEN_HOST}:${LISTEN_PORT} -> http://${TARGET_HOST}:${TARGET_PORT}`,
  );
});

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.on(signal, () => {
    server.close(() => process.exit(0));
  });
}
