import crypto from "node:crypto";
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

// Must match PROXY_MARKER_HEADER in src/oauth.ts. Duplicated as a literal
// (not imported) so this file keeps its zero-dependency, standalone build --
// importing oauth.ts would pull in express's types for a build that's meant
// to run on the VM with nothing but node:http.
const PROXY_MARKER_HEADER = "x-c2c-via-proxy";

// Set ONLY by this proxy, and only after a signature verified. The backend
// treats it as proof the request really came from the named provider.
//
// Unlike PROXY_MARKER_HEADER, which is overwritten unconditionally below and
// is therefore safe against a client sending its own copy, this one is
// CONDITIONAL -- so it has to be deleted explicitly before the conditional
// set. filteredHeaders forwards every non-hop-by-hop client header, so
// without the delete a caller could simply send `x-c2c-verified: github`
// themselves and the backend would believe the proxy vouched for it.
//
// The value names the provider rather than being a bare "1", so adding a
// second webhook source later can't silently widen an existing boolean.
const VERIFIED_HEADER = "x-c2c-verified";

// GitHub-specific, deliberately not a generic "verify any webhook" layer.
// Every provider signs differently -- header name, algorithm, what exactly
// is covered -- so each gets its own verifier rather than one abstraction
// that fits none of them properly.
const GITHUB_SIG_HEADER = "x-hub-signature-256";
const GITHUB_SECRET = process.env.C2C_GITHUB_WEBHOOK_SECRET ?? "";

/**
 * Reads a positive integer from the environment, falling back loudly.
 *
 * `Number("25MB")` is NaN, and `size > NaN` is ALWAYS FALSE -- so a
 * plausible human value silently switches off the very guard it was meant
 * to configure. Fail-open, on a memory-DoS bound, from a typo. `Number("")`
 * and `Number("0")` are 0, which rejects every request instead: safe, but
 * equally not what anyone meant.
 *
 * Anything that is not a finite positive integer is refused and the default
 * used, with a line saying so.
 */
function positiveIntEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || !Number.isInteger(value) || value <= 0) {
    console.error(
      `[c2c-proxy] ${name}="${raw}" is not a positive integer number of bytes; using ${fallback}`,
    );
    return fallback;
  }
  return value;
}

// Verifying means buffering, and buffering an arbitrary body is a memory
// DoS. Measured in BYTES. GitHub's own delivery limit is 25 MB; anything
// larger is rejected without being read into memory.
const MAX_WEBHOOK_BYTES = positiveIntEnv("C2C_WEBHOOK_MAX_BYTES", 25 * 1024 * 1024);

/**
 * GitHub signs the RAW request bytes with HMAC-SHA256 and sends the hex
 * digest as `sha256=<hex>`. This proxy is the only component that still has
 * those bytes: the backend's express app parses JSON before any route runs,
 * and re-serializing a parsed body does not reproduce what was signed (key
 * order, whitespace and unicode escaping all differ).
 *
 * timingSafeEqual, never `==` -- GitHub's own documentation calls this out.
 * It throws on a length mismatch, so lengths are compared first.
 */
function verifiedAsGithub(body: Buffer, signature: string): boolean {
  if (!GITHUB_SECRET) return false;
  const expected = `sha256=${crypto.createHmac("sha256", GITHUB_SECRET).update(body).digest("hex")}`;
  const a = Buffer.from(expected, "utf8");
  const b = Buffer.from(signature, "utf8");
  if (a.length !== b.length) return false;
  return crypto.timingSafeEqual(a, b);
}

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

/**
 * Reads the whole body, up to MAX_WEBHOOK_BYTES. Resolves null if the cap is
 * exceeded, having stopped reading rather than continuing to accumulate.
 */
function readCappedBody(req: http.IncomingMessage): Promise<Buffer | null> {
  return new Promise((resolve) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let done = false;
    const finish = (value: Buffer | null) => {
      if (done) return;
      done = true;
      resolve(value);
    };
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_WEBHOOK_BYTES) {
        // Pause rather than destroy. Destroying the request here tears down
        // the socket before the 413 can be written, so the client sees a
        // connection reset and no status at all -- measured as curl
        // reporting HTTP 000. Stop consuming, let the caller answer, and
        // the response path closes the connection afterwards.
        req.pause();
        finish(null);
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => finish(Buffer.concat(chunks)));
    req.on("error", () => finish(null));
  });
}

const server = http.createServer((clientReq, clientRes) => {
  const headers = filteredHeaders(clientReq.headers);
  // c2c-mcp's DNS-rebinding guard checks the Host header's hostname against
  // localhost/127.0.0.1/[::1]. Forward the *target's* host here, not
  // whatever public host/IP the client actually connected to -- otherwise
  // every request gets rejected 403 by the backend, not proxied at all.
  headers.host = `${TARGET_HOST}:${TARGET_PORT}`;
  // Tell the backend this request came from the public internet, not a
  // same-machine caller -- that's what gates the OAuth check on /mcp. Every
  // request through this proxy gets it set to "1" regardless of what the
  // client sent, so a public caller can't spoof "I'm local" by omitting or
  // forging their own copy of this header.
  headers[PROXY_MARKER_HEADER] = "1";
  // Unconditional, and it must stay unconditional. This header is set below
  // only when a signature verifies; a client-supplied copy would otherwise
  // survive filteredHeaders and reach the backend looking proxy-issued.
  delete headers[VERIFIED_HEADER];

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

  // A request carrying a provider signature header is CLAIMING to be that
  // provider, so it gets buffered and verified before anything is forwarded;
  // a failed claim is hostile and is rejected here, at the edge, rather than
  // reaching the backend at all.
  //
  // Only signed requests are buffered. Everything else keeps piping
  // unmodified -- the unbuffered path is what makes SSE and MCP streaming
  // come out correct on the far leg (see the HOP_BY_HOP note above), and
  // buffering it wholesale would break both.
  const signature = clientReq.headers[GITHUB_SIG_HEADER];
  if (typeof signature === "string") {
    void readCappedBody(clientReq).then((body) => {
      if (!body) {
        proxyReq.destroy();
        if (!clientRes.headersSent) {
          // `connection: close` because the rest of the oversized body was
          // never read off the socket -- the connection cannot be reused.
          clientRes.writeHead(413, { "content-type": "text/plain", connection: "close" });
        }
        clientRes.end("Payload too large", () => clientReq.destroy());
        return;
      }
      if (!verifiedAsGithub(body, signature)) {
        proxyReq.destroy();
        // Deliberately identical for a bad signature and a missing secret:
        // a caller learns nothing about which it was.
        console.error(`[c2c-proxy] rejected unverified GitHub delivery for ${clientReq.url}`);
        if (!clientRes.headersSent) clientRes.writeHead(401, { "content-type": "text/plain" });
        clientRes.end("Unauthorized");
        return;
      }
      proxyReq.setHeader(VERIFIED_HEADER, "github");
      proxyReq.end(body);
    });
    return;
  }

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
  // Fail closed, and say so. With no secret, a signed delivery cannot be
  // verified and is rejected -- which is the safe direction, but silently
  // 401-ing every webhook is the kind of thing that gets debugged for an
  // hour, so the reason is stated at startup rather than only in the logs.
  console.log(
    GITHUB_SECRET
      ? "GitHub webhook verification ENABLED (C2C_GITHUB_WEBHOOK_SECRET set)"
      : "GitHub webhook verification DISABLED -- C2C_GITHUB_WEBHOOK_SECRET unset, so any signed delivery is rejected 401",
  );
});

for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.on(signal, () => {
    server.close(() => process.exit(0));
  });
}
