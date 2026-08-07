#!/bin/bash

export C2C_MCP_PUBLIC_URL=https://c2c.framesift.ai/mcp

# Local dev server port, and the port it's tunneled to on the VM. Both
# overridable via env var (not a cmdline flag, for consistency with every
# other port/path knob in this project -- C2C_MCP_PORT, C2C_PROXY_TARGET_PORT,
# C2C_MAIL_WATCH_DIRS are all env vars already; a flag would need its own
# parsing for no real benefit, since an env var already works fine inline:
# C2C_TUNNEL_REMOTE_PORT=8767 ./run-c2c-mcp.sh).
#
# The remote port has had to move twice now (8765 -> 8766 -> 8767): an
# abruptly-killed SSH client leaves the remote sshd holding the old port's
# binding open ("remote port forwarding failed for listen port <N>" on
# every reconnect attempt), with no lsof/ss/fuser/sudo available on that
# host to find and clear it -- bumping the port is the actual fix each time
# this recurs, not a one-off. Whenever this changes, proxy.cjs's own
# C2C_PROXY_TARGET_PORT (set when starting it on the VM, see README.md) must
# be updated to match -- the two sides don't discover each other.
C2C_MCP_PORT="${C2C_MCP_PORT:-8765}"
C2C_TUNNEL_REMOTE_PORT="${C2C_TUNNEL_REMOTE_PORT:-8767}"
export C2C_MCP_PORT

# 0. Refuse to start if something is already listening on C2C_MCP_PORT,
#    instead of racing a fresh server against a stale one that was never
#    actually killed -- exactly what happened once already: an old `tsx`
#    process outlived several restart attempts, kept answering /health with
#    stale (pre-this-session) output, and both processes writing to the
#    same logs/stdout.log without the old one being killed first produced a
#    file that was mostly NUL bytes (each truncating the other's writes out
#    from under it). Printing the actual PID(s) means the fix is a
#    copy-pasteable command, not a manual lsof/ps hunt.
EXISTING_PIDS="$(lsof -nP -iTCP:"$C2C_MCP_PORT" -sTCP:LISTEN -t 2>/dev/null | tr '\n' ' ' | sed 's/ *$//')"
if [ -n "$EXISTING_PIDS" ]; then
  echo "❌ Something is already listening on port $C2C_MCP_PORT (PID(s): $EXISTING_PIDS)." >&2
  echo "   Kill it first, then re-run this script:" >&2
  echo >&2
  echo "   kill $EXISTING_PIDS" >&2
  echo >&2
  exit 1
fi

# 1. Catch Ctrl+C (SIGINT) and exit (SIGTERM) signals to clean up
trap 'echo -e "\n🛑 Stopping services..."; kill $DEV_PID 2>/dev/null; exit 0' SIGINT SIGTERM

# 2. Build once and run the compiled output (not tsx) -- this is the
#    public-facing deployment, so a standard build+run is preferable to
#    per-request TS transpilation. `npm run dist/index.js` is not a real
#    npm script (there's no script by that name in package.json); the
#    correct invocation is `node dist/index.js` directly, matching
#    package.json's own "start" script.
echo "🔨 Building..."
npm run build || { echo "❌ Build failed -- check the tsc output above."; exit 1; }

echo "🚀 Starting local server on port $C2C_MCP_PORT..."
mkdir -p ./logs
node dist/index.js >./logs/stdout.log 2>./logs/err.log &
DEV_PID=$!

# 3. Poll /health instead of a blind sleep -- a fixed sleep either wastes
#    time or (worse) races ahead of a slow start, opening the tunnel before
#    anything is listening behind it.
echo "⏳ Waiting for local server to come up..."
HEALTH=""
for _ in $(seq 1 50); do
  HEALTH="$(curl -s "http://127.0.0.1:$C2C_MCP_PORT/health")"
  if echo "$HEALTH" | grep -q '"ok":true'; then
    echo "✅ Local server up: $HEALTH"
    break
  fi
  sleep 0.2
done
if ! echo "$HEALTH" | grep -q '"ok":true'; then
  echo "❌ Local server did not come up in time -- check ./logs/err.log"
  kill "$DEV_PID" 2>/dev/null
  exit 1
fi

echo "🔗 Opening SSH Tunnel (remote $C2C_TUNNEL_REMOTE_PORT -> local $C2C_MCP_PORT, auto-reconnects on drop, Ctrl+C to stop)..."
# 4. Run SSH in the foreground (blocks the script here), retrying if it
#    drops. ServerAliveInterval/CountMax send keepalive probes every 30s so
#    an idle NAT/firewall timeout doesn't silently kill the tunnel -- this
#    was happening every few minutes without them, taking the whole script
#    down with it (SSH exiting fell through to the old unconditional
#    "kill $DEV_PID" below). The dev server now stays up across reconnects;
#    only Ctrl+C (the trap above) kills it.
while true; do
  ssh -R "$C2C_TUNNEL_REMOTE_PORT:127.0.0.1:$C2C_MCP_PORT" framzapo@framesift.ai -N -p21098 \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes
  echo "⚠️  SSH tunnel dropped, reconnecting in 3s..."
  sleep 3
done
