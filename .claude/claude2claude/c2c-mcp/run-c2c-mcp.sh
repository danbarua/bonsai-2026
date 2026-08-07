#!/bin/bash

export C2C_MCP_PUBLIC_URL=https://c2c.framesift.ai/mcp

# 1. Catch Ctrl+C (SIGINT) and exit (SIGTERM) signals to clean up
trap 'echo -e "\n🛑 Stopping services..."; kill $DEV_PID 2>/dev/null; exit 0' SIGINT SIGTERM

echo "🚀 Starting local server..."
# 2. Run npm dev in the background and save its process ID (PID)
mkdir -p ./logs
npm run dev >./logs/stdout.log 2>./logs/err.log &
DEV_PID=$!

# 3. Give the local server 2 seconds to initialize
sleep 2

echo "🔗 Opening SSH Tunnel (auto-reconnects on drop, Ctrl+C to stop)..."
# 4. Run SSH in the foreground (blocks the script here), retrying if it
#    drops. ServerAliveInterval/CountMax send keepalive probes every 30s so
#    an idle NAT/firewall timeout doesn't silently kill the tunnel -- this
#    was happening every few minutes without them, taking the whole script
#    down with it (SSH exiting fell through to the old unconditional
#    "kill $DEV_PID" below). The dev server now stays up across reconnects;
#    only Ctrl+C (the trap above) kills it.
while true; do
  ssh -R 8765:127.0.0.1:8765 framzapo@framesift.ai -N -p21098 \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes
  echo "⚠️  SSH tunnel dropped, reconnecting in 3s..."
  sleep 3
done
