#!/bin/bash
set -e

echo "========================================="
echo "UxPlay Container Starting"
echo "========================================="

# If you are NOT mounting host dbus/avahi, you'd need to start them here.
# But with the compose mounts above, you should NOT start container daemons.

echo "Starting UxPlay server..."
export GST_DEBUG=2   # helps catch gstreamer errors in logs

# Use -nh (don’t append @hostname), and set explicit base port.
# Force headless sinks so it won’t try to open a window.
uxplay -n "PoseAPI" -nh -p 7000 -fps 30 -vs fakesink -as fakesink 2>&1 | tee /tmp/uxplay.log &
UXPLAY_PID=$!

sleep 2
if ! kill -0 $UXPLAY_PID 2>/dev/null; then
  echo "ERROR: UxPlay exited immediately!"
  tail -200 /tmp/uxplay.log || true
  exit 1
fi

echo "Starting frame capture service..."
python3 /app/capture_and_send.py &
CAPTURE_PID=$!

# keep container alive
wait $UXPLAY_PID
