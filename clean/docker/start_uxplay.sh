#!/bin/bash
# Start script for UxPlay and frame capture service

echo "========================================="
echo "UxPlay Container Starting"
echo "========================================="

# Start UxPlay in background with video output
# -n: Server name visible on network
# -vs: Video sink (use autovideosink for automatic selection, or fakesink to just advertise)
# -fps 30: Target frame rate
echo "Starting UxPlay server..."
echo "Note: Using fakesink to focus on network discovery first"

# Use stdbuf to disable output buffering so we see logs immediately
stdbuf -oL -eL uxplay -n "PoseAPI" -vs fakesink -fps 30 -v 2>&1 &

UXPLAY_PID=$!
echo "UxPlay started with PID: $UXPLAY_PID"

# Wait for UxPlay to initialize and show output
sleep 5
echo ""
echo "UxPlay should now be advertising on the network..."
echo "If you cannot see it, mDNS/Bonjour may not work from Docker on Windows"

# Start frame capture and forwarding service
echo "Starting frame capture service..."
python3 /app/capture_and_send.py &

CAPTURE_PID=$!
echo "Capture service started with PID: $CAPTURE_PID"

# Monitor both processes
echo ""
echo "========================================="
echo "Services running:"
echo "  UxPlay PID: $UXPLAY_PID"
echo "  Capture PID: $CAPTURE_PID"
echo "========================================="
echo ""
echo "Connect your visionOS device to 'PoseAPI'"
echo "Press Ctrl+C to stop"
echo ""

# Wait for either process to exit
wait -n

# If we get here, one process died
echo "A service has stopped, shutting down container..."
kill $UXPLAY_PID $CAPTURE_PID 2>/dev/null
exit 1
