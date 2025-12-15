#!/bin/bash
# Start script for UxPlay and frame capture service

echo "========================================="
echo "UxPlay Container Starting"
echo "========================================="

# Start UxPlay in background with video output to stdout
# -n: Server name visible on network
# -vs: Video sink (use autovideosink or appsink for capture)
echo "Starting UxPlay server..."
uxplay -n "Kubuntu Backend" -vs "appsink" &

UXPLAY_PID=$!
echo "UxPlay started with PID: $UXPLAY_PID"

# Wait a moment for UxPlay to initialize
sleep 3

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
echo "Connect your visionOS device to 'Kubuntu Backend'"
echo "Press Ctrl+C to stop"
echo ""

# Wait for either process to exit
wait -n

# If we get here, one process died
echo "A service has stopped, shutting down container..."
kill $UXPLAY_PID $CAPTURE_PID 2>/dev/null
exit 1
