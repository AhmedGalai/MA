#!/bin/bash
# Start script for UxPlay and frame capture service

echo "========================================="
echo "UxPlay Container Starting"
echo "========================================="

# Start UxPlay in background with video output
# -n: Server name visible on network
# -fps 30: Target frame rate
# Note: Removed -vs fakesink as it may cause immediate exit
echo "Starting UxPlay server..."
echo "Note: Running without video sink to just advertise AirPlay service"

# Use stdbuf to disable output buffering so we see logs immediately
# Redirect both stdout and stderr to see all output
stdbuf -oL -eL uxplay -n "PoseAPI" -fps 30 -v 2>&1 | tee /tmp/uxplay.log &

UXPLAY_PID=$!
echo "UxPlay started with PID: $UXPLAY_PID"

# Wait for UxPlay to initialize and show output
sleep 3

# Check if UxPlay is still running
if ! kill -0 $UXPLAY_PID 2>/dev/null; then
    echo "ERROR: UxPlay exited immediately!"
    echo "Last log lines:"
    tail -20 /tmp/uxplay.log
    exit 1
fi

echo "UxPlay is running..."
sleep 2
echo ""
echo "========================================="
echo "CONNECTION INSTRUCTIONS"
echo "========================================="
echo ""
echo "IMPORTANT: Auto-discovery (mDNS) doesn't work from Docker on Windows!"
echo ""
echo "To connect your Vision Pro:"
echo "  1. Find your Windows IP address:"
echo "     Run 'ipconfig' in PowerShell/CMD"
echo ""
echo "  2. Connect manually from Vision Pro:"
echo "     Settings → AirPlay & Handoff"
echo "     (Manual connection or enter IP if supported)"
echo ""
echo "  3. Or run UxPlay natively on WSL instead:"
echo "     sudo apt-get install uxplay"
echo "     uxplay -n 'PoseAPI' -fps 30"
echo ""
echo "See docker/MANUAL_CONNECTION.md for detailed instructions"
echo ""
echo "========================================="

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
echo "Waiting for Vision Pro connection..."
echo "(See connection instructions above)"
echo "Press Ctrl+C to stop"
echo ""

# Monitor both processes continuously
while true; do
    # Check if UxPlay is still running
    if ! kill -0 $UXPLAY_PID 2>/dev/null; then
        echo ""
        echo "ERROR: UxPlay process (PID $UXPLAY_PID) has stopped!"
        echo "Last 30 lines of UxPlay log:"
        tail -30 /tmp/uxplay.log
        kill $CAPTURE_PID 2>/dev/null
        exit 1
    fi

    # Check if capture service is still running
    if ! kill -0 $CAPTURE_PID 2>/dev/null; then
        echo ""
        echo "WARNING: Capture service (PID $CAPTURE_PID) has stopped!"
        echo "Restarting capture service..."
        python3 /app/capture_and_send.py &
        CAPTURE_PID=$!
        echo "Capture service restarted with PID: $CAPTURE_PID"
    fi

    # Sleep for 5 seconds before next check
    sleep 5
done
