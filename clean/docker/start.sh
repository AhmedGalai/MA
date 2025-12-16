#!/bin/bash
set -e

echo "========================================"
echo "UxPlay AirPlay Receiver Container"
echo "========================================"
echo ""

# Start avahi-daemon for mDNS/AirPlay discovery
echo "Starting Avahi daemon for AirPlay discovery..."
avahi-daemon --daemonize --no-drop-root

# Wait for avahi to start
sleep 2

# Check if avahi is running
if ! pgrep -x "avahi-daemon" > /dev/null; then
    echo "WARNING: Avahi daemon failed to start"
    echo "AirPlay device may not be discoverable on the network"
fi

echo "Avahi daemon started"
echo ""

# Display configuration
echo "Configuration:"
echo "  Device Name: ${UXPLAY_DEVICE_NAME:-AirPlay-Pipeline}"
echo "  Main API: ${MAIN_API_HOST}:${MAIN_API_PORT}"
echo "  Resolution: ${UXPLAY_WIDTH:-1920}x${UXPLAY_HEIGHT:-1080}"
echo ""

# Start UxPlay module
echo "Starting UxPlay AirPlay receiver..."
echo "========================================"
echo ""

exec python3 /app/uxplay_module.py \
    --device-name "${UXPLAY_DEVICE_NAME:-AirPlay-Pipeline}" \
    --width "${UXPLAY_WIDTH:-1920}" \
    --height "${UXPLAY_HEIGHT:-1080}" \
    --api-url "http://${MAIN_API_HOST}:${MAIN_API_PORT}"
