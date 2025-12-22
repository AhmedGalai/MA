#!/bin/bash
set -e

echo "========================================"
echo "UxPlay + API Container"
echo "========================================"
echo ""

echo "Starting Avahi daemon for AirPlay discovery..."
mkdir -p /var/run/dbus /run/dbus
rm -f /run/dbus/pid
dbus-daemon --system --fork
avahi-daemon --daemonize --no-drop-root
sleep 2

if ! pgrep -x "avahi-daemon" > /dev/null; then
  echo "WARNING: Avahi daemon failed to start"
  echo "AirPlay device may not be discoverable on the network"
fi

echo ""
echo "Configuration:"
echo "  Device Name: ${UXPLAY_DEVICE_NAME:-AirPlay-Pipeline}"
echo "  API: ${API_HOST:-0.0.0.0}:${API_PORT:-8000}"
echo "  Resolution: ${UXPLAY_WIDTH:-1920}x${UXPLAY_HEIGHT:-1080}"
echo "  ArUco Dict: ${ARUCO_DICT:-DICT_4X4_250}"
echo "  ArUco Grid: ${ARUCO_ROWS:-3}x${ARUCO_COLS:-4}"
echo "  Marker Size: ${ARUCO_MARKER_SIZE_M:-0.03} m"
echo "  Separation: ${ARUCO_SEPARATION_M:-0.01} m"
echo "  Process FPS: ${PROCESS_FPS:-15.0}"
echo "  RS: ${RS_WIDTH:-640}x${RS_HEIGHT:-480} @ ${RS_FPS:-30}"
echo ""

ARGS=(
  --host "${API_HOST:-0.0.0.0}"
  --port "${API_PORT:-8000}"
  --device-name "${UXPLAY_DEVICE_NAME:-AirPlay-Pipeline}"
  --width "${UXPLAY_WIDTH:-1920}"
  --height "${UXPLAY_HEIGHT:-1080}"
  --fps "${PROCESS_FPS:-15.0}"
  --aruco-dict "${ARUCO_DICT:-DICT_4X4_250}"
  --aruco-rows "${ARUCO_ROWS:-3}"
  --aruco-cols "${ARUCO_COLS:-4}"
  --marker-size-m "${ARUCO_MARKER_SIZE_M:-0.03}"
  --separation-m "${ARUCO_SEPARATION_M:-0.01}"
  --rs-width "${RS_WIDTH:-640}"
  --rs-height "${RS_HEIGHT:-480}"
  --rs-fps "${RS_FPS:-30}"
)

if [[ -n "${UXPLAY_BINARY}" ]]; then
  ARGS+=(--uxplay-binary "${UXPLAY_BINARY}")
fi

exec python3 /app/final_pipeline.py "${ARGS[@]}"
#exec python3 /app/basic_main_api_with_uxplay_rs.py "${ARGS[@]}"
#exec python3 /app/avp_foundationpose_pipeline.py "${ARGS[@]}"
#exec python3 /app/rs_foundationpose_pipeline.py "${ARGS[@]}"
