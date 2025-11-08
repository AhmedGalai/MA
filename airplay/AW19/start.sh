#!/bin/bash
# AW19 System Startup Script (Linux/macOS)
# Uses default Python from PATH

PYTHON="python"

echo "========================================"
echo "AW19 - AVP Vision Processing System"
echo "========================================"
echo ""
echo "Python: $(which $PYTHON)"
echo "Version: $($PYTHON --version)"
echo ""

# Check if Python exists
if ! command -v $PYTHON &> /dev/null; then
    echo "ERROR: Python not found in PATH"
    echo "Please install Python or update the PYTHON variable in this script."
    echo ""
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "main_api.py" ]; then
    echo "ERROR: Please run this script from the AW19 directory"
    echo "Usage: cd AW19 && ./start.sh"
    echo ""
    exit 1
fi

echo "Starting system components..."
echo ""

# Function to start a component in background
start_component() {
    local name=$1
    local script=$2
    local logfile=$3

    echo "Starting $name..."
    nohup $PYTHON $script > $logfile 2>&1 &
    local pid=$!
    echo "  PID: $pid"
    echo "  Log: $logfile"
    sleep 1
}

# Create logs directory
mkdir -p logs

echo "[1/4] Starting Main API (port 5000)..."
start_component "Main API" "main_api.py" "logs/main_api.log"
sleep 3

echo "[2/4] Selecting default model (cube.ply) when ready..."
$PYTHON select_default_model.py --url http://localhost:5000 --model cube.ply --timeout 180 || true

echo "[3/4] Starting Screen Capture UI..."
start_component "Screen Capture" "screen_capture.py" "logs/screen_capture.log"
sleep 1

echo "[4/4] Starting Debug Viewer..."
start_component "Debug Viewer" "tk_debugging_client.py" "logs/debug_viewer.log"

echo ""
echo "========================================"
echo "All components started successfully!"
echo "========================================"
echo ""
echo "Services:"
echo "  - Main API:        http://localhost:5000"
echo "  - Screen Capture:  UI window"
echo "  - Debug Viewer:    UI window"
echo ""
echo "Logs are in ./logs/ directory"
echo ""
echo "To view logs in real-time:"
echo "  tail -f logs/main_api.log"
echo ""
echo "To stop all processes:"
echo "  pkill -f 'python.*main_api.py'"
echo "  pkill -f 'python.*screen_capture.py'"
echo "  pkill -f 'python.*tk_debugging_client.py'"
echo ""
echo "Or use: ./stop.sh"
echo ""
