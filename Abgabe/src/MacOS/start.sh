#!/bin/bash

################################################################################
# Start Script for AR Pose Estimation System (macOS)
#
# This script launches the Python backend API server for the Vision Pro
# pose estimation system.
#
# Usage:
#   ./start.sh                    # Start main API only
#   ./start.sh --with-capture     # Start API + screen capture
#   ./start.sh --with-debug       # Start API + debug viewer
#   ./start.sh --full             # Start all components
#   ./start.sh --help             # Show help
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
VENV_DIR="$BACKEND_DIR/venv"

# Log file
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  AR Pose Estimation System - Backend Startup${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        print_error "Python not found. Please install Python 3.8+"
        exit 1
    fi

    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
    print_info "Using Python $PYTHON_VERSION at $(which $PYTHON_CMD)"
}

check_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        print_warn "Virtual environment not found at $VENV_DIR"
        print_info "Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        print_info "Virtual environment created"
    fi
}

activate_venv() {
    print_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
}

install_dependencies() {
    if [ ! -f "$BACKEND_DIR/requirements.txt" ]; then
        print_warn "requirements.txt not found, creating basic one..."
        cat > "$BACKEND_DIR/requirements.txt" <<EOF
numpy>=1.24.0
opencv-python>=4.8.0
torch>=2.0.0
transformers>=4.30.0
pillow>=10.0.0
flask>=2.3.0
flask-cors>=4.0.0
requests>=2.31.0
# Optional: Uncomment if using RealSense
# pyrealsense2>=2.54.0
EOF
    fi

    print_info "Installing/updating dependencies..."
    pip install --upgrade pip -q
    pip install -r "$BACKEND_DIR/requirements.txt" -q
    print_info "Dependencies installed"
}

check_models() {
    if [ ! -d "$SCRIPT_DIR/models" ]; then
        print_error "Models directory not found at $SCRIPT_DIR/models"
        print_info "Please ensure .ply model files are present"
        exit 1
    fi

    MODEL_COUNT=$(find "$SCRIPT_DIR/models" -name "*.ply" | wc -l)
    print_info "Found $MODEL_COUNT .ply model(s) in models directory"
}

check_port() {
    PORT=$1
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        print_warn "Port $PORT is already in use"
        return 1
    fi
    return 0
}

start_main_api() {
    print_info "Starting Main API Server..."

    if ! check_port 5000; then
        print_error "Port 5000 is already in use. Please stop the existing service."
        exit 1
    fi

    cd "$BACKEND_DIR"

    # Set PYTHONPATH to include backend directory for imports
    export PYTHONPATH="$BACKEND_DIR:$PYTHONPATH"

    print_info "Main API will be available at: http://0.0.0.0:5000"
    print_info "Logs will be written to: $LOG_DIR/main_api.log"
    echo ""

    $PYTHON_CMD main_api.py 2>&1 | tee "$LOG_DIR/main_api.log"
}

start_screen_capture() {
    print_info "Starting Screen Capture..."

    cd "$BACKEND_DIR"
    export PYTHONPATH="$BACKEND_DIR:$PYTHONPATH"

    print_info "Screen capture will send frames to http://localhost:5000/receive_frame"
    print_info "Logs: $LOG_DIR/screen_capture.log"

    $PYTHON_CMD screen_capture.py > "$LOG_DIR/screen_capture.log" 2>&1 &
    CAPTURE_PID=$!
    echo $CAPTURE_PID > "$LOG_DIR/screen_capture.pid"
    print_info "Screen capture started (PID: $CAPTURE_PID)"
}

start_debug_viewer() {
    print_info "Starting Debug Viewer..."

    cd "$BACKEND_DIR/full_python_pipeline"
    export PYTHONPATH="$BACKEND_DIR:$PYTHONPATH"

    print_info "Debug viewer will connect to http://localhost:5000"
    print_info "Logs: $LOG_DIR/debug_viewer.log"

    $PYTHON_CMD tk_debugging_unified.py > "$LOG_DIR/debug_viewer.log" 2>&1 &
    DEBUG_PID=$!
    echo $DEBUG_PID > "$LOG_DIR/debug_viewer.pid"
    print_info "Debug viewer started (PID: $DEBUG_PID)"
}

stop_background_processes() {
    print_info "Stopping background processes..."

    if [ -f "$LOG_DIR/screen_capture.pid" ]; then
        CAPTURE_PID=$(cat "$LOG_DIR/screen_capture.pid")
        if kill -0 $CAPTURE_PID 2>/dev/null; then
            kill $CAPTURE_PID
            print_info "Stopped screen capture (PID: $CAPTURE_PID)"
        fi
        rm -f "$LOG_DIR/screen_capture.pid"
    fi

    if [ -f "$LOG_DIR/debug_viewer.pid" ]; then
        DEBUG_PID=$(cat "$LOG_DIR/debug_viewer.pid")
        if kill -0 $DEBUG_PID 2>/dev/null; then
            kill $DEBUG_PID
            print_info "Stopped debug viewer (PID: $DEBUG_PID)"
        fi
        rm -f "$LOG_DIR/debug_viewer.pid"
    fi
}

show_help() {
    cat <<EOF
Usage: $0 [OPTIONS]

Start the AR Pose Estimation System backend components

OPTIONS:
    (none)              Start main API server only (default)
    --with-capture      Start API + screen capture
    --with-debug        Start API + debug viewer
    --full              Start all components (API + capture + debug)
    --stop              Stop all background processes
    --setup             Setup virtual environment and install dependencies
    --help              Show this help message

EXAMPLES:
    $0                          # Start API only
    $0 --with-capture           # Start API and screen capture
    $0 --full                   # Start all components
    $0 --stop                   # Stop background processes

CONFIGURATION:
    Edit backend/app_config.py to change:
    - API ports and URLs
    - Depth mode (RealSense vs Transformers)
    - Model selection
    - ROI parameters

VISION PRO SETUP:
    1. Start backend: ./start.sh
    2. Note the IP address displayed
    3. In Vision Pro app, enter: http://<IP>:5000
    4. Select model and open immersive space

For more information, see docs/PROJECT_STRUCTURE.md
EOF
}

################################################################################
# Main Logic
################################################################################

# Trap exit to cleanup
trap stop_background_processes EXIT INT TERM

# Parse arguments
WITH_CAPTURE=false
WITH_DEBUG=false
SETUP_ONLY=false
STOP_ONLY=false

case "${1:-}" in
    --help|-h)
        show_help
        exit 0
        ;;
    --setup)
        SETUP_ONLY=true
        ;;
    --stop)
        STOP_ONLY=true
        ;;
    --with-capture)
        WITH_CAPTURE=true
        ;;
    --with-debug)
        WITH_DEBUG=true
        ;;
    --full)
        WITH_CAPTURE=true
        WITH_DEBUG=true
        ;;
    "")
        # Default: API only
        ;;
    *)
        print_error "Unknown option: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

# Handle stop command
if [ "$STOP_ONLY" = true ]; then
    stop_background_processes
    exit 0
fi

# Print header
print_header
echo ""

# Environment checks
check_python
check_venv
activate_venv

# Setup mode
if [ "$SETUP_ONLY" = true ]; then
    install_dependencies
    check_models
    print_info "Setup complete!"
    exit 0
fi

# Install dependencies if needed
install_dependencies
check_models

echo ""
print_info "Backend directory: $BACKEND_DIR"
print_info "Models directory: $SCRIPT_DIR/models"
print_info "Logs directory: $LOG_DIR"
echo ""

# Get local IP for Vision Pro connection
LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
if [ -n "$LOCAL_IP" ]; then
    print_info "Vision Pro should connect to: http://${LOCAL_IP}:5000"
    echo ""
fi

# Start optional components in background
if [ "$WITH_CAPTURE" = true ]; then
    start_screen_capture
    sleep 2
fi

if [ "$WITH_DEBUG" = true ]; then
    start_debug_viewer
    sleep 2
fi

# Start main API (blocking)
print_info "Starting main components..."
echo ""
sleep 1

start_main_api
