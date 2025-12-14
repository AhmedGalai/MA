#!/bin/bash
# Start Debug Viewer for Pose Estimation Pipeline
#
# This script provides a convenient way to launch the Debug Viewer
# with proper environment setup.
#
# Usage:
#   ./start_debug_viewer.sh [OPTIONS]
#
# Options:
#   --api-url URL          API URL (default: http://127.0.0.1:8000)
#   --width W              Window width in pixels (default: 1280)
#   --height H             Window height in pixels (default: 800)
#   --no-check             Skip API availability check
#   -h, --help             Show this help message

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"

# Default configuration
API_URL="http://127.0.0.1:8000"
WINDOW_WIDTH="1280"
WINDOW_HEIGHT="800"
CHECK_API=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-url)
            API_URL="$2"
            shift 2
            ;;
        --width)
            WINDOW_WIDTH="$2"
            shift 2
            ;;
        --height)
            WINDOW_HEIGHT="$2"
            shift 2
            ;;
        --no-check)
            CHECK_API=false
            shift
            ;;
        -h|--help)
            grep "^#" "$0" | grep -v "^#!/" | sed 's/# //g'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Function to check if API is available
check_api_availability() {
    local url="$1"
    local timeout=3

    echo "Checking API availability at $url..."

    if command -v curl &> /dev/null; then
        if curl -s --connect-timeout "$timeout" "$url/health" &>/dev/null; then
            echo "✓ API is available"
            return 0
        else
            echo "✗ API is not responding"
            return 1
        fi
    elif command -v wget &> /dev/null; then
        if wget --timeout="$timeout" -q -O /dev/null "$url/health" 2>/dev/null; then
            echo "✓ API is available"
            return 0
        else
            echo "✗ API is not responding"
            return 1
        fi
    else
        echo "⚠ Cannot check API (curl/wget not available)"
        return 0
    fi
}

# Check Python availability
if ! command -v "$PYTHON_EXECUTABLE" &> /dev/null; then
    echo "Error: Python executable not found: $PYTHON_EXECUTABLE"
    exit 1
fi

echo "=========================================="
echo "Debug Viewer for Pose Estimation Pipeline"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Python: $PYTHON_EXECUTABLE"
echo "  API URL: $API_URL"
echo "  Window Size: ${WINDOW_WIDTH}x${WINDOW_HEIGHT}"
echo ""

# Check API availability if requested
if [ "$CHECK_API" = true ]; then
    if ! check_api_availability "$API_URL"; then
        echo ""
        echo "Warning: API is not available at $API_URL"
        echo "Make sure main_api.py is running:"
        echo "  cd $SCRIPT_DIR"
        echo "  python3 main_api.py"
        echo ""
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

echo ""
echo "Launching Debug Viewer..."
echo ""

# Launch the debug viewer
cd "$SCRIPT_DIR"
"$PYTHON_EXECUTABLE" debug_viewer.py \
    --api-url "$API_URL" \
    --width "$WINDOW_WIDTH" \
    --height "$WINDOW_HEIGHT"
