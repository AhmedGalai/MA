#!/bin/bash
# AW19 System Stop Script (Linux/macOS)

echo "========================================"
echo "Stopping AW19 System"
echo "========================================"
echo ""

echo "Stopping all AW19 processes..."

# Kill all Python processes running AW19 scripts
pkill -f "python.*main_api.py" && echo "  ✓ Stopped Main API"
pkill -f "python.*screen_capture.py" && echo "  ✓ Stopped Screen Capture"
pkill -f "python.*tk_debugging_client.py" && echo "  ✓ Stopped Debug Viewer"

sleep 1

echo ""
echo "========================================"
echo "All components stopped"
echo "========================================"
echo ""
