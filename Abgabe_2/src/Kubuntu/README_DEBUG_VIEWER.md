# Debug Viewer for Pose Estimation Pipeline

A professional-grade tkinter-based visual debugging tool for real-time monitoring of the pose estimation pipeline.

## Quick Start

```bash
# Install dependencies
pip install pillow requests numpy opencv-python

# Launch the viewer (with main_api.py running)
python3 debug_viewer.py

# Or use the startup script
./start_debug_viewer.sh
```

## Documentation Files

- **DEBUG_VIEWER_GUIDE.md** - Complete user guide (Installation, Features, Troubleshooting)
- **QUICKSTART.txt** - Quick reference card (Common commands, examples)
- **README_DEBUG_VIEWER.md** - This file (Overview and file index)

## Features

- Real-time 2x3 grid display with 6 panels
- Camera feeds (RGB, Mask, Depth) and system status
- Live statistics tracking and performance monitoring
- Configurable polling rate (1-10 Hz)
- Connect/Disconnect controls
- Color-coded status indicators

## File Structure

```
Kubuntu/
├── debug_viewer.py                 # Main application (718 lines)
├── DEBUG_VIEWER_GUIDE.md           # User documentation
├── QUICKSTART.txt                  # Quick reference
├── README_DEBUG_VIEWER.md          # This file
├── start_debug_viewer.sh           # Launch script
├── example_debug_session.py        # Example patterns
└── [other pipeline files]
```

## System Requirements

- Python 3.7+
- Linux/WSL2
- tkinter (built-in or via apt)
- X11 display for GUI

## Basic Usage

1. Ensure main_api.py is running
2. Launch debug_viewer.py
3. Click "Connect" button
4. Monitor real-time status and statistics

## Support

- See DEBUG_VIEWER_GUIDE.md for detailed documentation
- See QUICKSTART.txt for common tasks
- Review example_debug_session.py for integration patterns

## Status

- Created: 2025-12-14
- Version: 1.0
- Status: Production-Ready
- All requirements met and verified
