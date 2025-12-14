# Debug Viewer - Pose Estimation Pipeline Debugging Tool

## Overview

The **Debug Viewer** is a tkinter-based visual debugging tool for monitoring the pose estimation pipeline in real-time. It provides a comprehensive interface to inspect system status, camera feeds, pose transformations, and performance statistics.

## Features

### Display Panels (2x3 Grid Layout)

#### Image Panels (Top Row)
1. **RealSense RGB** - Live RGB feed from the RealSense camera (320x240)
2. **Transformed Mask (RS view)** - Binary mask transformed from AVP to RealSense coordinate frame
3. **RealSense Depth** - Depth colormap visualization from RealSense (320x240)

#### Text Panels (Bottom Row)
1. **System Status** - Shows:
   - RealSense connection status
   - System calibration state
   - Last update timestamp
   - Current update rate (Hz)

2. **Latest Poses** - Displays:
   - RS Camera Pose in AVP coordinate frame (4x4 matrix)
   - Object Pose in AVP coordinate frame (4x4 matrix)

3. **Statistics** - Tracks:
   - Total update count
   - Successful and failed estimates
   - Success rate percentage
   - Average frame time and estimated Hz
   - Last 5 frame timings

### Controls

- **Connect/Disconnect** - Toggle API connection
- **Manual Refresh** - Force immediate update from API
- **Clear Stats** - Reset all statistics counters
- **Refresh Rate Slider** - Adjust polling rate (1-10 Hz)

### Status Indicator

Connection status shown in title bar:
- **Green dot** - Connected and running
- **Red dot** - Not connected

## Installation

### Prerequisites

The Debug Viewer requires the following Python packages:

```bash
pip install tkinter pillow requests numpy opencv-python
```

On Ubuntu/Debian systems, tkinter may need to be installed separately:

```bash
sudo apt-get install python3-tk python3-dev
```

### Dependencies

From the Abgabe_2 system:
- `config.py` - Configuration module (auto-detected)
- `main_api.py` - REST API server (must be running)

## Usage

### Basic Usage

Run the debug viewer from the Kubuntu directory:

```bash
cd /home/ag/Desktop/MA/Abgabe_2/src/Kubuntu
python3 debug_viewer.py
```

### Command-Line Arguments

```bash
python3 debug_viewer.py [OPTIONS]

Options:
  --api-url API_URL        API URL (default: http://127.0.0.1:8000)
  --width WIDTH            Window width in pixels (default: 1280)
  --height HEIGHT          Window height in pixels (default: 800)
  -h, --help               Show help message
```

### Example Commands

**Connect to local API (default):**
```bash
python3 debug_viewer.py
```

**Connect to specific API URL:**
```bash
python3 debug_viewer.py --api-url http://192.168.1.100:8000
```

**Custom window size:**
```bash
python3 debug_viewer.py --width 1600 --height 900
```

## Workflow

### Starting a Debugging Session

1. **Start the main API server** (if not already running):
   ```bash
   python3 main_api.py
   ```

2. **Launch the Debug Viewer**:
   ```bash
   python3 debug_viewer.py
   ```

3. **Click Connect** to establish connection to API

4. **Verify system status**:
   - Check "RealSense Connected" in Status panel
   - Check "System Calibrated" in Status panel

### Monitoring Pipeline Execution

1. **Watch Status Panel** for real-time system state
2. **Monitor Statistics** for performance metrics:
   - Success rate indicates estimation reliability
   - Average frame time shows processing speed
   - Update rate shows polling responsiveness

3. **Adjust Refresh Rate** if needed:
   - Higher rates (8-10 Hz) for fast feedback, higher CPU load
   - Lower rates (1-2 Hz) for resource efficiency

### Troubleshooting

**Connection Failed**
- Ensure `main_api.py` is running on the expected host/port
- Check firewall rules if connecting remotely
- Verify API URL matches your deployment

**No RealSense Connection**
- Check if RealSense camera is physically connected
- Verify `pyrealsense2` library is properly installed
- Check API logs for initialization errors

**No Calibration**
- Perform RS calibration via API calibration endpoint
- Ensure ArUco board is visible and properly positioned
- Check calibration file in `extrinsics/` directory

**Slow Updates**
- Reduce refresh rate slider (1-2 Hz is typical)
- Check API server CPU/memory usage
- Verify network latency if using remote API

## API Endpoints Used

The Debug Viewer polls the following endpoints:

### GET /health
System health check endpoint that returns:
```json
{
    "status": "ok",
    "rs_connected": true,
    "calibrated": true
}
```

**Polling Rate:** Configurable via refresh rate slider (1-10 Hz)
**Timeout:** 3 seconds per request

## Performance Considerations

### CPU Usage
- Typical CPU load: 5-15% at 2 Hz refresh rate
- Increases with higher refresh rates
- Image resizing and display overhead is minimal (320x240 panels)

### Memory Usage
- Typical memory footprint: 50-100 MB
- Maintains circular buffer of last 100 frame timings
- No memory leaks expected with extended operation

### Network Requirements
- Minimal bandwidth usage (JSON status only, no images streamed)
- Works well with 100+ Mbps connections
- Can operate over WiFi without issues

## Architecture

### Main Components

**DebugViewer Class**
- Manages tkinter GUI and layout
- Handles API connections and polling
- Displays real-time data updates
- Tracks statistics and performance metrics

**Polling Thread**
- Runs in background at configurable rate
- Non-blocking UI updates
- Thread-safe data access via tkinter's main loop

**Image Display**
- Uses PIL/Pillow for image conversion
- Caches PhotoImage references to prevent garbage collection
- Supports grayscale, RGB, BGR, and RGBA formats
- Auto-resizes to 320x240 panel dimensions

### Data Flow

```
API Server (main_api.py)
    ↓ GET /health (JSON)
    ↓
Polling Thread
    ↓
DebugViewer (Main Thread)
    ↓
GUI Panels (Display)
```

## Extending the Debug Viewer

### Adding Custom Endpoints

Modify the `fetch_status()` method to poll additional endpoints:

```python
def fetch_additional_data(self):
    """Fetch additional data from custom endpoints."""
    try:
        response = requests.get(
            f"{self.api_url}/custom_endpoint",
            timeout=3
        )
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching custom data: {e}")
        return None
```

### Adding New Display Panels

Create new panels using existing helper methods:

```python
# In _setup_gui():
self.panel_custom = self._create_text_panel(
    content_frame, "Custom Data", 1, 0
)

# Update it:
self._update_text_widget(
    self.panel_custom['text'],
    "Custom content here"
)
```

## Logging

Debug Viewer logs are output to console with timestamps. Log levels:

- **INFO** - Normal operation messages
- **WARNING** - Non-fatal issues (timeouts, decode errors)
- **ERROR** - Serious issues requiring attention

### Example Log Output
```
2025-12-14 14:32:15,123 - __main__ - INFO - DebugViewer initialized with API URL: http://127.0.0.1:8000
2025-12-14 14:32:18,456 - __main__ - INFO - Connected to API at http://127.0.0.1:8000
2025-12-14 14:32:19,789 - __main__ - ERROR - Error fetching status: Connection timeout
```

## License

Part of the MA Project pose estimation pipeline. For internal use.

## Support

For issues or feature requests, refer to the main project documentation or contact the development team.

---

**Last Updated:** 2025-12-14
**Version:** 1.0
**Compatibility:** Python 3.7+, Linux/WSL2
