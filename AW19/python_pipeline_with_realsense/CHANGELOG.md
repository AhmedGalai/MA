# AW19 Changelog

## Latest Updates

### Bug Fixes (2025-11-08)

#### Fixed tk_debugging_client.py Errors (renamed from tk_hypercam_2.py)
- **Issue**: AttributeError for `left_var`, `top_var`, `width_var`, etc.
- **Cause**: Debug viewer still referenced screen capture configuration variables
- **Fix**: Removed all screen capture variable references from:
  - `_update_from_config()` - Now only updates HSV settings
  - `_apply_settings()` - Only sends HSV and tolerance config to API
  - Made config format compatible with new API structure

#### Added API Connection Retries
- **Issue**: Viewer would fail if API not ready
- **Solution**: Implemented retry logic with:
  - 10 retry attempts with 1-second delays
  - Visual feedback showing "Connecting to API... (attempt X/10)"
  - Helpful error message if all retries fail
  - Automatic retry every 5 seconds in background
  - User-friendly message explaining what to check

#### Improved Startup Scripts

**start.bat** (Windows)
- Uses specific Python environment: `C:\Users\Lenovo\.guv\envs\masterarbeit\Scripts\python.exe`
- Added note about depth model loading time (10-30 seconds)
- Checks if Python exists before starting
- Shows service URLs when complete

**start.sh** (Linux/macOS)
- Uses default `python` from PATH
- Creates `logs/` directory
- Logs all output to separate files
- Shows PIDs for each process
- Includes instructions for monitoring and stopping

**stop.sh** (Linux/macOS)
- Cleanly stops all AW19 processes
- Uses pkill for safe termination

### Configuration Format Changes

**Old Format** (deprecated):
```json
{
  "left": 934,
  "top": 100,
  "width": 812,
  "height": 1080,
  "fps": 30,
  "hsv_center": [90, 128, 128],
  "tolerances": {
    "h": 12,
    "s": 50,
    "v": 50
  }
}
```

**New Format** (current):
```json
{
  "hsv_center": [90, 128, 128],
  "h_tol": 12,
  "s_tol": 50,
  "v_tol": 50
}
```

The debug viewer now handles both formats for backward compatibility.

## Component Responsibilities

### screen_capture.py
- ✅ Handles ALL screen capture configuration
- ✅ Has its own UI with sliders
- ✅ Independent from debug viewer

### tk_debugging_client.py (Debug Viewer)
- ✅ Only displays pipeline results
- ✅ Only configures HSV mask parameters
- ✅ No longer controls screen capture
- ✅ Automatic retry on API connection failure

### main_api.py
- ✅ Receives frames from screen_capture
- ✅ Coordinates CV pipeline
- ✅ Manages models and pose requests
- ✅ Provides configuration endpoints

### computer_vision_pipeline.py
- ✅ GPU-optimized processing
- ✅ ArUco detection
- ✅ Pose estimation
- ✅ Mask extraction
- ✅ Depth estimation

## Usage After Fixes

### Windows
```bash
# Just run:
start.bat

# Or specific Python:
# Edit start.bat to change PYTHON variable
```

### Linux/macOS
```bash
# Make executable (first time only)
chmod +x start.sh stop.sh

# Start system
./start.sh

# Stop system
./stop.sh

# Monitor logs
tail -f logs/main_api.log
```

## Testing

After these fixes:

1. **Debug Viewer** should start without errors
2. **Connection retry** shows progress and helpful messages
3. **No AttributeError** for missing variables
4. **Clean startup** with all components

Test command:
```bash
python test_system.py
```

## Known Behavior

- Debug viewer will retry connection for ~10 seconds
- First API startup takes 10-30 seconds (depth model download)
- Subsequent startups are much faster (~3-5 seconds)
- If API not ready, viewer shows warning but keeps retrying

## Files Modified

1. `tk_debugging_client.py` - Fixed variable references, added retries
2. `start.bat` - Specific Python path, better messages
3. `start.sh` - New file with logging
4. `stop.sh` - New file for clean shutdown
5. `CHANGELOG.md` - This file

---

**Status**: All known bugs fixed ✓
**Version**: AW19.1
**Date**: 2025-11-08
