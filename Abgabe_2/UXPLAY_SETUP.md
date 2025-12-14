# UxPlay Integration Guide

## Overview

The UxPlay integration provides AirPlay mirroring capability to receive RGB frames from Apple Vision Pro. This implementation uses a Docker container running UxPlay with GStreamer output to continuously write frames to a shared directory, which can then be captured on-demand by the backend.

## Architecture

```
VisionOS Device → AirPlay → UxPlay (Docker) → Shared Directory (/frames/latest.jpg)
                                                    ↓
                                    User clicks "Capture" button
                                                    ↓
                                    uxplay_capture.py reads frame
                                                    ↓
                                    POST /receive_frame → main_api.py
```

**Key Design**: On-demand frame capture (not continuous streaming)

## Files Created

### Docker Configuration
1. **docker/uxplay/Dockerfile** - UxPlay container image
   - Based on Ubuntu 22.04
   - Installs GStreamer and UxPlay from source
   - Exposes ports 7000, 7001, 7100 (AirPlay)

2. **docker/uxplay/start-uxplay.sh** - Container startup script
   - Configures GStreamer pipeline
   - Outputs to `/frames/latest.jpg`
   - Uses `multifilesink` with `max-files=1` (overwrites same file)

3. **docker-compose.yml** - Docker Compose configuration
   - Host network mode (required for mDNS/Bonjour discovery)
   - Volume mount: `./frames:/frames:rw`
   - Auto-restart policy

### Python Services
4. **uxplay_capture.py** - Frame capture service
   - `UxPlayCapture` class for frame operations
   - Checks frame age (< 2 seconds)
   - Base64 encodes and sends to API
   - Command-line interface: `--action capture/monitor`

### Convenience Scripts
5. **start_uxplay.sh** - Start UxPlay container
   - Creates frames directory
   - Builds Docker image if needed
   - Starts container with docker-compose
   - Displays usage instructions

6. **stop_uxplay.sh** - Stop UxPlay container
   - Gracefully stops and removes container
   - Preserves frames directory

## Modified Files

### Backend API (main_api.py)
Added two new endpoints:

1. **POST /capture_frame?purpose=<purpose>**
   - Triggers frame capture from UxPlay
   - Calls `uxplay_capture.py` internally
   - Returns success/failure status

2. **POST /receive_frame**
   - Receives base64-encoded frame from uxplay_capture.py
   - Stores frame globally with timestamp
   - Accepts purpose parameter for logging

### Debug Viewer (debug_viewer.py)
Added UxPlay Frame Capture section with:
- "Capture for ArUco Calibration" button
- "Capture for ROI Selection" button
- Status indicator showing capture result
- Background threading to avoid GUI blocking

### Configuration (config.py)
Added UxPlay configuration section:
```python
"uxplay": {
    "enabled": True,
    "frame_dir": "./frames",
    "docker_compose_file": "./docker-compose.yml",
    "device_name": "Kubuntu Backend",
    "max_frame_age": 2.0
}
```

### Documentation (README.md)
- Updated directory structure
- Added UxPlay setup section
- Added frame capture endpoints documentation
- Updated software requirements (added Docker)

## Usage

### 1. Start UxPlay
```bash
cd /home/ag/Desktop/MA/Abgabe_2/src/Kubuntu
./start_uxplay.sh
```

Expected output:
```
=========================================
Starting UxPlay Docker Container
=========================================
...
UxPlay is now running!
=========================================
```

### 2. Connect VisionOS Device
On your Apple Vision Pro or iOS device:
1. Open Control Center
2. Tap "Screen Mirroring"
3. Select "Kubuntu Backend"
4. Device screen will mirror to UxPlay

### 3. Verify Frame Updates
```bash
# Monitor frame file updates
watch -n 1 ls -lh frames/latest.jpg

# View frames updating
python uxplay_capture.py --action monitor
```

### 4. Capture Frames

**Option A: Debug Viewer**
```bash
python debug_viewer.py
# Click "Capture for ArUco Calibration" or "Capture for ROI Selection"
```

**Option B: Command Line**
```bash
python uxplay_capture.py --action capture --purpose aruco_calibration
```

**Option C: API Call**
```bash
curl -X POST http://localhost:8000/capture_frame?purpose=roi_selection
```

### 5. Stop UxPlay
```bash
./stop_uxplay.sh
```

## Troubleshooting

### Container won't start
```bash
# Check Docker is running
docker info

# Check docker-compose version
docker-compose --version

# View logs
docker-compose logs -f uxplay
```

### Device can't find "Kubuntu Backend"
- Ensure both devices on same network
- Check firewall allows ports 7000, 7001, 7100
- Verify container is running: `docker-compose ps`
- Check mDNS/Bonjour is working: `avahi-browse -a` (Linux)

### Frames not updating
```bash
# Check if container is running
docker-compose ps uxplay

# Check logs for errors
docker-compose logs uxplay

# Verify AirPlay connection
docker exec -it uxplay-airplay ps aux | grep uxplay
```

### Frame capture fails
- Check frame age: `stat frames/latest.jpg`
- Verify API is running: `curl http://localhost:8000/health`
- Test manual capture: `python uxplay_capture.py --action capture`

### High latency
- UxPlay continuously writes at 30fps
- Frame capture is instant (just file read)
- If frames are stale, check AirPlay connection quality

## Technical Details

### GStreamer Pipeline
```bash
videorate ! video/x-raw,framerate=30/1 !
videoconvert !
jpegenc quality=90 !
multifilesink location=/frames/latest.jpg max-files=1
```

Components:
- `videorate`: Ensures constant 30fps
- `videoconvert`: Converts to appropriate format
- `jpegenc quality=90`: JPEG compression at 90% quality
- `multifilesink max-files=1`: Overwrites same file (no cleanup needed)

### Frame Storage
- Location: `./frames/latest.jpg`
- Format: JPEG (quality 90%)
- Update rate: 30 fps (from AirPlay stream)
- File size: ~100-200 KB per frame
- Age check: Must be < 2 seconds old

### Network Requirements
- **Ports**: 7000 (control), 7001 (data), 7100 (mirroring)
- **Protocol**: AirPlay (proprietary, reverse-engineered by UxPlay)
- **Discovery**: mDNS/Bonjour (requires multicast)
- **Network**: Same subnet required for discovery

## VisionOS App Modifications

To add capture buttons to your VisionOS app, add this to `ContentView.swift`:

```swift
Section("Frame Capture") {
    Button("Capture for ArUco Calibration") {
        Task {
            await captureCurrentFrame(purpose: "aruco_calibration")
        }
    }

    Button("Capture for ROI Selection") {
        Task {
            await captureCurrentFrame(purpose: "roi_selection")
        }
    }

    Text(captureStatus)
        .font(.caption)
        .foregroundColor(captureSuccess ? .green : .red)
}

// Add function
@MainActor
func captureCurrentFrame(purpose: String) async {
    do {
        let url = URL(string: "\(baseURL)/capture_frame?purpose=\(purpose)")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let (_, response) = try await URLSession.shared.data(for: request)

        if let httpResponse = response as? HTTPURLResponse,
           httpResponse.statusCode == 200 {
            captureStatus = "✓ Frame captured"
            captureSuccess = true
        } else {
            captureStatus = "✗ Capture failed"
            captureSuccess = false
        }
    } catch {
        captureStatus = "✗ Error: \(error.localizedDescription)"
        captureSuccess = false
    }
}
```

## Benefits of This Approach

1. **Simple**: File-based sharing, no complex streaming protocols
2. **Reliable**: Works for single-frame capture (user requirement)
3. **On-demand**: Only captures when needed, saves bandwidth
4. **UI-driven**: User controls capture timing explicitly
5. **Debugging**: Can inspect `./frames/latest.jpg` directly
6. **Hybrid**: UxPlay isolated in Docker, backend runs native
7. **No polling**: Backend doesn't waste resources checking for frames

## Workflow Integration

### ArUco Calibration
1. Place ArUco board in view of both RealSense and VisionOS
2. Start AirPlay mirroring to UxPlay
3. Click "Capture for ArUco Calibration" in Debug Viewer
4. Backend captures frame, detects ArUco markers
5. Calibration proceeds with captured frame

### ROI Selection
1. User positions object in scene
2. Start AirPlay mirroring to UxPlay
3. Click "Capture for ROI Selection" in VisionOS app
4. Backend captures frame
5. User draws ROI/mask on captured frame
6. Proceeds with pose estimation

## Performance

- **Frame capture latency**: < 50ms (file read + base64 encode)
- **Frame update rate**: 30 fps (from AirPlay)
- **Frame age check**: < 2 seconds (configurable)
- **Network bandwidth**: ~3-6 Mbps (H.264 AirPlay stream)
- **Disk I/O**: Minimal (1 file overwrite at 30fps)

## Limitations

- Requires same network for AirPlay discovery
- VisionOS must actively mirror (can't capture without mirroring)
- Frame quality limited by AirPlay compression
- Only latest frame available (no buffering)

## Future Improvements

- Add frame buffering for burst capture
- Support multiple simultaneous clients
- Add frame quality metrics (blur detection, etc.)
- Implement frame caching with timestamps
- Add REST endpoint to retrieve latest frame directly

## Related Documentation

- Main README: `/home/ag/Desktop/MA/Abgabe_2/README.md`
- Implementation Summary: `IMPLEMENTATION_SUMMARY.md`
- Completion Report: `COMPLETION_REPORT.md`
- Debug Viewer Guide: `DEBUG_VIEWER_GUIDE.md`
