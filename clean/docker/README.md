# UxPlay Docker Container for AVP Integration

This Docker container runs UxPlay to receive AirPlay screen mirroring from Apple Vision Pro (or other Apple devices) and forwards captured frames to the main API.

## Architecture

```
┌─────────────────┐
│  visionOS App   │
│ (Apple Vision)  │
└────────┬────────┘
         │ AirPlay Screen Mirroring
         ↓
┌─────────────────┐
│ Docker Container│
│    (UxPlay)     │  ← Receives AirPlay stream
└────────┬────────┘
         │ HTTP POST /receive_frame
         ↓
┌─────────────────┐
│   main_api.py   │  ← Running on host (port 8000)
│  (WSL/Linux)    │
└─────────────────┘
```

## Prerequisites

- Docker Desktop installed on Windows with WSL2 backend
- Main API running on host machine (port 8000)
- visionOS device on the same network

## Setup Instructions

### 1. Build the Docker Image

From the project root directory:

```bash
docker-compose build
```

This will:
- Build Ubuntu base image with UxPlay dependencies
- Install Python and OpenCV
- Clone and compile UxPlay from source
- Copy frame capture scripts

### 2. Start the Container

**Option A: Host Network Mode (Recommended for Linux)**

```bash
docker-compose up uxplay
```

This uses `network_mode: host` which allows UxPlay to advertise via Bonjour/mDNS properly.

**Option B: Bridge Network Mode (For Docker Desktop)**

If host mode doesn't work on Docker Desktop:

1. Edit `docker-compose.yml`
2. Comment out the `uxplay` service
3. Uncomment the `uxplay-bridge` service
4. Run:
   ```bash
   docker-compose up uxplay-bridge
   ```

### 3. Start Main API

On your host machine:

```bash
cd /home/ag/Desktop/MA/clean
python3 main_api.py
```

### 4. Connect visionOS Device

1. On your Apple Vision Pro, enable AirPlay/Screen Mirroring
2. Look for "Kubuntu Backend" in the available devices
3. Connect to it
4. Your screen should start mirroring

### 5. View Frames in Debug Viewer

```bash
python3 debug_viewer.py
```

- Click "Fetch AVP Frame" to manually get the latest frame
- Or check "Auto-update AVP" to continuously update

## Configuration

### Change API Host/Port

Edit `docker-compose.yml`:

```yaml
environment:
  - API_HOST=host.docker.internal
  - API_PORT=8000
```

### Change Server Name

Edit `docker/start_uxplay.sh`:

```bash
uxplay -n "Your Server Name" -vs "appsink" &
```

### Adjust Capture Frame Rate

Edit `docker/capture_and_send.py`:

```python
CAPTURE_FPS = 10  # Frames per second
```

## Ports Used

- **7000**: AirPlay control port
- **7001**: AirPlay data port
- **7100**: AirPlay mirror port
- **6000-6001/udp**: AirPlay audio ports

These ports must be available on your host/container.

## Troubleshooting

### "No video source found"

The capture service will keep retrying. This is normal if:
- UxPlay hasn't fully started yet
- No device is connected yet

Once you connect from visionOS, it should start working.

### "Cannot connect to API"

Check that:
- Main API is running: `curl http://localhost:8000/health`
- Firewall allows connections from Docker
- API_HOST is set correctly (`host.docker.internal` for Docker Desktop)

### Device not appearing in AirPlay list

If using bridge mode:
- Make sure all ports are mapped correctly
- Check that mDNS/Bonjour is working on your network
- Try using host network mode instead

If using host mode:
- Check that ports 7000, 7001, 7100 are not in use
- Verify network connectivity between device and host

### Frame rate is too slow

- Increase `CAPTURE_FPS` in `capture_and_send.py`
- Check network bandwidth between device and host
- Monitor Docker container resource usage

### Frames are stale/old in debug viewer

Check the "age" value:
- < 1 second: Normal operation
- > 1 second: Capture may be slow or connection interrupted

## Logs

View container logs:

```bash
docker-compose logs -f uxplay
```

View API logs:

```bash
tail -f /tmp/main_api.log
```

## Stopping the Container

```bash
docker-compose down
```

Or press Ctrl+C in the terminal where it's running.

## Advanced Usage

### Running without Docker Compose

Build:
```bash
docker build -t uxplay-avp -f docker/Dockerfile.uxplay docker/
```

Run:
```bash
docker run -d \
  --name uxplay_avp \
  --network host \
  -e API_HOST=host.docker.internal \
  -e API_PORT=8000 \
  uxplay-avp
```

### Accessing Container Shell

```bash
docker exec -it uxplay_avp bash
```

### Manual UxPlay Testing

Inside container:
```bash
uxplay -n "Test Server"
```

## Files

- `Dockerfile.uxplay` - Container image definition
- `capture_and_send.py` - Frame capture and forwarding script
- `start_uxplay.sh` - Container startup script
- `../docker-compose.yml` - Docker Compose configuration

## Next Steps

Once frames are being received:

1. **Test AVP frame display**: Click "Fetch AVP Frame" in debug viewer
2. **Enable auto-update**: Check "Auto-update AVP" checkbox
3. **Calibrate AVP camera**: Display ArUco board visible in mirroring
4. **Calculate transformation**: Show ArUco board to both RS and AVP cameras
5. **View intrinsics**: Click "Get Intrinsics" button
6. **View transformation**: Click "Get Transformation" button

## References

- UxPlay GitHub: https://github.com/antimof/UxPlay
- AVP Integration Docs: ../AVP_INTEGRATION.md
- Main API Docs: ../README.md
