# UxPlay Docker Container

This Docker container provides a complete UxPlay AirPlay receiver environment that captures video from visionOS devices and forwards frames to the main API for pose estimation.

## Features

- **UxPlay AirPlay Receiver**: Built from latest source
- **Avahi/mDNS**: Automatic AirPlay device discovery
- **Python Integration**: Captures frames and forwards to API
- **GStreamer Pipeline**: Efficient video processing
- **Network Mode Host**: Full network access for AirPlay

## Quick Start

### 1. Build the Container

```bash
# Build the Docker image
docker-compose build

# Or build manually
docker build -t uxplay-airplay -f docker/Dockerfile .
```

### 2. Configure

Edit `docker-compose.yml` to set your configuration:

```yaml
environment:
  - UXPLAY_DEVICE_NAME=AirPlay-Pipeline  # Name visible on visionOS
  - MAIN_API_HOST=192.168.178.68         # Your host machine IP
  - MAIN_API_PORT=8000                   # Main API port
  - UXPLAY_WIDTH=1920                    # Video width
  - UXPLAY_HEIGHT=1080                   # Video height
```

**Important**: Update `MAIN_API_HOST` to your actual host machine IP address (not localhost or 127.0.0.1, since we're in a container).

### 3. Run

```bash
# Start with docker-compose (recommended)
docker-compose up

# Or run manually
docker run --rm -it \
  --network host \
  --cap-add NET_ADMIN \
  --cap-add NET_RAW \
  -e MAIN_API_HOST=192.168.178.68 \
  -e MAIN_API_PORT=8000 \
  uxplay-airplay
```

### 4. Connect from visionOS

1. On your Vision Pro, open Control Center
2. Look for "AirPlay-Pipeline" (or your custom device name)
3. Select it and start screen mirroring
4. Frames will be forwarded to your main API

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UXPLAY_DEVICE_NAME` | AirPlay-Pipeline | AirPlay device name |
| `UXPLAY_WIDTH` | 1920 | Video frame width |
| `UXPLAY_HEIGHT` | 1080 | Video frame height |
| `MAIN_API_HOST` | 192.168.178.68 | Main API host IP |
| `MAIN_API_PORT` | 8000 | Main API port |

### Network Requirements

The container uses **host network mode** for AirPlay to work properly. This is required for:
- mDNS/Bonjour discovery
- AirPlay protocol communication
- Proper port binding

### Ports Used

- **7000**: AirPlay control
- **7001**: AirPlay data
- **6000-6001**: Additional AirPlay ports
- **5353/udp**: mDNS (Bonjour/Avahi)

## Troubleshooting

### AirPlay Device Not Visible

1. **Check network mode**: Container must use `network_mode: host`
2. **Check avahi daemon**:
   ```bash
   docker exec uxplay-airplay pgrep avahi-daemon
   ```
3. **Check firewall**: Ensure ports 5353/udp, 7000, 7001 are open
4. **Same network**: Ensure Vision Pro and host are on same network
5. **Try privileged mode**: Uncomment `privileged: true` in docker-compose.yml

### No Frames Captured

1. **Check API is running**: Ensure main_api.py is running on host
2. **Check API URL**: Verify `MAIN_API_HOST` is correct (use host IP, not localhost)
3. **Check logs**:
   ```bash
   docker-compose logs -f uxplay
   ```
4. **Test connection**:
   ```bash
   docker exec uxplay-airplay curl http://192.168.178.68:8000/health
   ```

### Connection Drops

1. **Check network stability**
2. **Increase buffer size** (modify GStreamer pipeline in uxplay_module.py)
3. **Check CPU/memory** usage on host

### View Logs

```bash
# View logs
docker-compose logs -f

# View avahi status
docker exec uxplay-airplay avahi-browse -a

# Check running processes
docker exec uxplay-airplay ps aux
```

## Development

### Rebuild After Changes

```bash
# Rebuild and restart
docker-compose up --build

# Force rebuild
docker-compose build --no-cache
```

### Access Container Shell

```bash
# Start bash in running container
docker exec -it uxplay-airplay bash

# Or start with shell
docker-compose run --rm uxplay bash
```

### Test UxPlay Directly

```bash
# Inside container
uxplay -n TestDevice
```

## Architecture

```
┌─────────────────┐
│   Vision Pro    │
│  (visionOS)     │
└────────┬────────┘
         │ AirPlay
         │ (mDNS discovery via avahi)
         ▼
┌─────────────────────────────┐
│   Docker Container          │
│  ┌─────────────────────┐   │
│  │  Avahi Daemon       │   │
│  │  (mDNS/Bonjour)     │   │
│  └──────────┬──────────┘   │
│             ▼               │
│  ┌─────────────────────┐   │
│  │  UxPlay Process     │   │
│  │  (AirPlay Server)   │   │
│  └──────────┬──────────┘   │
│             │ stdout        │
│             ▼               │
│  ┌─────────────────────┐   │
│  │  uxplay_module.py   │   │
│  │  (Frame Capture)    │   │
│  └──────────┬──────────┘   │
└─────────────┼───────────────┘
              │ HTTP POST
              ▼
┌─────────────────────────────┐
│   Host Machine              │
│  ┌─────────────────────┐   │
│  │  main_api.py        │   │
│  │  (Pose Estimation)  │   │
│  └─────────────────────┘   │
└─────────────────────────────┘
```

## Notes

- Container runs in **host network mode** for AirPlay compatibility
- Avahi daemon provides mDNS/Bonjour for device discovery
- Frames are captured via GStreamer pipeline and sent to main API
- Built on Ubuntu 22.04 with latest UxPlay from source

## License

See main project LICENSE file.
