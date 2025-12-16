# Quick Start: AVP Integration

Get Apple Vision Pro screen mirroring working with RealSense in 5 steps.

## Step 1: Build Docker Container

```bash
cd /home/ag/Desktop/MA/clean
docker-compose build
```

⏱️ Takes ~5-10 minutes (first time only)

## Step 2: Start Main API

```bash
python3 main_api.py
```

✓ You should see:
```
RealSense pipeline started successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 3: Start UxPlay Container

In a new terminal:

```bash
cd /home/ag/Desktop/MA/clean
docker-compose up uxplay
```

✓ You should see:
```
UxPlay Container Starting
Starting UxPlay server...
Starting frame capture service...
```

## Step 4: Connect visionOS Device

On your Apple Vision Pro:
1. Open Control Center
2. Enable Screen Mirroring
3. Select **"Kubuntu Backend"**
4. Your screen starts mirroring

✓ Container logs should show:
```
✓ Found working video source
Stats: X frames captured, Y sent
```

## Step 5: Start Debug Viewer

In a third terminal:

```bash
python3 debug_viewer.py
```

Click **"Fetch AVP Frame"** → You should see your mirrored screen!

---

## What's Next?

### Automatic Updates

Check **"Auto-update AVP"** in debug viewer to continuously update frames.

### Camera Calibration

1. **Print ArUco board**: See `config.py` for board parameters
   - Dictionary: DICT_4X4_50
   - Grid: 3×4 markers
   - Marker size: 30mm
   - Separation: 10mm

2. **Calibrate RealSense**: Hold board in front of RealSense camera
3. **Calibrate AVP**: Display board visible in screen mirroring
4. Click **"Get Intrinsics"** to see calibration status

### Coordinate Transformation

Once both cameras calibrated:
1. Show ArUco board to **both cameras simultaneously**
2. Click **"Get Transformation"**
3. View T_avp_rs transformation matrix

---

## Troubleshooting

### "No video source found"

Wait ~30 seconds. If still not working:
- Check visionOS device is connected and mirroring
- Restart Docker container: `docker-compose restart uxplay`

### "Cannot connect to API"

- Check API is running: `curl http://localhost:8000/health`
- Check no firewall blocking port 8000

### Device not appearing in AirPlay list

**Linux/WSL users:**
- Use host network mode (default in docker-compose.yml)

**Docker Desktop users:**
- Try bridge mode: Edit `docker-compose.yml`, uncomment `uxplay-bridge` service

### Frames are stale (age > 1s)

- Check Docker container logs: `docker-compose logs uxplay`
- Increase frame rate in `docker/capture_and_send.py`: `CAPTURE_FPS = 15`

---

## Full Documentation

- **AVP Integration**: [AVP_INTEGRATION.md](AVP_INTEGRATION.md)
- **Docker Setup**: [docker/README.md](docker/README.md)
- **API Documentation**: [README.md](README.md)

## Stop Everything

```bash
# Stop containers
docker-compose down

# Stop API (in its terminal)
Ctrl+C

# Stop debug viewer (in its terminal)
Ctrl+C
```
