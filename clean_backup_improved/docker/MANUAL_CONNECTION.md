# Manual Connection Guide for Vision Pro

## Why Auto-Discovery Doesn't Work

mDNS/Bonjour service discovery doesn't work from Docker containers on Windows/WSL due to:
- Multicast networking limitations in Docker Desktop
- Port 5353 already in use by Windows' Bonjour service
- Network isolation between container and host

## Solution: Connect Manually by IP

### Step 1: Find Your Computer's IP Address

On your Windows host, open PowerShell or CMD and run:
```bash
ipconfig
```

Look for your network adapter's IPv4 address. Example:
```
Wireless LAN adapter Wi-Fi:
   IPv4 Address. . . . . . . . . . . : 192.168.1.100
```

### Step 2: Start the UxPlay Container

```bash
cd docker
docker-compose up uxplay
```

Wait for the message: "UxPlay should now be advertising on the network..."

### Step 3: Connect from Vision Pro

On your Vision Pro:

1. Open **Settings** → **AirPlay & Handoff**
2. If "PoseAPI" doesn't appear automatically, use the manual connection method:
   - Some visionOS versions support manual IP entry in AirPlay settings
   - Or use the **Control Center** and look for screen mirroring options

3. **Alternative**: Use the **Files app** or **Safari** on Vision Pro:
   - Try connecting to: `http://192.168.1.100:7000` (replace with your IP)
   - Some AirPlay clients support manual server URLs

### Step 4: Verify Connection

If connected successfully, you should see:
- Container logs showing: "Raop-rtp: SETUP, setup socket"
- Frame capture script detecting video source
- Frames being sent to the API at `http://host.docker.internal:8000/receive_frame`

## Alternative: Run UxPlay Natively

If Docker networking continues to cause issues, run UxPlay directly on Windows/WSL:

### On WSL (Ubuntu):
```bash
# Install UxPlay
sudo apt-get update
sudo apt-get install uxplay

# Run it
uxplay -n "PoseAPI" -fps 30 -v
```

This bypasses Docker networking entirely and should make mDNS discovery work reliably. Your Vision Pro should see "PoseAPI" automatically.

## Troubleshooting

### Can't see UxPlay at all?
- Ensure Vision Pro and computer are on the same WiFi network
- Check Windows Firewall isn't blocking ports 7000, 7001, 7100, 6000, 6001
- Try disabling antivirus temporarily to test

### Connection drops or freezes?
- Check WiFi signal strength on both devices
- Reduce network congestion (close streaming apps, downloads)
- Try reducing FPS in UxPlay: `-fps 15` instead of `-fps 30`

### Still not working?
Check the container logs for specific error messages:
```bash
docker logs uxplay_avp
```
