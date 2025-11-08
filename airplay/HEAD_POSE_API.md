# Head Pose API Documentation

## Overview

The AVP API now includes endpoints to receive and retrieve head pose data from Apple Vision Pro (AVP) or other tracking devices.

## Endpoints

### POST /head_pose

Send head pose data to the API.

**Request Body (JSON):**

```json
{
  "position": [x, y, z],           // Position in meters (required)
  "rotation": [pitch, yaw, roll],  // Euler angles in radians (required)
  "quaternion": [x, y, z, w],      // Quaternion (optional)
  "timestamp": 1234567890.123,     // Unix timestamp (optional, defaults to current time)
  "confidence": 0.95,              // Tracking confidence 0.0-1.0 (optional, defaults to 1.0)
  "metadata": {                    // Additional metadata (optional)
    "device": "AVP",
    "tracking_quality": "high",
    "frame": 123
  }
}
```

**Response (Success):**

```json
{
  "success": true,
  "received_at": 1234567890.456
}
```

**Example:**

```python
import requests

payload = {
    "position": [0.0, 1.6, -0.5],
    "rotation": [0.1, 0.0, 0.0],
    "quaternion": [0, 0, 0, 1],
    "confidence": 0.95,
    "metadata": {"device": "AVP"}
}

response = requests.post("http://localhost:5000/head_pose", json=payload)
print(response.json())
```

---

### GET /head_pose

Retrieve the latest head pose data from the API.

**Response (Success):**

```json
{
  "head_pose": {
    "position": [0.0, 1.6, -0.5],
    "rotation": [0.1, 0.0, 0.0],
    "quaternion": [0, 0, 0, 1],
    "timestamp": 1234567890.123,
    "confidence": 0.95,
    "metadata": {
      "device": "AVP",
      "tracking_quality": "high"
    }
  },
  "received_at": 1234567890.456,
  "age_seconds": 0.333
}
```

**Response (No Data Available):**

```json
{
  "error": "No head pose data available"
}
```

Status Code: `404`

**Example:**

```python
import requests

response = requests.get("http://localhost:5000/head_pose")
if response.status_code == 200:
    data = response.json()
    print(f"Position: {data['head_pose']['position']}")
    print(f"Data age: {data['age_seconds']:.3f} seconds")
else:
    print("No head pose data available")
```

---

## Data Format Details

### Position

3D position in meters relative to the tracking origin:

- `x`: Left/right (negative = left, positive = right)
- `y`: Up/down (negative = down, positive = up)
- `z`: Forward/backward (negative = backward, positive = forward)

Example: `[0.0, 1.6, -0.5]` = centered, 1.6m high, 0.5m back

### Rotation (Euler Angles)

Rotation in radians using the intrinsic rotation sequence:

- `pitch`: Rotation around X-axis (nodding up/down)
- `yaw`: Rotation around Y-axis (turning left/right)
- `roll`: Rotation around Z-axis (tilting head)

Example: `[0.1, 0.0, 0.0]` = slightly looking down

### Quaternion

Quaternion representation of rotation (more stable than Euler):

- `[x, y, z, w]`

Identity (no rotation): `[0, 0, 0, 1]`

### Confidence

Tracking quality/confidence score:

- `1.0` = perfect tracking
- `0.5` = moderate quality
- `0.0` = lost tracking

### Metadata

Optional dictionary containing additional information:

```json
{
  "device": "AVP",
  "tracking_quality": "high",
  "frame": 123,
  "session_id": "abc123",
  "user_id": "user456"
}
```

---

## Integration with Client

The Tkinter client (`tk_hypercam_2.py`) automatically displays head pose data when available:

1. Enable the "Show Head Pose" checkbox in the client
2. View data in the "Head Pose" tab
3. Data includes:
   - Position (x, y, z)
   - Rotation (pitch, yaw, roll)
   - Quaternion (x, y, z, w)
   - Confidence score
   - Data age (staleness indicator)

---

## Example Usage

### Send from AVP Device

```python
import requests
import time

def send_avp_pose(position, rotation):
    """Send head pose from Apple Vision Pro"""
    payload = {
        "position": position,
        "rotation": rotation,
        "timestamp": time.time(),
        "confidence": 0.95,
        "metadata": {"device": "AVP"}
    }
    requests.post("http://localhost:5000/head_pose", json=payload)

# Example: Send current pose
send_avp_pose(
    position=[0.0, 1.6, -0.5],
    rotation=[0.1, 0.0, 0.0]
)
```

### Continuous Streaming

```python
import requests
import time

while True:
    # Get head pose from your tracking system
    position, rotation = get_current_head_pose()

    # Send to API
    requests.post("http://localhost:5000/head_pose", json={
        "position": position,
        "rotation": rotation,
        "timestamp": time.time(),
        "confidence": 0.95
    })

    # Stream at 30 FPS
    time.sleep(1/30)
```

### Retrieve and Use

```python
import requests

# Get latest head pose
response = requests.get("http://localhost:5000/head_pose")
if response.status_code == 200:
    data = response.json()
    head_pose = data['head_pose']

    # Use the data
    position = head_pose['position']
    rotation = head_pose['rotation']

    # Check data freshness
    age = data['age_seconds']
    if age > 1.0:
        print("Warning: Head pose data is stale!")
```

---

## Testing

Use the provided example script:

```bash
# Send static pose
python send_head_pose_example.py static

# Send animated poses (10 seconds)
python send_head_pose_example.py animated

# Retrieve current pose
python send_head_pose_example.py retrieve
```

---

## Notes

- Head pose data is stored in memory (not persisted)
- Only the latest pose is kept
- Data age is calculated automatically
- Thread-safe implementation
- No authentication required (localhost only)

---

## Workflow

```
┌─────────────────┐
│  AVP / Tracking │
│     Device      │
└────────┬────────┘
         │ POST /head_pose
         ↓
┌─────────────────┐
│   AVP API       │
│  (Flask Server) │
└────────┬────────┘
         │ GET /head_pose
         ↓
┌─────────────────┐
│  Tkinter Client │
│   (Display)     │
└─────────────────┘
```

---

## Error Handling

**400 Bad Request:**
- No data provided in POST request

**404 Not Found:**
- No head pose data available (GET request)

**500 Internal Server Error:**
- Exception during processing
- Check API logs for details
