#!/usr/bin/env python3
"""
Example script showing how to send head pose data to the AVP API.
This demonstrates the expected format for AVP (Apple Vision Pro) head pose data.
"""

import requests
import time
import math

API_BASE_URL = "http://localhost:5000"

def send_head_pose(position, rotation, quaternion=None, confidence=1.0, metadata=None):
    """
    Send head pose data to the API.

    Parameters:
    -----------
    position : list of 3 floats
        [x, y, z] position in meters
    rotation : list of 3 floats
        [pitch, yaw, roll] in radians (Euler angles)
    quaternion : list of 4 floats, optional
        [x, y, z, w] quaternion representation
    confidence : float, optional
        Confidence score (0.0 to 1.0)
    metadata : dict, optional
        Additional metadata
    """

    payload = {
        "position": position,
        "rotation": rotation,
        "timestamp": time.time(),
        "confidence": confidence
    }

    if quaternion is not None:
        payload["quaternion"] = quaternion
    else:
        # If no quaternion provided, send default
        payload["quaternion"] = [0, 0, 0, 1]

    if metadata is not None:
        payload["metadata"] = metadata

    try:
        response = requests.post(f"{API_BASE_URL}/head_pose", json=payload, timeout=2)
        if response.status_code == 200:
            print(f"✓ Head pose sent successfully")
            return True
        else:
            print(f"✗ Failed to send head pose: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error sending head pose: {e}")
        return False

def demo_static_pose():
    """Send a single static head pose"""
    print("\n=== Sending Static Head Pose ===")

    position = [0.0, 1.6, -0.5]  # Person standing 1.6m high, 0.5m back
    rotation = [0.1, 0.0, 0.0]    # Slightly looking down (pitch)

    send_head_pose(
        position=position,
        rotation=rotation,
        confidence=0.95,
        metadata={"device": "AVP", "tracking_quality": "high"}
    )

def demo_animated_pose(duration=10, fps=30):
    """Send animated head pose (person looking around)"""
    print(f"\n=== Sending Animated Head Pose ({duration}s @ {fps} FPS) ===")

    frame_count = 0
    start_time = time.time()

    while time.time() - start_time < duration:
        # Simulate head movement
        t = time.time() - start_time

        # Position: slight bobbing motion
        position = [
            0.0,
            1.6 + 0.02 * math.sin(t * 2),  # Slight vertical movement
            -0.5
        ]

        # Rotation: looking around
        rotation = [
            0.1 * math.sin(t * 0.5),        # Pitch (up/down)
            0.3 * math.sin(t * 0.3),        # Yaw (left/right)
            0.05 * math.sin(t * 0.7)        # Roll (tilt)
        ]

        # Convert Euler to quaternion (simplified - not mathematically accurate)
        # In real implementation, use proper quaternion math
        quaternion = [
            math.sin(rotation[0]/2),
            math.sin(rotation[1]/2),
            math.sin(rotation[2]/2),
            math.cos(rotation[0]/2)
        ]

        success = send_head_pose(
            position=position,
            rotation=rotation,
            quaternion=quaternion,
            confidence=0.92 + 0.05 * math.sin(t),
            metadata={
                "device": "AVP",
                "frame": frame_count,
                "tracking_quality": "high"
            }
        )

        if success:
            frame_count += 1

        # Maintain FPS
        time.sleep(1.0 / fps)

    print(f"\nSent {frame_count} frames in {duration} seconds ({frame_count/duration:.1f} FPS)")

def demo_retrieve_pose():
    """Retrieve and display the latest head pose from API"""
    print("\n=== Retrieving Head Pose from API ===")

    try:
        response = requests.get(f"{API_BASE_URL}/head_pose", timeout=2)
        if response.status_code == 200:
            data = response.json()
            head_pose = data.get('head_pose', {})

            print(f"\nPosition: {head_pose.get('position')}")
            print(f"Rotation: {head_pose.get('rotation')}")
            print(f"Quaternion: {head_pose.get('quaternion')}")
            print(f"Confidence: {head_pose.get('confidence')}")
            print(f"Age: {data.get('age_seconds', 0):.3f} seconds")

        elif response.status_code == 404:
            print("No head pose data available yet")
        else:
            print(f"Error retrieving head pose: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Head Pose Example - Send data to AVP API")
    print("=" * 60)

    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("\nAvailable modes:")
        print("  1. static   - Send a single static pose")
        print("  2. animated - Send animated poses (10 seconds)")
        print("  3. retrieve - Get latest pose from API")
        print()
        mode = input("Select mode (1/2/3): ").strip()

    if mode in ['1', 'static']:
        demo_static_pose()
        time.sleep(1)
        demo_retrieve_pose()

    elif mode in ['2', 'animated']:
        demo_animated_pose(duration=10, fps=30)
        demo_retrieve_pose()

    elif mode in ['3', 'retrieve']:
        demo_retrieve_pose()

    else:
        print("Invalid mode selected")
        sys.exit(1)

    print("\n" + "=" * 60)
