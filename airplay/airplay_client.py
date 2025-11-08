#!/usr/bin/env python3
"""
AirPlay Screen Mirroring Client
Mirrors your Windows/Linux screen to a Mac mini using AirPlay protocol
"""

import socket
import struct
import time
import threading
from zeroconf import ServiceBrowser, Zeroconf, ServiceListener
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AirPlayDevice:
    """Represents an AirPlay-capable device (e.g., Mac mini)"""

    def __init__(self, name: str, address: str, port: int, properties: Dict):
        self.name = name
        self.address = address
        self.port = port
        self.properties = properties

    def __str__(self):
        return f"{self.name} ({self.address}:{self.port})"


class AirPlayDiscovery(ServiceListener):
    """Discovers AirPlay devices on the network using mDNS/Bonjour"""

    def __init__(self):
        self.devices: List[AirPlayDevice] = []
        self.lock = threading.Lock()

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when an AirPlay service is discovered"""
        info = zc.get_service_info(type_, name)
        if info:
            with self.lock:
                # Parse addresses (IPv4)
                addresses = [socket.inet_ntoa(addr) for addr in info.addresses
                           if len(addr) == 4]

                if addresses:
                    properties = {}
                    if info.properties:
                        properties = {k.decode('utf-8'): v.decode('utf-8')
                                    for k, v in info.properties.items()}

                    device = AirPlayDevice(
                        name=info.server.rstrip('.'),
                        address=addresses[0],
                        port=info.port,
                        properties=properties
                    )
                    self.devices.append(device)
                    logger.info(f"Discovered AirPlay device: {device}")

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when an AirPlay service is removed"""
        logger.info(f"AirPlay device removed: {name}")

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        """Called when an AirPlay service is updated"""
        pass

    def get_devices(self) -> List[AirPlayDevice]:
        """Get list of discovered devices"""
        with self.lock:
            return self.devices.copy()


class AirPlayConnection:
    """Manages connection to an AirPlay device"""

    def __init__(self, device: AirPlayDevice):
        self.device = device
        self.session_id = None
        self.connected = False

    def connect(self) -> bool:
        """Establish connection to the AirPlay device"""
        try:
            logger.info(f"Connecting to {self.device}...")
            import requests

            # First, check device info to understand capabilities
            if not self._check_device_info():
                logger.warning("Could not retrieve device info, proceeding anyway...")

            # Check if device requires pairing/authentication
            features = self.device.properties.get('features', '0x0')
            logger.info(f"Device features: {features}")
            logger.info(f"Device properties: {self.device.properties}")

            # Try to establish connection using different methods

            # Method 1: Try /info endpoint first to check compatibility
            info_url = f"http://{self.device.address}:{self.device.port}/info"
            headers = self._get_common_headers()

            try:
                info_response = requests.get(info_url, headers=headers, timeout=5)
                if info_response.status_code == 200:
                    logger.info(f"Device info response: {info_response.text[:200]}")
            except Exception as e:
                logger.warning(f"Could not get device info: {e}")

            # Method 2: Try pairing if required
            if self._requires_authentication():
                logger.info("Device requires authentication, attempting pairing...")
                if not self._pair_device():
                    logger.error("Pairing failed")
                    return False

            # Method 3: Establish the actual connection
            # Try POST to /play for mirroring initiation
            play_url = f"http://{self.device.address}:{self.device.port}/play"

            # Prepare connection headers
            connection_headers = self._get_common_headers()
            connection_headers.update({
                'Content-Type': 'application/x-apple-binary-plist',
                'X-Apple-Session-ID': self._generate_session_id(),
            })

            # Try initiating connection
            logger.info("Attempting to initiate AirPlay session...")
            response = requests.post(play_url, headers=connection_headers, timeout=5)

            if response.status_code in [200, 101]:
                self.connected = True
                logger.info(f"Successfully connected to {self.device.name}")
                return True
            else:
                logger.error(f"Play endpoint failed with status {response.status_code}")

                # Try alternative: reverse connection method
                logger.info("Trying reverse connection method...")
                reverse_url = f"http://{self.device.address}:{self.device.port}/reverse"
                reverse_response = requests.post(reverse_url, headers=connection_headers, timeout=5)

                if reverse_response.status_code in [200, 101]:
                    self.connected = True
                    logger.info(f"Successfully connected via reverse method")
                    return True
                else:
                    logger.error(f"Reverse connection failed with status {reverse_response.status_code}")
                    logger.error(f"Response body: {reverse_response.text[:200] if reverse_response.text else 'empty'}")
                    return False

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _get_common_headers(self) -> dict:
        """Get common HTTP headers for AirPlay requests"""
        return {
            'User-Agent': 'AirPlay/550.10',
            'X-Apple-Device-ID': self._get_device_id(),
            'X-Apple-Transition': 'Dissolve',
        }

    def _check_device_info(self) -> bool:
        """Check device information and capabilities"""
        try:
            import requests
            url = f"http://{self.device.address}:{self.device.port}/server-info"
            headers = self._get_common_headers()

            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                logger.info(f"Device supports server-info endpoint")
                return True
            return False
        except:
            return False

    def _requires_authentication(self) -> bool:
        """Check if device requires authentication"""
        # Check device properties for authentication requirements
        features = self.device.properties.get('features', '0x0')
        # Convert hex string to int and check authentication bit
        try:
            features_int = int(features, 16) if isinstance(features, str) else int(features)
            # Bit 14 (0x4000) indicates authentication required
            return (features_int & 0x4000) != 0
        except:
            return False

    def _pair_device(self) -> bool:
        """Attempt to pair with the device"""
        try:
            import requests

            # Try pairing endpoint
            pair_url = f"http://{self.device.address}:{self.device.port}/pair-setup"
            headers = self._get_common_headers()
            headers['Content-Type'] = 'application/octet-stream'

            # Note: Real pairing requires SRP (Secure Remote Password) protocol
            # This is a simplified attempt
            logger.info("Attempting device pairing (note: may require user interaction on receiver)")

            response = requests.post(pair_url, headers=headers, timeout=10)

            if response.status_code == 200:
                logger.info("Pairing successful")
                return True
            else:
                logger.warning(f"Pairing returned status {response.status_code}")
                # Don't fail here, as some devices may not need pairing
                return True

        except Exception as e:
            logger.warning(f"Pairing attempt failed: {e}")
            # Don't fail here, continue to try connection anyway
            return True

    def _get_device_id(self) -> str:
        """Generate a unique device identifier"""
        import uuid
        mac = ':'.join(['{:02x}'.format((uuid.getnode() >> i) & 0xff)
                       for i in range(0, 8*6, 8)][::-1])
        return mac.upper()

    def _generate_session_id(self) -> str:
        """Generate a session ID for this connection"""
        import uuid
        session_id = str(uuid.uuid4())
        self.session_id = session_id
        return session_id

    def disconnect(self):
        """Disconnect from the AirPlay device"""
        if self.connected:
            logger.info(f"Disconnecting from {self.device.name}")
            self.connected = False


class AirPlayStreamer:
    """Handles screen capture and streaming to AirPlay device"""

    def __init__(self, connection: AirPlayConnection):
        self.connection = connection
        self.streaming = False
        self.stream_thread = None

    def start_streaming(self):
        """Start capturing and streaming the screen"""
        if not self.connection.connected:
            logger.error("Not connected to AirPlay device")
            return False

        self.streaming = True
        self.stream_thread = threading.Thread(target=self._stream_loop)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        logger.info("Started screen streaming")
        return True

    def stop_streaming(self):
        """Stop streaming"""
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2)
        logger.info("Stopped screen streaming")

    def _stream_loop(self):
        """Main streaming loop - captures and sends frames"""
        try:
            from mss import mss
            import cv2
            import numpy as np

            with mss() as sct:
                # Get the primary monitor
                monitor = sct.monitors[1]

                while self.streaming:
                    # Capture screen
                    screenshot = sct.grab(monitor)

                    # Convert to numpy array
                    frame = np.array(screenshot)

                    # Convert BGRA to BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                    # Encode frame
                    encoded_frame = self._encode_frame(frame)

                    # Send frame to AirPlay device
                    self._send_frame(encoded_frame)

                    # Control frame rate (30 fps)
                    time.sleep(1/30)

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            self.streaming = False

    def _encode_frame(self, frame):
        """Encode frame using H.264"""
        import cv2

        # Resize for better streaming performance (1280x720)
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            new_width = 1280
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))

        # Encode as JPEG for now (H.264 requires more complex setup)
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buffer.tobytes()

    def _send_frame(self, frame_data):
        """Send encoded frame to AirPlay device"""
        # This is a simplified version
        # Real AirPlay requires RTSP/RTP streaming with proper timing
        # For now, we'll just log the attempt
        pass


class AirPlayClient:
    """Main AirPlay client for screen mirroring"""

    def __init__(self):
        self.discovery = AirPlayDiscovery()
        self.zeroconf = None
        self.connection: Optional[AirPlayConnection] = None
        self.streamer: Optional[AirPlayStreamer] = None

    def discover_devices(self, timeout: float = 5.0) -> List[AirPlayDevice]:
        """Discover AirPlay devices on the network"""
        logger.info("Discovering AirPlay devices...")

        self.zeroconf = Zeroconf()
        browser = ServiceBrowser(
            self.zeroconf,
            "_airplay._tcp.local.",
            self.discovery
        )

        # Wait for discovery
        time.sleep(timeout)

        devices = self.discovery.get_devices()
        logger.info(f"Found {len(devices)} AirPlay device(s)")

        return devices

    def connect_to_device(self, device: AirPlayDevice) -> bool:
        """Connect to a specific AirPlay device"""
        self.connection = AirPlayConnection(device)
        success = self.connection.connect()

        if success:
            self.streamer = AirPlayStreamer(self.connection)

        return success

    def start_mirroring(self) -> bool:
        """Start screen mirroring"""
        if not self.streamer:
            logger.error("Not connected to any device")
            return False

        return self.streamer.start_streaming()

    def stop_mirroring(self):
        """Stop screen mirroring"""
        if self.streamer:
            self.streamer.stop_streaming()

    def disconnect(self):
        """Disconnect from device and cleanup"""
        if self.streamer:
            self.streamer.stop_streaming()

        if self.connection:
            self.connection.disconnect()

        if self.zeroconf:
            self.zeroconf.close()

        logger.info("Disconnected")


def main():
    """Main entry point"""
    print("=" * 60)
    print("AirPlay Screen Mirroring Client")
    print("Mirror your screen to Mac mini or Apple TV")
    print("=" * 60)
    print()

    client = AirPlayClient()

    try:
        # Discover devices
        devices = client.discover_devices(timeout=5)

        if not devices:
            print("No AirPlay devices found on the network.")
            print("Make sure your Mac mini has AirPlay Receiver enabled in System Settings.")
            return

        # Display devices
        print("\nAvailable AirPlay devices:")
        for i, device in enumerate(devices, 1):
            print(f"{i}. {device}")

        # Select device
        choice = input("\nSelect device number (or 'q' to quit): ").strip()

        if choice.lower() == 'q':
            return

        try:
            device_index = int(choice) - 1
            if device_index < 0 or device_index >= len(devices):
                print("Invalid selection")
                return

            selected_device = devices[device_index]
        except ValueError:
            print("Invalid input")
            return

        # Connect to device
        print(f"\nConnecting to {selected_device.name}...")
        if not client.connect_to_device(selected_device):
            print("Failed to connect to device")
            return

        print("\n" + "=" * 60)
        print("IMPORTANT NOTE:")
        print("=" * 60)
        print("This is a proof-of-concept implementation.")
        print("Full AirPlay screen mirroring requires:")
        print("  1. FairPlay DRM encryption (proprietary)")
        print("  2. H.264 hardware encoding")
        print("  3. RTSP/RTP streaming protocol")
        print("  4. Device pairing and authentication")
        print()
        print("For production use, consider:")
        print("  - Commercial solutions (AirParrot, Reflector)")
        print("  - Using built-in macOS screen sharing")
        print("=" * 60)
        print()

        input("Press Enter to exit...")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        client.disconnect()


if __name__ == "__main__":
    main()
