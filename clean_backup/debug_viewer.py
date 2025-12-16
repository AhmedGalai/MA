#!/usr/bin/env python3
"""
Debug Viewer for Pose Estimation Pipeline

A tkinter-based visual debugging tool for monitoring the pose estimation pipeline
in real-time. Displays camera feeds, system status, poses, and statistics in a
configurable grid layout.

Connects to main_api.py running on localhost:8000 and polls API endpoints for
live data updates.

Author: Debug Tools
Date: 2025-12-14
"""

import tkinter as tk
from tkinter import ttk
import threading
import requests
import logging
import numpy as np
import cv2
import base64
from PIL import Image, ImageTk
from datetime import datetime
from typing import Optional, Dict, Any
import json
from pathlib import Path
import time

# Try to import config, fall back to defaults if not available
try:
    from config import CONFIG
    API_HOST = CONFIG["network"]["main_api_host"]
    API_PORT = CONFIG["network"]["main_api_port"]
except ImportError:
    API_HOST = "127.0.0.1"
    API_PORT = 8000

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DebugViewer:
    """
    Tkinter-based debug viewer for pose estimation pipeline.

    Displays real-time camera feeds, system status, pose matrices, and statistics
    in a 2x3 grid layout with configurable polling and display refresh rates.

    Attributes:
        api_url (str): Base URL for API connections
        root (tk.Tk): Root tkinter window
        is_connected (bool): Current connection state
        update_rate_hz (float): Current polling rate in Hz
        stats (dict): Statistics tracking (frame count, success rate, timing)
    """

    def __init__(self, api_url: str = None, width: int = 1280, height: int = 800):
        """
        Initialize the DebugViewer.

        Args:
            api_url (str): Base URL for API (e.g., 'http://localhost:8000').
                          If None, constructs from API_HOST and API_PORT.
            width (int): Window width in pixels. Default: 1280
            height (int): Window height in pixels. Default: 800
        """
        # Set API URL
        if api_url is None:
            self.api_url = f"http://{API_HOST}:{API_PORT}"
        else:
            self.api_url = api_url.rstrip('/')

        # Window and state
        self.root = tk.Tk()
        self.root.title("Pose Estimation Pipeline Debug Viewer")
        self.root.geometry(f"{width}x{height}")

        # Connection and update state
        self.is_connected = False
        self.update_rate_hz = 2.0
        self.polling_thread = None
        self.should_stop = False

        # View selection state
        self.current_view = "All Cameras"  # Default view

        # Statistics tracking
        self.stats = {
            'total_frames': 0,
            'successful_estimates': 0,
            'failed_estimates': 0,
            'last_update_time': None,
            'average_frame_time': 0.0,
            'frame_times': []
        }

        # Cached frame data
        self.cached_data = {
            'status': {},
            'rgb_image': None,
            'depth_image': None,
            'aruco_image': None,
            'aruco_markers': 0,
            'aruco_ids': [],
            'avp_rgb_image': None,
            'avp_aruco_image': None,
            'avp_timestamp': None,
            'avp_age': None,
            'intrinsics_rs': None,
            'intrinsics_avp': None,
            'transformation': None,
            'last_fetch_time': None
        }

        # Setup GUI
        self._setup_gui()

        # Setup window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        logger.info(f"DebugViewer initialized with API URL: {self.api_url}")

    def _setup_gui(self):
        """Setup the tkinter GUI layout."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Title and connection status bar
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(
            title_frame,
            text="Pose Estimation Pipeline Debug Viewer",
            font=("Arial", 14, "bold")
        ).pack(side=tk.LEFT)

        self.status_indicator = ttk.Label(
            title_frame,
            text="● Disconnected",
            font=("Arial", 10),
            foreground="red"
        )
        self.status_indicator.pack(side=tk.RIGHT, padx=10)

        # Divider
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # View selection
        view_frame = ttk.Frame(main_frame)
        view_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(view_frame, text="View:", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        self.view_var = tk.StringVar(value=self.current_view)
        view_options = ["All Cameras", "RealSense Only", "AVP Only", "Side-by-Side"]
        self.view_selector = ttk.Combobox(
            view_frame,
            textvariable=self.view_var,
            values=view_options,
            state="readonly",
            width=20
        )
        self.view_selector.pack(side=tk.LEFT, padx=5)
        self.view_selector.bind("<<ComboboxSelected>>", self._on_view_changed)

        # Divider
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Content area - split into images (top) and data (bottom)
        content_container = ttk.Frame(main_frame)
        content_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Images area (2/3 of space)
        images_frame = ttk.Frame(content_container)
        images_frame.pack(fill=tk.BOTH, expand=True)

        # Create grid for image panels (2 rows x 3 columns)
        for i in range(2):
            images_frame.grid_rowconfigure(i, weight=1, minsize=200)
        for j in range(3):
            images_frame.grid_columnconfigure(j, weight=1, minsize=250)

        # Create all image panels (will show/hide based on view selection)
        self.panel_rgb = self._create_image_panel(
            images_frame, "RealSense RGB", 0, 0
        )
        self.panel_aruco = self._create_image_panel(
            images_frame, "RS ArUco Detection", 0, 1
        )
        self.panel_depth = self._create_image_panel(
            images_frame, "RealSense Depth", 0, 2
        )
        self.panel_avp_rgb = self._create_image_panel(
            images_frame, "AVP RGB", 1, 0
        )
        self.panel_avp_aruco = self._create_image_panel(
            images_frame, "AVP ArUco Detection", 1, 1
        )
        # Empty slot at 1,2 for future use

        # Divider
        ttk.Separator(content_container, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Consolidated data panel (1/3 of space)
        data_frame = ttk.LabelFrame(content_container, text="System Data", padding=5)
        data_frame.pack(fill=tk.BOTH, expand=False, pady=5)

        # Scrollable text area for all data
        text_scroll_frame = ttk.Frame(data_frame)
        text_scroll_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_scroll_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.data_text = tk.Text(
            text_scroll_frame,
            height=15,
            font=("Courier", 9),
            bg="#f0f0f0",
            relief=tk.SUNKEN,
            state=tk.DISABLED,
            yscrollcommand=scrollbar.set,
            wrap=tk.WORD
        )
        self.data_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.data_text.yview)

        # Divider
        ttk.Separator(main_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)

        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Left controls
        left_control = ttk.Frame(control_frame)
        left_control.pack(side=tk.LEFT, padx=5)

        self.connect_button = ttk.Button(
            left_control,
            text="Connect",
            command=self.toggle_connection,
            width=15
        )
        self.connect_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            left_control,
            text="Manual Refresh",
            command=self.manual_refresh,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            left_control,
            text="Clear Stats",
            command=self.clear_statistics,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        # Right controls - refresh rate
        right_control = ttk.Frame(control_frame)
        right_control.pack(side=tk.RIGHT, padx=5)

        ttk.Label(right_control, text="Refresh Rate (Hz):").pack(side=tk.LEFT, padx=5)

        self.refresh_rate_var = tk.DoubleVar(value=self.update_rate_hz)
        self.refresh_slider = ttk.Scale(
            right_control,
            from_=1,
            to=10,
            orient=tk.HORIZONTAL,
            variable=self.refresh_rate_var,
            command=self._on_refresh_rate_change,
            length=200
        )
        self.refresh_slider.pack(side=tk.LEFT, padx=5)

        self.refresh_label = ttk.Label(right_control, text="2.0 Hz", width=6)
        self.refresh_label.pack(side=tk.LEFT, padx=5)

        # AVP Controls section
        avp_frame = ttk.LabelFrame(main_frame, text="AVP (Apple Vision Pro) Controls", padding=10)
        avp_frame.pack(fill=tk.X, padx=5, pady=5)

        # AVP buttons
        avp_buttons = ttk.Frame(avp_frame)
        avp_buttons.pack(side=tk.LEFT, padx=5)

        self.fetch_avp_btn = ttk.Button(
            avp_buttons,
            text="Fetch AVP Frame",
            command=self._fetch_avp_frame,
            width=20
        )
        self.fetch_avp_btn.pack(side=tk.LEFT, padx=5)

        self.auto_avp_var = tk.BooleanVar(value=False)
        self.auto_avp_check = ttk.Checkbutton(
            avp_buttons,
            text="Auto-update AVP",
            variable=self.auto_avp_var
        )
        self.auto_avp_check.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            avp_buttons,
            text="Get Intrinsics",
            command=self._fetch_intrinsics,
            width=15
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            avp_buttons,
            text="Get Transformation",
            command=self._fetch_transformation,
            width=18
        ).pack(side=tk.LEFT, padx=5)

        # Status display
        avp_status_frame = ttk.Frame(avp_frame)
        avp_status_frame.pack(side=tk.RIGHT, padx=5)

        ttk.Label(avp_status_frame, text="AVP Status:").pack(side=tk.LEFT, padx=5)

        self.avp_status_var = tk.StringVar(value="Not connected")
        self.avp_status_label = ttk.Label(
            avp_status_frame,
            textvariable=self.avp_status_var,
            font=("Arial", 9, "italic"),
            foreground="gray"
        )
        self.avp_status_label.pack(side=tk.LEFT, padx=5)

    def _create_image_panel(self, parent, title: str, row: int, col: int) -> Dict:
        """
        Create an image display panel.

        Args:
            parent: Parent tkinter widget
            title (str): Panel title
            row (int): Grid row
            col (int): Grid column

        Returns:
            dict: Panel info dict with canvas and photo reference
        """
        frame = ttk.LabelFrame(parent, text=title, padding=5)
        frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

        canvas = tk.Canvas(frame, bg="black", width=320, height=240)
        canvas.pack(fill=tk.BOTH, expand=True)

        return {
            'frame': frame,
            'canvas': canvas,
            'photo': None,
            'title': title
        }

    def _create_text_panel(self, parent, title: str, row: int, col: int) -> Dict:
        """
        Create a text display panel.

        Args:
            parent: Parent tkinter widget
            title (str): Panel title
            row (int): Grid row
            col (int): Grid column

        Returns:
            dict: Panel info dict with text widget
        """
        frame = ttk.LabelFrame(parent, text=title, padding=5)
        frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

        text = tk.Text(
            frame,
            height=10,
            width=40,
            font=("Courier", 9),
            bg="#f0f0f0",
            relief=tk.SUNKEN,
            state=tk.DISABLED
        )
        text.pack(fill=tk.BOTH, expand=True)

        return {
            'frame': frame,
            'text': text,
            'title': title
        }

    def _on_refresh_rate_change(self, value):
        """Handle refresh rate slider change."""
        self.update_rate_hz = float(value)
        self.refresh_label.config(text=f"{self.update_rate_hz:.1f} Hz")

    def _on_view_changed(self, event=None):
        """Handle view selection change."""
        self.current_view = self.view_var.get()
        self._update_panel_visibility()
        logger.info(f"View changed to: {self.current_view}")

    def _update_panel_visibility(self):
        """Show/hide image panels based on current view selection."""
        view = self.current_view

        if view == "All Cameras":
            # Show all panels
            self.panel_rgb['frame'].grid()
            self.panel_aruco['frame'].grid()
            self.panel_depth['frame'].grid()
            self.panel_avp_rgb['frame'].grid()
            self.panel_avp_aruco['frame'].grid()

        elif view == "RealSense Only":
            # Show only RealSense panels
            self.panel_rgb['frame'].grid()
            self.panel_aruco['frame'].grid()
            self.panel_depth['frame'].grid()
            self.panel_avp_rgb['frame'].grid_remove()
            self.panel_avp_aruco['frame'].grid_remove()

        elif view == "AVP Only":
            # Show only AVP panels
            self.panel_rgb['frame'].grid_remove()
            self.panel_aruco['frame'].grid_remove()
            self.panel_depth['frame'].grid_remove()
            self.panel_avp_rgb['frame'].grid()
            self.panel_avp_aruco['frame'].grid()

        elif view == "Side-by-Side":
            # Show RGB comparison: RS RGB + AVP RGB
            self.panel_rgb['frame'].grid()
            self.panel_avp_rgb['frame'].grid()
            self.panel_aruco['frame'].grid_remove()
            self.panel_depth['frame'].grid_remove()
            self.panel_avp_aruco['frame'].grid_remove()

    def toggle_connection(self):
        """Toggle connection to API."""
        if self.is_connected:
            self.disconnect()
        else:
            self.connect()

    def connect(self):
        """
        Connect to the API and start polling.

        Launches a background thread to continuously poll API endpoints at the
        configured refresh rate.
        """
        try:
            # Test connection
            response = requests.get(
                f"{self.api_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                self.is_connected = True
                self.connect_button.config(text="Disconnect")
                self._update_status_indicator(True)

                # Start polling thread
                if self.polling_thread is None or not self.polling_thread.is_alive():
                    self.should_stop = False
                    self.polling_thread = threading.Thread(
                        target=self._polling_loop,
                        daemon=True
                    )
                    self.polling_thread.start()

                logger.info(f"Connected to API at {self.api_url}")
            else:
                logger.warning(f"API health check returned {response.status_code}")
                self._update_status_text("Status", f"Connection failed (HTTP {response.status_code})")

        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to {self.api_url}")
            self._update_status_text("Status", f"Connection failed - Cannot reach {self.api_url}")
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self._update_status_text("Status", f"Connection error: {str(e)[:50]}")

    def disconnect(self):
        """Disconnect from the API and stop polling."""
        self.is_connected = False
        self.should_stop = True
        self.connect_button.config(text="Connect")
        self._update_status_indicator(False)
        self._update_status_text("Status", "Disconnected")

        logger.info("Disconnected from API")

    def _polling_loop(self):
        """Background thread loop that polls API at configured rate."""
        while self.is_connected and not self.should_stop:
            try:
                # Schedule GUI update on main thread for thread safety
                self.root.after(0, self.update_display)
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")

            # Sleep based on refresh rate - use time.sleep() not root.after()
            sleep_time = 1.0 / self.update_rate_hz
            time.sleep(sleep_time)

    def manual_refresh(self):
        """Manually trigger a display update."""
        try:
            self.update_display()
        except Exception as e:
            logger.error(f"Error in manual refresh: {e}")

    def update_display(self):
        """
        Main update loop that fetches data and updates all panels.

        Fetches status from /health endpoint and RGBD frames from /get_rgbd_frame.
        Updates:
        - System status panel
        - RGB and Depth image panels
        - AVP panels (if auto-update enabled)
        - Statistics
        - Display refresh timestamp
        """
        start_time = datetime.now()

        try:
            # Fetch status
            status = self.fetch_status()
            if status is not None:
                self.cached_data['status'] = status
                self.cached_data['last_fetch_time'] = datetime.now()
                self._update_status_text_panel()
                self.stats['total_frames'] += 1

                # Fetch and display RGBD frames if RealSense is connected
                if status.get('rs_connected', False):
                    rgbd_data = self.fetch_rgbd_frame()
                    if rgbd_data is not None:
                        self._update_rgbd_panels(rgbd_data)

                    # Fetch and display ArUco detection frame
                    aruco_data = self.fetch_aruco_frame()
                    if aruco_data is not None:
                        self._update_aruco_panel(aruco_data)

                # Fetch and display AVP frames if auto-update is enabled
                if self.auto_avp_var.get():
                    # Fetch AVP RGB frame
                    avp_data = self.fetch_avp_latest_frame()
                    if avp_data is not None:
                        self._update_avp_rgb_panel(avp_data)

                        # Update AVP status
                        age = avp_data.get('age_seconds', 0)
                        if age < 1.0:
                            self.avp_status_var.set(f"AVP: Connected (age: {age:.2f}s)")
                            self.avp_status_label.config(foreground="green")
                        else:
                            self.avp_status_var.set(f"AVP: Stale (age: {age:.1f}s)")
                            self.avp_status_label.config(foreground="orange")

                    # Fetch AVP ArUco frame
                    avp_aruco_data = self.fetch_avp_aruco_frame()
                    if avp_aruco_data is not None:
                        self._update_avp_aruco_panel(avp_aruco_data)

                # Fetch and cache head pose data
                head_pose_data = self.fetch_head_pose_data()
                self.cached_data['head_pose_data'] = head_pose_data

                # Fetch and cache RS camera pose in AVP frame
                rs_pose_data = self.fetch_rs_pose_in_avp_data()
                self.cached_data['rs_pose_data'] = rs_pose_data

                # Fetch and cache intrinsics data
                intrinsics_data = self.fetch_intrinsics_data()
                if intrinsics_data:
                    self.cached_data['intrinsics_rs'] = intrinsics_data.get('rs')
                    self.cached_data['intrinsics_avp'] = intrinsics_data.get('avp')

                # Fetch and cache transformation data
                transformation_data = self.fetch_transformation_data()
                if transformation_data:
                    self.cached_data['transformation'] = transformation_data

                # Update consolidated data panel with all information
                self._update_consolidated_data_panel()

                # Track timing
                elapsed = (datetime.now() - start_time).total_seconds()
                self._track_frame_time(elapsed)

        except Exception as e:
            logger.error(f"Error updating display: {e}")

    def fetch_status(self) -> Optional[Dict[str, Any]]:
        """
        Fetch system status from /health endpoint.

        Returns:
            dict: Status information including RS connection and calibration state
            None: If request fails
        """
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=3
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Health check returned {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning("Health check request timed out")
            return None
        except Exception as e:
            logger.error(f"Error fetching status: {e}")
            return None

    def fetch_rgbd_frame(self) -> Optional[Dict[str, Any]]:
        """
        Fetch RGBD frame from /get_rgbd_frame endpoint.

        Returns:
            dict: Frame data with 'rgb' and 'depth' base64-encoded images
            None: If request fails
        """
        try:
            response = requests.get(
                f"{self.api_url}/get_rgbd_frame",
                timeout=3
            )
            if response.status_code == 200:
                return response.json()
            else:
                # Don't log warnings if RealSense is not connected (503)
                if response.status_code != 503:
                    logger.warning(f"RGBD frame fetch returned {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning("RGBD frame request timed out")
            return None
        except Exception as e:
            logger.error(f"Error fetching RGBD frame: {e}")
            return None

    def fetch_aruco_frame(self) -> Optional[Dict[str, Any]]:
        """
        Fetch ArUco detection frame from /get_aruco_frame endpoint.

        Returns:
            dict: Frame data with 'rgb' base64-encoded image and detection info
            None: If request fails
        """
        try:
            response = requests.get(
                f"{self.api_url}/get_aruco_frame",
                timeout=3
            )
            if response.status_code == 200:
                return response.json()
            else:
                # Don't log warnings if RealSense is not connected (503)
                if response.status_code != 503:
                    logger.warning(f"ArUco frame fetch returned {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning("ArUco frame request timed out")
            return None
        except Exception as e:
            logger.error(f"Error fetching ArUco frame: {e}")
            return None

    def fetch_avp_latest_frame(self) -> Optional[Dict[str, Any]]:
        """
        Fetch latest AVP frame from /get_avp_latest_frame endpoint.

        Returns:
            dict: Frame data with 'rgb' base64-encoded image, timestamp, and age
            None: If request fails
        """
        try:
            response = requests.get(
                f"{self.api_url}/get_avp_latest_frame",
                timeout=3
            )
            if response.status_code == 200:
                return response.json()
            else:
                if response.status_code != 503:
                    logger.warning(f"AVP frame fetch returned {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning("AVP frame request timed out")
            return None
        except Exception as e:
            logger.error(f"Error fetching AVP frame: {e}")
            return None

    def fetch_avp_aruco_frame(self) -> Optional[Dict[str, Any]]:
        """
        Fetch AVP ArUco detection frame from /get_avp_aruco_frame endpoint.

        Returns:
            dict: Frame data with ArUco detection and intrinsics info
            None: If request fails
        """
        try:
            response = requests.get(
                f"{self.api_url}/get_avp_aruco_frame",
                timeout=3
            )
            if response.status_code == 200:
                return response.json()
            else:
                if response.status_code != 503:
                    logger.warning(f"AVP ArUco frame fetch returned {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning("AVP ArUco frame request timed out")
            return None
        except Exception as e:
            logger.error(f"Error fetching AVP ArUco frame: {e}")
            return None

    def fetch_intrinsics_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch camera intrinsics from /get_intrinsics endpoint.

        Returns:
            dict: Intrinsics data for both RS and AVP cameras
            None: If request fails
        """
        try:
            response = requests.get(
                f"{self.api_url}/get_intrinsics",
                timeout=3
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Intrinsics fetch returned {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning("Intrinsics request timed out")
            return None
        except Exception as e:
            logger.error(f"Error fetching intrinsics: {e}")
            return None

    def fetch_transformation_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch coordinate transformation from /get_transformation endpoint.

        Returns:
            dict: Transformation matrix data
            None: If request fails
        """
        try:
            response = requests.get(
                f"{self.api_url}/get_transformation",
                timeout=3
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Transformation fetch returned {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning("Transformation request timed out")
            return None
        except Exception as e:
            logger.error(f"Error fetching transformation: {e}")
            return None

    def fetch_head_pose_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch head pose data from /get_head_pose endpoint.

        Returns:
            dict: Head pose data with position, quaternion, timestamp
            None: If request fails
        """
        try:
            response = requests.get(
                f"{self.api_url}/get_head_pose",
                timeout=3
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                # No head pose data available yet
                return None
            else:
                logger.warning(f"Head pose fetch returned {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning("Head pose request timed out")
            return None
        except Exception as e:
            logger.error(f"Error fetching head pose: {e}")
            return None

    def fetch_rs_pose_in_avp_data(self) -> Optional[Dict[str, Any]]:
        """
        Fetch RS camera pose in AVP frame from /get_rs_pose_in_avp endpoint.

        Returns:
            dict: RS camera pose data with position, quaternion, calibration status
            None: If request fails
        """
        try:
            response = requests.get(
                f"{self.api_url}/get_rs_pose_in_avp",
                timeout=3
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"RS pose in AVP fetch returned {response.status_code}")
                return None

        except requests.exceptions.Timeout:
            logger.warning("RS pose in AVP request timed out")
            return None
        except Exception as e:
            logger.error(f"Error fetching RS pose in AVP: {e}")
            return None

    def _update_consolidated_data_panel(self):
        """Update the consolidated data panel with all system information."""
        text_content = ""

        # === SYSTEM STATUS ===
        text_content += "=" * 80 + "\n"
        text_content += "SYSTEM STATUS\n"
        text_content += "=" * 80 + "\n\n"

        status = self.cached_data.get('status', {})
        rs_connected = status.get('rs_connected', False)
        calibrated = status.get('calibrated', False)

        text_content += f"RealSense Connected:  {'✓ Yes' if rs_connected else '✗ No'}\n"
        text_content += f"System Calibrated:    {'✓ Yes' if calibrated else '✗ No'}\n"

        last_update = self.cached_data.get('last_fetch_time')
        if last_update:
            text_content += f"Last Update:          {last_update.strftime('%H:%M:%S.%f')[:-3]}\n"

        if len(self.stats['frame_times']) > 0:
            avg_time = self.stats['average_frame_time']
            if avg_time > 0:
                fps = 1.0 / avg_time
                text_content += f"Update Rate:          {fps:.1f} Hz\n"

        # ArUco detection
        aruco_markers = self.cached_data.get('aruco_markers', 0)
        aruco_ids = self.cached_data.get('aruco_ids', [])
        text_content += f"\nArUco Markers Detected: {aruco_markers}\n"
        if aruco_ids:
            ids_str = ', '.join(map(str, aruco_ids))
            text_content += f"Marker IDs: {ids_str}\n"

        # === HEAD POSE (VisionOS) ===
        text_content += "\n" + "=" * 80 + "\n"
        text_content += "HEAD POSE (VisionOS)\n"
        text_content += "=" * 80 + "\n\n"

        head_pose_data = self.cached_data.get('head_pose_data')
        if head_pose_data is None:
            text_content += "No head pose data available - waiting for VisionOS device to connect...\n"
        else:
            position = head_pose_data.get('position', [0, 0, 0])
            quaternion = head_pose_data.get('quaternion', [0, 0, 0, 1])
            age = head_pose_data.get('age_seconds', 0)
            reception_count = head_pose_data.get('reception_count', 0)
            reception_rate = head_pose_data.get('reception_rate', None)

            text_content += f"Position (m):         X={position[0]:>7.3f}  Y={position[1]:>7.3f}  Z={position[2]:>7.3f}\n"
            text_content += f"Quaternion:           X={quaternion[0]:>7.3f}  Y={quaternion[1]:>7.3f}  Z={quaternion[2]:>7.3f}  W={quaternion[3]:>7.3f}\n"

            if age < 0.5:
                status_text = "LIVE ✓"
            elif age < 2.0:
                status_text = "RECENT"
            else:
                status_text = "STALE ⚠"

            text_content += f"Status:               {status_text} (age: {age:.2f}s)\n"
            text_content += f"Reception Count:      {reception_count}\n"
            if reception_rate is not None:
                text_content += f"Reception Rate:       {reception_rate:.1f} Hz\n"

        # === RS CAMERA POSE IN AVP FRAME ===
        text_content += "\n" + "=" * 80 + "\n"
        text_content += "RS CAMERA POSE IN AVP FRAME\n"
        text_content += "=" * 80 + "\n\n"

        rs_pose_data = self.cached_data.get('rs_pose_data')
        if rs_pose_data is None or not rs_pose_data.get('calibrated', False):
            text_content += "Not calibrated\n\n"
            text_content += "Requirements:\n"
            text_content += "  • Perform RS calibration (detect ArUco board with RealSense)\n"
            text_content += "  • Perform AVP calibration (detect ArUco board with Vision Pro)\n"
            text_content += "  • Both cameras must see the same board\n"
        else:
            position = rs_pose_data.get('position', [0, 0, 0])
            quaternion = rs_pose_data.get('quaternion', [0, 0, 0, 1])
            head_pose_age = rs_pose_data.get('head_pose_age', None)

            text_content += f"Position (m):         X={position[0]:>7.3f}  Y={position[1]:>7.3f}  Z={position[2]:>7.3f}\n"
            text_content += f"Quaternion:           X={quaternion[0]:>7.3f}  Y={quaternion[1]:>7.3f}  Z={quaternion[2]:>7.3f}  W={quaternion[3]:>7.3f}\n"
            text_content += f"Calibrated:           ✓ YES\n"

            if head_pose_age is not None:
                if head_pose_age < 0.5:
                    text_content += f"Head Pose:            LIVE ({head_pose_age:.2f}s)\n"
                elif head_pose_age < 2.0:
                    text_content += f"Head Pose:            Recent ({head_pose_age:.2f}s)\n"
                else:
                    text_content += f"Head Pose:            Stale ({head_pose_age:.1f}s) ⚠\n"

            text_content += "\nThis shows where the RealSense camera is located in your VisionOS headset view.\n"

        # === CAMERA INTRINSICS ===
        text_content += "\n" + "=" * 80 + "\n"
        text_content += "CAMERA INTRINSICS\n"
        text_content += "=" * 80 + "\n\n"

        intrinsics_rs = self.cached_data.get('intrinsics_rs')
        intrinsics_avp = self.cached_data.get('intrinsics_avp')

        text_content += "[RealSense]\n"
        if intrinsics_rs and intrinsics_rs.get('calculated', False):
            K = intrinsics_rs.get('K')
            if K is not None:
                K_arr = np.array(K)
                text_content += self._format_matrix(K_arr, precision=2)
                text_content += f"Method: {intrinsics_rs.get('method', 'N/A')}\n"
        else:
            text_content += "Not calculated yet\n"

        text_content += "\n[AVP (Vision Pro)]\n"
        if intrinsics_avp and intrinsics_avp.get('calculated', False):
            K = intrinsics_avp.get('K')
            if K is not None:
                K_arr = np.array(K)
                text_content += self._format_matrix(K_arr, precision=2)
                text_content += f"Method: {intrinsics_avp.get('method', 'N/A')}\n"
        else:
            text_content += "Not calculated yet\n"

        # === COORDINATE TRANSFORMATION ===
        text_content += "\n" + "=" * 80 + "\n"
        text_content += "COORDINATE TRANSFORMATION\n"
        text_content += "=" * 80 + "\n\n"

        transformation = self.cached_data.get('transformation')
        if transformation and transformation.get('calibrated', False):
            text_content += "[T_avp_rs] - Transform from RS to AVP frame\n\n"
            T_avp_rs = transformation.get('T_avp_rs')
            if T_avp_rs is not None:
                T_arr = np.array(T_avp_rs)
                if T_arr.shape == (4, 4):
                    text_content += self._format_matrix(T_arr, precision=4)

            timestamp = transformation.get('timestamp')
            if timestamp:
                dt = datetime.fromtimestamp(timestamp)
                text_content += f"\nCalculated: {dt.strftime('%H:%M:%S')}\n"
        else:
            text_content += "Not calibrated yet\n\n"
            text_content += "Requirements:\n"
            text_content += "  • Both cameras must detect the same ArUco board\n"
            text_content += "  • Intrinsics must be calculated for both\n"

        # === STATISTICS ===
        text_content += "\n" + "=" * 80 + "\n"
        text_content += "STATISTICS\n"
        text_content += "=" * 80 + "\n\n"

        text_content += f"Total Updates:        {self.stats['total_frames']}\n"
        text_content += f"Successful:           {self.stats['successful_estimates']}\n"
        text_content += f"Failed:               {self.stats['failed_estimates']}\n"

        if self.stats['total_frames'] > 0:
            success_rate = (self.stats['successful_estimates'] / self.stats['total_frames'] * 100)
            text_content += f"Success Rate:         {success_rate:.1f}%\n"

        if len(self.stats['frame_times']) > 0:
            avg_time = self.stats['average_frame_time']
            text_content += f"\nAvg Frame Time:       {avg_time*1000:.2f} ms\n"
            if avg_time > 0:
                fps = 1.0 / avg_time
                text_content += f"Estimated Rate:       {fps:.1f} Hz\n"

        # Update the consolidated text widget
        self.data_text.config(state=tk.NORMAL)
        self.data_text.delete('1.0', tk.END)
        self.data_text.insert('1.0', text_content)
        self.data_text.config(state=tk.DISABLED)

    def _update_rgbd_panels(self, rgbd_data: Dict[str, Any]):
        """
        Update RGB and Depth image panels.

        Args:
            rgbd_data (dict): Dictionary containing 'rgb' and 'depth' base64-encoded images
        """
        try:
            # Decode and display RGB image
            if 'rgb' in rgbd_data:
                rgb_b64 = rgbd_data['rgb']
                # Remove data URL prefix if present
                if ',' in rgb_b64:
                    rgb_b64 = rgb_b64.split(',')[1]

                # Decode base64 to image
                rgb_bytes = base64.b64decode(rgb_b64)
                rgb_np = np.frombuffer(rgb_bytes, np.uint8)
                rgb_image = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)

                if rgb_image is not None:
                    # Convert BGR to RGB for display
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                    self.display_image(self.panel_rgb, rgb_image)

            # Decode and display Depth colormap
            if 'depth' in rgbd_data:
                depth_b64 = rgbd_data['depth']
                # Remove data URL prefix if present
                if ',' in depth_b64:
                    depth_b64 = depth_b64.split(',')[1]

                # Decode base64 to image
                depth_bytes = base64.b64decode(depth_b64)
                depth_np = np.frombuffer(depth_bytes, np.uint8)
                depth_image = cv2.imdecode(depth_np, cv2.IMREAD_COLOR)

                if depth_image is not None:
                    # Convert BGR to RGB for display
                    depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)
                    self.display_image(self.panel_depth, depth_image)

        except Exception as e:
            logger.error(f"Error updating RGBD panels: {e}")

    def _update_aruco_panel(self, aruco_data: Dict[str, Any]):
        """
        Update the ArUco detection image panel.

        Args:
            aruco_data (dict): Dictionary containing 'rgb' base64-encoded image
                              and detection info (markers_detected, marker_ids)
        """
        try:
            # Decode and display ArUco annotated image
            if 'rgb' in aruco_data:
                rgb_b64 = aruco_data['rgb']
                # Remove data URL prefix if present
                if ',' in rgb_b64:
                    rgb_b64 = rgb_b64.split(',')[1]

                # Decode base64 to image
                rgb_bytes = base64.b64decode(rgb_b64)
                rgb_np = np.frombuffer(rgb_bytes, np.uint8)
                aruco_image = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)

                if aruco_image is not None:
                    # Convert BGR to RGB for display
                    aruco_image = cv2.cvtColor(aruco_image, cv2.COLOR_BGR2RGB)
                    self.display_image(self.panel_aruco, aruco_image)

                    # Cache detection info
                    self.cached_data['aruco_markers'] = aruco_data.get('markers_detected', 0)
                    self.cached_data['aruco_ids'] = aruco_data.get('marker_ids', [])

        except Exception as e:
            logger.error(f"Error updating ArUco panel: {e}")

    def _update_avp_rgb_panel(self, avp_data: Dict[str, Any]):
        """
        Update the AVP RGB image panel.

        Args:
            avp_data (dict): Dictionary containing 'rgb' base64-encoded image
        """
        try:
            if 'rgb' in avp_data:
                rgb_b64 = avp_data['rgb']
                # Remove data URL prefix if present
                if ',' in rgb_b64:
                    rgb_b64 = rgb_b64.split(',')[1]

                # Decode base64 to image
                rgb_bytes = base64.b64decode(rgb_b64)
                rgb_np = np.frombuffer(rgb_bytes, np.uint8)
                rgb_image = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)

                if rgb_image is not None:
                    # Convert BGR to RGB for display
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                    self.display_image(self.panel_avp_rgb, rgb_image)

                    # Cache AVP frame info
                    self.cached_data['avp_rgb_image'] = rgb_image
                    self.cached_data['avp_timestamp'] = avp_data.get('timestamp')
                    self.cached_data['avp_age'] = avp_data.get('age_seconds')

        except Exception as e:
            logger.error(f"Error updating AVP RGB panel: {e}")

    def _update_avp_aruco_panel(self, avp_aruco_data: Dict[str, Any]):
        """
        Update the AVP ArUco detection image panel.

        Args:
            avp_aruco_data (dict): Dictionary containing ArUco detection data
        """
        try:
            if 'rgb' in avp_aruco_data:
                rgb_b64 = avp_aruco_data['rgb']
                # Remove data URL prefix if present
                if ',' in rgb_b64:
                    rgb_b64 = rgb_b64.split(',')[1]

                # Decode base64 to image
                rgb_bytes = base64.b64decode(rgb_b64)
                rgb_np = np.frombuffer(rgb_bytes, np.uint8)
                aruco_image = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)

                if aruco_image is not None:
                    # Convert BGR to RGB for display
                    aruco_image = cv2.cvtColor(aruco_image, cv2.COLOR_BGR2RGB)
                    self.display_image(self.panel_avp_aruco, aruco_image)

                    # Cache AVP ArUco image
                    self.cached_data['avp_aruco_image'] = aruco_image

        except Exception as e:
            logger.error(f"Error updating AVP ArUco panel: {e}")

    def _update_intrinsics_panel(self, intrinsics_data: Dict[str, Any]):
        """
        Update the Camera Intrinsics text panel.

        Args:
            intrinsics_data (dict): Dictionary containing RS and AVP intrinsics
        """
        text_content = "Camera Intrinsics\n" + "=" * 40 + "\n\n"

        # RealSense intrinsics
        if 'rs' in intrinsics_data:
            rs_data = intrinsics_data['rs']
            text_content += "[RealSense]\n"

            if rs_data.get('calculated', False):
                K = rs_data.get('K')
                if K is not None:
                    K_arr = np.array(K)
                    text_content += self._format_matrix(K_arr, precision=2)
                    text_content += f"Method: {rs_data.get('method', 'N/A')}\n"

                    timestamp = rs_data.get('timestamp')
                    if timestamp:
                        dt = datetime.fromtimestamp(timestamp)
                        text_content += f"Calculated: {dt.strftime('%H:%M:%S')}\n"
            else:
                text_content += "Not calculated yet\n"

        text_content += "\n"

        # AVP intrinsics
        if 'avp' in intrinsics_data:
            avp_data = intrinsics_data['avp']
            text_content += "[AVP]\n"

            if avp_data.get('calculated', False):
                K = avp_data.get('K')
                if K is not None:
                    K_arr = np.array(K)
                    text_content += self._format_matrix(K_arr, precision=2)
                    text_content += f"Method: {avp_data.get('method', 'N/A')}\n"

                    timestamp = avp_data.get('timestamp')
                    if timestamp:
                        dt = datetime.fromtimestamp(timestamp)
                        text_content += f"Calculated: {dt.strftime('%H:%M:%S')}\n"
            else:
                text_content += "Not calculated yet\n"

        self._update_text_widget(self.panel_intrinsics['text'], text_content)

    def _update_transformation_panel(self, transformation_data: Dict[str, Any]):
        """
        Update the Coordinate Transformation text panel.

        Args:
            transformation_data (dict): Dictionary containing transformation matrix
        """
        text_content = "Coordinate Transformation\n" + "=" * 40 + "\n\n"

        if transformation_data.get('calibrated', False):
            text_content += "[T_avp_rs]\n"
            text_content += "Transform from RS to AVP frame\n\n"

            T_avp_rs = transformation_data.get('T_avp_rs')
            if T_avp_rs is not None:
                T_arr = np.array(T_avp_rs)
                if T_arr.shape == (4, 4):
                    text_content += self._format_matrix(T_arr, precision=4)
                else:
                    text_content += "Invalid matrix shape\n"

            timestamp = transformation_data.get('timestamp')
            if timestamp:
                dt = datetime.fromtimestamp(timestamp)
                text_content += f"\nCalculated: {dt.strftime('%H:%M:%S')}\n"
        else:
            text_content += "Not calibrated yet\n\n"
            text_content += "Requirements:\n"
            text_content += "- Both cameras must detect\n"
            text_content += "  the same ArUco board\n"
            text_content += "- Intrinsics must be calculated\n"

        self._update_text_widget(self.panel_transformation['text'], text_content)

    def _update_head_pose_panel(self, head_pose_data: Optional[Dict[str, Any]]):
        """
        Update the Head Pose text panel.

        Args:
            head_pose_data (dict): Dictionary containing head pose data from VisionOS
        """
        text_content = "Head Pose (VisionOS)\n" + "=" * 40 + "\n\n"

        if head_pose_data is None:
            text_content += "No head pose data available\n\n"
            text_content += "Waiting for VisionOS device\n"
            text_content += "to stream head pose...\n"
        else:
            position = head_pose_data.get('position', [0, 0, 0])
            quaternion = head_pose_data.get('quaternion', [0, 0, 0, 1])
            age = head_pose_data.get('age_seconds', 0)
            reception_count = head_pose_data.get('reception_count', 0)
            reception_rate = head_pose_data.get('reception_rate', None)

            # Position
            text_content += "[Position] (meters)\n"
            text_content += f"  X: {position[0]:>7.3f} m\n"
            text_content += f"  Y: {position[1]:>7.3f} m\n"
            text_content += f"  Z: {position[2]:>7.3f} m\n\n"

            # Quaternion
            text_content += "[Orientation] (quaternion)\n"
            text_content += f"  X: {quaternion[0]:>7.3f}\n"
            text_content += f"  Y: {quaternion[1]:>7.3f}\n"
            text_content += f"  Z: {quaternion[2]:>7.3f}\n"
            text_content += f"  W: {quaternion[3]:>7.3f}\n\n"

            # Status
            text_content += "[Status]\n"
            if age < 0.5:
                status_text = "LIVE"
                text_content += f"  Status: {status_text} ✓\n"
            elif age < 2.0:
                status_text = "RECENT"
                text_content += f"  Status: {status_text}\n"
            else:
                status_text = "STALE"
                text_content += f"  Status: {status_text} ⚠\n"

            text_content += f"  Age: {age:.2f}s\n"
            text_content += f"  Count: {reception_count}\n"
            if reception_rate is not None:
                text_content += f"  Rate: {reception_rate:.1f} Hz\n"

        self._update_text_widget(self.panel_head_pose['text'], text_content)

    def _update_rs_pose_panel(self, rs_pose_data: Optional[Dict[str, Any]]):
        """
        Update the RS Camera in AVP Frame text panel.

        Args:
            rs_pose_data (dict): Dictionary containing RS camera pose in AVP frame
        """
        text_content = "RS Camera in AVP Frame\n" + "=" * 40 + "\n\n"

        if rs_pose_data is None or not rs_pose_data.get('calibrated', False):
            text_content += "Not calibrated\n\n"
            text_content += "Requirements:\n"
            text_content += "- Perform RS calibration\n"
            text_content += "- Perform AVP calibration\n"
            text_content += "- Both must see ArUco board\n"
            if rs_pose_data and 'message' in rs_pose_data:
                text_content += f"\n{rs_pose_data['message']}\n"
        else:
            position = rs_pose_data.get('position', [0, 0, 0])
            quaternion = rs_pose_data.get('quaternion', [0, 0, 0, 1])
            head_pose_age = rs_pose_data.get('head_pose_age', None)

            # Position
            text_content += "[RS Camera Position]\n"
            text_content += f"  X: {position[0]:>7.3f} m\n"
            text_content += f"  Y: {position[1]:>7.3f} m\n"
            text_content += f"  Z: {position[2]:>7.3f} m\n\n"

            # Orientation
            text_content += "[RS Camera Orientation]\n"
            text_content += f"  X: {quaternion[0]:>7.3f}\n"
            text_content += f"  Y: {quaternion[1]:>7.3f}\n"
            text_content += f"  Z: {quaternion[2]:>7.3f}\n"
            text_content += f"  W: {quaternion[3]:>7.3f}\n\n"

            # Status
            text_content += "[Status]\n"
            text_content += "  Calibrated: YES ✓\n"
            if head_pose_age is not None:
                if head_pose_age < 0.5:
                    text_content += f"  Head pose: LIVE ({head_pose_age:.2f}s)\n"
                elif head_pose_age < 2.0:
                    text_content += f"  Head pose: Recent ({head_pose_age:.2f}s)\n"
                else:
                    text_content += f"  Head pose: Stale ({head_pose_age:.1f}s) ⚠\n"

            # Note
            text_content += "\nThis shows where the RS camera\n"
            text_content += "is located in your headset view.\n"

        self._update_text_widget(self.panel_rs_pose['text'], text_content)

    def _update_poses_panel(self, poses_data: Dict[str, Any]):
        """
        Update the Poses text panel.

        Args:
            poses_data (dict): Dictionary containing pose matrices
        """
        text_content = "Latest Poses\n" + "=" * 40 + "\n\n"

        if not poses_data:
            text_content += "No pose data available\n"
        else:
            # RS Pose
            if 'pose_rs_in_avp' in poses_data:
                pose = poses_data['pose_rs_in_avp']
                text_content += "[RS Camera Pose in AVP]\n"
                if isinstance(pose, (list, np.ndarray)):
                    pose_arr = np.array(pose)
                    if pose_arr.shape == (4, 4):
                        text_content += self._format_matrix(pose_arr, precision=4)
                    else:
                        text_content += str(pose)[:100] + "\n"
                else:
                    text_content += "Invalid pose format\n"

                text_content += "\n"

            # Object Pose
            if 'pose_object_in_avp' in poses_data:
                pose = poses_data['pose_object_in_avp']
                text_content += "[Object Pose in AVP]\n"
                if isinstance(pose, (list, np.ndarray)):
                    pose_arr = np.array(pose)
                    if pose_arr.shape == (4, 4):
                        text_content += self._format_matrix(pose_arr, precision=4)
                    else:
                        text_content += str(pose)[:100] + "\n"
                else:
                    text_content += "Invalid pose format\n"

        self._update_text_widget(self.panel_poses['text'], text_content)

    def _update_stats_panel(self):
        """Update the Statistics text panel."""
        text_content = "Statistics\n" + "=" * 40 + "\n\n"

        text_content += f"Total Updates: {self.stats['total_frames']}\n"
        text_content += f"Successful: {self.stats['successful_estimates']}\n"
        text_content += f"Failed: {self.stats['failed_estimates']}\n"

        if self.stats['total_frames'] > 0:
            success_rate = (
                self.stats['successful_estimates'] / self.stats['total_frames'] * 100
            )
            text_content += f"\nSuccess Rate: {success_rate:.1f}%\n"

        if len(self.stats['frame_times']) > 0:
            avg_time = self.stats['average_frame_time']
            text_content += f"\nAvg Frame Time: {avg_time*1000:.2f} ms\n"

            if avg_time > 0:
                fps = 1.0 / avg_time
                text_content += f"Estimated Rate: {fps:.1f} Hz\n"

        text_content += "\n[Frame Timing]\n"
        if len(self.stats['frame_times']) > 0:
            times = self.stats['frame_times'][-5:]  # Last 5
            for i, t in enumerate(times):
                text_content += f"  Frame {i}: {t*1000:.2f} ms\n"

        self._update_text_widget(self.panel_stats['text'], text_content)

    def _track_frame_time(self, elapsed_seconds: float):
        """
        Track frame timing statistics.

        Args:
            elapsed_seconds (float): Time taken for last frame update
        """
        self.stats['frame_times'].append(elapsed_seconds)

        # Keep only last 100 frames
        if len(self.stats['frame_times']) > 100:
            self.stats['frame_times'] = self.stats['frame_times'][-100:]

        # Calculate average
        self.stats['average_frame_time'] = (
            sum(self.stats['frame_times']) / len(self.stats['frame_times'])
        )

        self.stats['last_update_time'] = datetime.now()
        # Stats are now part of consolidated panel, no separate update needed

    def _update_status_text(self, panel: str, message: str):
        """
        Update a specific status text panel.

        Args:
            panel (str): Panel name ('Status', 'Poses', or 'Stats')
            message (str): Message to display
        """
        if panel == "Status":
            text_content = f"{message}\n"
            self._update_text_widget(self.panel_status['text'], text_content)

    def _update_text_widget(self, text_widget: tk.Text, content: str):
        """
        Update text widget content safely.

        Args:
            text_widget (tk.Text): Text widget to update
            content (str): New content
        """
        text_widget.config(state=tk.NORMAL)
        text_widget.delete('1.0', tk.END)
        text_widget.insert('1.0', content)
        text_widget.config(state=tk.DISABLED)

    def display_image(self, panel: Dict, image_array: np.ndarray,
                     resize: bool = True):
        """
        Display an image on a canvas panel with dynamic resizing.

        Args:
            panel (dict): Panel dictionary from _create_image_panel
            image_array (np.ndarray): Image as numpy array (RGB or grayscale)
            resize (bool): Whether to resize to panel dimensions. Default: True
        """
        if image_array is None:
            return

        try:
            # Convert to RGB if needed
            if len(image_array.shape) == 2:
                # Grayscale - convert to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                # RGBA - convert to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            elif image_array.shape[2] == 3 and image_array.dtype == np.uint8:
                # BGR to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            canvas = panel['canvas']

            # Get actual canvas size
            canvas.update_idletasks()  # Force update to get accurate size
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            # Use minimum size if canvas hasn't been drawn yet
            if canvas_width < 10:
                canvas_width = 320
            if canvas_height < 10:
                canvas_height = 240

            # Resize if requested - maintain aspect ratio
            if resize:
                img_h, img_w = image_array.shape[:2]

                # Calculate aspect ratios
                img_aspect = img_w / img_h
                canvas_aspect = canvas_width / canvas_height

                # Resize to fit canvas while maintaining aspect ratio
                if img_aspect > canvas_aspect:
                    # Image is wider - fit to width
                    new_w = canvas_width
                    new_h = int(canvas_width / img_aspect)
                else:
                    # Image is taller - fit to height
                    new_h = canvas_height
                    new_w = int(canvas_height * img_aspect)

                image_array = cv2.resize(image_array, (new_w, new_h))

            # Convert to PIL Image
            pil_image = Image.fromarray(image_array.astype(np.uint8))

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(pil_image)

            # Update canvas - center the image
            canvas.delete('all')
            canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo)
            panel['photo'] = photo  # Keep reference to prevent garbage collection

            # Force canvas to update/redraw
            canvas.update_idletasks()
            canvas.update()  # Force immediate redraw

        except Exception as e:
            logger.error(f"Error displaying image on {panel['title']}: {e}")

    def _update_status_indicator(self, connected: bool):
        """
        Update the connection status indicator.

        Args:
            connected (bool): Connection state
        """
        if connected:
            self.status_indicator.config(text="● Connected", foreground="green")
        else:
            self.status_indicator.config(text="● Disconnected", foreground="red")

    def _format_matrix(self, matrix: np.ndarray, precision: int = 3) -> str:
        """
        Format a matrix for display.

        Args:
            matrix (np.ndarray): Matrix to format
            precision (int): Decimal places. Default: 3

        Returns:
            str: Formatted matrix string
        """
        lines = []
        for row in matrix:
            row_str = "  ".join([f"{val:.{precision}f}" for val in row])
            lines.append(f"[ {row_str} ]\n")
        return "".join(lines)

    def clear_statistics(self):
        """Clear all statistics counters."""
        self.stats = {
            'total_frames': 0,
            'successful_estimates': 0,
            'failed_estimates': 0,
            'last_update_time': None,
            'average_frame_time': 0.0,
            'frame_times': []
        }
        self._update_consolidated_data_panel()
        logger.info("Statistics cleared")

    def _capture_for_aruco(self):
        """Capture frame from UxPlay for ArUco calibration."""
        self.capture_status_var.set("Capturing for ArUco...")
        self.capture_status_label.config(foreground="blue")
        logger.info("Capturing frame for ArUco calibration")

        # Run capture in background thread to avoid blocking GUI
        def capture_thread():
            try:
                response = requests.post(
                    f"{self.api_url}/capture_frame?purpose=aruco_calibration",
                    timeout=5
                )

                if response.status_code == 200:
                    self.capture_status_var.set("✓ ArUco frame captured")
                    self.capture_status_label.config(foreground="green")
                    logger.info("ArUco frame captured successfully")
                else:
                    error_msg = response.json().get('error', 'Unknown error')
                    self.capture_status_var.set(f"✗ Capture failed: {error_msg}")
                    self.capture_status_label.config(foreground="red")
                    logger.error(f"ArUco capture failed: {error_msg}")

            except requests.exceptions.Timeout:
                self.capture_status_var.set("✗ Capture timeout")
                self.capture_status_label.config(foreground="red")
                logger.error("ArUco capture timed out")

            except requests.exceptions.ConnectionError:
                self.capture_status_var.set("✗ Connection error")
                self.capture_status_label.config(foreground="red")
                logger.error("ArUco capture connection error")

            except Exception as e:
                self.capture_status_var.set(f"✗ Error: {str(e)[:30]}")
                self.capture_status_label.config(foreground="red")
                logger.error(f"ArUco capture error: {e}")

        threading.Thread(target=capture_thread, daemon=True).start()

    def _capture_for_roi(self):
        """Capture frame from UxPlay for ROI selection."""
        self.capture_status_var.set("Capturing for ROI...")
        self.capture_status_label.config(foreground="blue")
        logger.info("Capturing frame for ROI selection")

        # Run capture in background thread to avoid blocking GUI
        def capture_thread():
            try:
                response = requests.post(
                    f"{self.api_url}/capture_frame?purpose=roi_selection",
                    timeout=5
                )

                if response.status_code == 200:
                    self.capture_status_var.set("✓ ROI frame captured")
                    self.capture_status_label.config(foreground="green")
                    logger.info("ROI frame captured successfully")
                else:
                    error_msg = response.json().get('error', 'Unknown error')
                    self.capture_status_var.set(f"✗ Capture failed: {error_msg}")
                    self.capture_status_label.config(foreground="red")
                    logger.error(f"ROI capture failed: {error_msg}")

            except requests.exceptions.Timeout:
                self.capture_status_var.set("✗ Capture timeout")
                self.capture_status_label.config(foreground="red")
                logger.error("ROI capture timed out")

            except requests.exceptions.ConnectionError:
                self.capture_status_var.set("✗ Connection error")
                self.capture_status_label.config(foreground="red")
                logger.error("ROI capture connection error")

            except Exception as e:
                self.capture_status_var.set(f"✗ Error: {str(e)[:30]}")
                self.capture_status_label.config(foreground="red")
                logger.error(f"ROI capture error: {e}")

        threading.Thread(target=capture_thread, daemon=True).start()

    def _fetch_avp_frame(self):
        """Fetch and display the latest AVP frame."""
        logger.info("Fetching AVP frame")

        # Fetch AVP RGB frame
        avp_data = self.fetch_avp_latest_frame()
        if avp_data is not None:
            self._update_avp_rgb_panel(avp_data)

            # Update AVP status label
            age = avp_data.get('age_seconds', 0)
            if age < 1.0:
                self.avp_status_var.set(f"AVP: Connected (age: {age:.2f}s)")
                self.avp_status_label.config(foreground="green")
            else:
                self.avp_status_var.set(f"AVP: Stale frame (age: {age:.1f}s)")
                self.avp_status_label.config(foreground="orange")

            logger.info(f"AVP frame displayed (age: {age:.2f}s)")
        else:
            self.avp_status_var.set("AVP: No frame available")
            self.avp_status_label.config(foreground="red")
            logger.warning("Failed to fetch AVP frame")

        # Fetch AVP ArUco frame
        avp_aruco_data = self.fetch_avp_aruco_frame()
        if avp_aruco_data is not None:
            self._update_avp_aruco_panel(avp_aruco_data)

            # Check if intrinsics were calculated
            if avp_aruco_data.get('intrinsics_calculated', False):
                logger.info("AVP intrinsics calculated!")

    def _fetch_intrinsics(self):
        """Fetch and display camera intrinsics."""
        logger.info("Fetching camera intrinsics")

        intrinsics_data = self.fetch_intrinsics_data()
        if intrinsics_data is not None:
            # Cache intrinsics
            self.cached_data['intrinsics_rs'] = intrinsics_data.get('rs')
            self.cached_data['intrinsics_avp'] = intrinsics_data.get('avp')

            # Update consolidated panel
            self._update_consolidated_data_panel()

            logger.info("Intrinsics data displayed")
        else:
            logger.warning("Failed to fetch intrinsics")

    def _fetch_transformation(self):
        """Fetch and display coordinate transformation."""
        logger.info("Fetching coordinate transformation")

        transformation_data = self.fetch_transformation_data()
        if transformation_data is not None:
            # Cache transformation
            self.cached_data['transformation'] = transformation_data

            # Update consolidated panel
            self._update_consolidated_data_panel()

            if transformation_data.get('calibrated', False):
                logger.info("Transformation matrix displayed")
            else:
                logger.warning("Transformation not calibrated yet")
        else:
            logger.warning("Failed to fetch transformation")

    def on_closing(self):
        """Handle window close event."""
        logger.info("Debug Viewer closing...")
        self.disconnect()
        self.should_stop = True

        # Wait briefly for thread to exit
        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=2)

        self.root.destroy()

    def run(self):
        """Start the debug viewer application."""
        logger.info("Starting Debug Viewer GUI")
        self.root.mainloop()


def main():
    """Main entry point for the debug viewer application."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Debug Viewer for Pose Estimation Pipeline"
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help=f"API URL (default: http://{API_HOST}:{API_PORT})"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Window width in pixels (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Window height in pixels (default: 800)"
    )

    args = parser.parse_args()

    # Create and run viewer
    viewer = DebugViewer(
        api_url=args.api_url,
        width=args.width,
        height=args.height
    )
    viewer.run()


if __name__ == "__main__":
    main()
