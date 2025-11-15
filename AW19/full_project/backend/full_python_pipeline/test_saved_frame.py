#!/usr/bin/env python3
"""
Saved Frame Viewer
Load and view saved pose request frames with overlay visualization
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import numpy as np
import cv2 as cv
from PIL import Image, ImageTk
import trimesh


class SavedFrameViewer:
    def __init__(self, root):
        self.root = root
        root.title("Saved Frame Viewer")
        root.geometry("1400x900")

        # State
        self.current_frame_dir = None
        self.photo_refs = {}
        self.frame_data = {}

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Build the complete UI layout"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)

        # Top: Controls panel
        controls_frame = ttk.LabelFrame(main_container, text="Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        self._build_controls(controls_frame)

        # Middle: Image display (2 rows x 3 columns)
        images_frame = ttk.Frame(main_container)
        images_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self._build_image_grid(images_frame)

        # Bottom: Info panel
        info_frame = ttk.LabelFrame(main_container, text="Frame Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        self._build_info_panel(info_frame)

    def _build_controls(self, parent):
        """Build the controls panel"""
        # Select folder button
        ttk.Button(parent, text="Select Frame Folder", command=self.select_frame_folder).pack(side=tk.LEFT, padx=5)

        # Current folder label
        self.folder_label = ttk.Label(parent, text="No folder selected", foreground="gray")
        self.folder_label.pack(side=tk.LEFT, padx=20)

        # Reload button
        self.reload_btn = ttk.Button(parent, text="Reload", command=self.load_frame_data, state=tk.DISABLED)
        self.reload_btn.pack(side=tk.LEFT, padx=5)

    def _build_image_grid(self, parent):
        """Build the image grid (2 rows x 3 columns)"""
        self.image_labels = {}
        image_titles = [
            ("RGB", "rgb"),
            ("Depth", "depth"),
            ("Mask", "mask"),
            ("Mesh (2D projection)", "mesh"),
            ("RGB with Pose Overlay", "pose_overlay"),
            ("Info", "info")
        ]

        for idx, (title, key) in enumerate(image_titles):
            row = idx // 3
            col = idx % 3

            frame = ttk.LabelFrame(parent, text=title, padding=5)
            frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)

            if key == "info":
                # Info text widget
                text_widget = tk.Text(frame, height=15, wrap=tk.WORD, font=("Courier", 9))
                scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
                text_widget.configure(yscrollcommand=scrollbar.set)
                text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                self.image_labels[key] = text_widget
            else:
                label = ttk.Label(frame, text="No data", anchor="center")
                label.pack(fill=tk.BOTH, expand=True)
                self.image_labels[key] = label

            parent.rowconfigure(row, weight=1)
            parent.columnconfigure(col, weight=1)

    def _build_info_panel(self, parent):
        """Build the info panel"""
        self.info_text = tk.Text(parent, height=8, wrap=tk.WORD, font=("Courier", 9))
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)

        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def select_frame_folder(self):
        """Select a frame folder to load"""
        # Start in saved_pose_requests if it exists
        initial_dir = "saved_pose_requests" if os.path.exists("saved_pose_requests") else "."

        folder_path = filedialog.askdirectory(
            title="Select Frame Folder",
            initialdir=initial_dir
        )

        if folder_path:
            self.current_frame_dir = folder_path
            self.folder_label.config(text=os.path.basename(folder_path), foreground="blue")
            self.reload_btn.config(state=tk.NORMAL)
            self.load_frame_data()

    def load_frame_data(self):
        """Load all data from the selected frame folder"""
        if not self.current_frame_dir:
            return

        try:
            self.frame_data = {}

            # Load RGB
            rgb_path = os.path.join(self.current_frame_dir, "rgb.png")
            if os.path.exists(rgb_path):
                rgb = cv.imread(rgb_path)
                self.frame_data['rgb'] = cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
                print(f"[INFO] Loaded RGB from {rgb_path}")

            # Load Depth
            depth_path = os.path.join(self.current_frame_dir, "depth.png")
            if os.path.exists(depth_path):
                depth = cv.imread(depth_path, cv.IMREAD_UNCHANGED)
                self.frame_data['depth'] = depth
                print(f"[INFO] Loaded depth from {depth_path}")

            # Load Mask
            mask_path = os.path.join(self.current_frame_dir, "mask.png")
            if os.path.exists(mask_path):
                mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
                self.frame_data['mask'] = mask
                print(f"[INFO] Loaded mask from {mask_path}")

            # Load Camera Matrix
            cam_K_path = os.path.join(self.current_frame_dir, "cam_K.txt")
            if os.path.exists(cam_K_path):
                K = np.loadtxt(cam_K_path)
                self.frame_data['K'] = K
                print(f"[INFO] Loaded camera matrix from {cam_K_path}")

            # Load Pose
            pose_path = os.path.join(self.current_frame_dir, "final_pose.txt")
            if os.path.exists(pose_path):
                pose = np.loadtxt(pose_path)
                self.frame_data['pose'] = pose
                print(f"[INFO] Loaded pose from {pose_path}")

            # Load Mesh
            mesh_path = os.path.join(self.current_frame_dir, "mesh.ply")
            if os.path.exists(mesh_path):
                try:
                    mesh = trimesh.load(mesh_path)
                    self.frame_data['mesh'] = mesh
                    print(f"[INFO] Loaded mesh from {mesh_path}")
                except Exception as e:
                    print(f"[WARNING] Could not load mesh: {e}")

            # Load Metadata
            metadata_dir = os.path.join(self.current_frame_dir, "metadata")
            if os.path.exists(metadata_dir):
                # Settings
                settings_path = os.path.join(metadata_dir, "settings.json")
                if os.path.exists(settings_path):
                    with open(settings_path, 'r') as f:
                        self.frame_data['settings'] = json.load(f)
                    print(f"[INFO] Loaded settings from {settings_path}")

                # AVP Head Pose
                avp_head_pose_path = os.path.join(metadata_dir, "avp_head_pose.json")
                if os.path.exists(avp_head_pose_path):
                    with open(avp_head_pose_path, 'r') as f:
                        self.frame_data['avp_head_pose'] = json.load(f)
                    print(f"[INFO] Loaded AVP head pose from {avp_head_pose_path}")

                # RealSense Transform
                rs_transform_path = os.path.join(metadata_dir, "realsense_transform.json")
                if os.path.exists(rs_transform_path):
                    with open(rs_transform_path, 'r') as f:
                        self.frame_data['rs_transform'] = json.load(f)
                    print(f"[INFO] Loaded RealSense transform from {rs_transform_path}")

                # RealSense Pose
                rs_pose_path = os.path.join(metadata_dir, "realsense_pose.json")
                if os.path.exists(rs_pose_path):
                    with open(rs_pose_path, 'r') as f:
                        self.frame_data['rs_pose'] = json.load(f)
                    print(f"[INFO] Loaded RealSense pose from {rs_pose_path}")

            # Update display
            self.update_display()

        except Exception as e:
            print(f"[ERROR] load_frame_data: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Load Error", f"Failed to load frame data: {e}")

    def update_display(self):
        """Update all displays with loaded data"""
        try:
            # Display RGB
            if 'rgb' in self.frame_data:
                self._set_image('rgb', self.frame_data['rgb'])

            # Display Depth with colormap
            if 'depth' in self.frame_data:
                depth_colored = self._apply_colormap(self.frame_data['depth'])
                self._set_image('depth', depth_colored)

            # Display Mask
            if 'mask' in self.frame_data:
                # Convert grayscale to RGB for display
                mask_rgb = cv.cvtColor(self.frame_data['mask'], cv.COLOR_GRAY2RGB)
                self._set_image('mask', mask_rgb)

            # Display Mesh projection
            if 'mesh' in self.frame_data and 'K' in self.frame_data and 'pose' in self.frame_data:
                mesh_img = self._render_mesh_2d()
                if mesh_img is not None:
                    self._set_image('mesh', mesh_img)

            # Display RGB with pose overlay
            if 'rgb' in self.frame_data and 'K' in self.frame_data and 'pose' in self.frame_data:
                overlay_img = self._draw_pose_overlay()
                if overlay_img is not None:
                    self._set_image('pose_overlay', overlay_img)

            # Update info panels
            self._update_info()

        except Exception as e:
            print(f"[ERROR] update_display: {e}")
            import traceback
            traceback.print_exc()

    def _apply_colormap(self, depth_array, colormap=cv.COLORMAP_TURBO):
        """Apply colormap to depth for visualization"""
        if depth_array.dtype == np.uint16:
            # Normalize to 8-bit
            depth_min = depth_array.min()
            depth_max = depth_array.max()
            if depth_max > depth_min:
                depth_normalized = ((depth_array - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_normalized = np.zeros_like(depth_array, dtype=np.uint8)
        else:
            depth_normalized = depth_array

        colored = cv.applyColorMap(depth_normalized, colormap)
        return colored

    def _render_mesh_2d(self):
        """Render mesh as 2D projection"""
        try:
            mesh = self.frame_data['mesh']
            K = self.frame_data['K']
            T = self.frame_data['pose']

            # Create blank image
            if 'rgb' in self.frame_data:
                height, width = self.frame_data['rgb'].shape[:2]
            else:
                height, width = 480, 640

            img = np.zeros((height, width, 3), dtype=np.uint8)

            # Transform mesh vertices to camera space
            vertices = mesh.vertices
            vertices_hom = np.hstack([vertices, np.ones((len(vertices), 1))])
            vertices_cam = (T @ vertices_hom.T).T[:, :3]

            # Project to image plane
            points_2d = []
            for v in vertices_cam:
                if v[2] > 0:  # In front of camera
                    p = K @ v
                    p = p / p[2]
                    points_2d.append([int(p[0]), int(p[1])])

            # Draw mesh edges
            if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                for face in mesh.faces:
                    pts = []
                    valid = True
                    for idx in face:
                        v = vertices_cam[idx]
                        if v[2] > 0:
                            p = K @ v
                            p = p / p[2]
                            pts.append([int(p[0]), int(p[1])])
                        else:
                            valid = False
                            break

                    if valid and len(pts) == 3:
                        pts = np.array(pts, dtype=np.int32)
                        cv.polylines(img, [pts], True, (0, 255, 0), 1)

            # Draw coordinate axes
            axis_length = 0.1  # 10cm
            origin = np.array([0, 0, 0, 1])
            x_axis = np.array([axis_length, 0, 0, 1])
            y_axis = np.array([0, axis_length, 0, 1])
            z_axis = np.array([0, 0, axis_length, 1])

            origin_cam = T @ origin
            x_cam = T @ x_axis
            y_cam = T @ y_axis
            z_cam = T @ z_axis

            def project(p):
                if p[2] > 0:
                    proj = K @ p[:3]
                    proj = proj / proj[2]
                    return (int(proj[0]), int(proj[1]))
                return None

            origin_2d = project(origin_cam)
            x_2d = project(x_cam)
            y_2d = project(y_cam)
            z_2d = project(z_cam)

            if origin_2d and x_2d:
                cv.line(img, origin_2d, x_2d, (0, 0, 255), 2)  # X-axis: Red
            if origin_2d and y_2d:
                cv.line(img, origin_2d, y_2d, (0, 255, 0), 2)  # Y-axis: Green
            if origin_2d and z_2d:
                cv.line(img, origin_2d, z_2d, (255, 0, 0), 2)  # Z-axis: Blue

            return img

        except Exception as e:
            print(f"[ERROR] _render_mesh_2d: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _draw_pose_overlay(self):
        """Draw pose overlay on RGB image"""
        try:
            rgb = self.frame_data['rgb'].copy()
            K = self.frame_data['K']
            T = self.frame_data['pose']

            # Extract rotation and translation
            R = T[:3, :3]
            t = T[:3, 3]

            # Convert rotation matrix to rotation vector
            rvec, _ = cv.Rodrigues(R)
            tvec = t.reshape(3, 1)

            # Define axis points in 3D (length = 0.1m = 100mm)
            axis_length = 0.1
            axis_points_3d = np.float32([
                [0, 0, 0],  # Origin
                [axis_length, 0, 0],  # X-axis (red)
                [0, axis_length, 0],  # Y-axis (green)
                [0, 0, axis_length]   # Z-axis (blue)
            ])

            # Project 3D points to 2D
            dist = np.array([0, 0, 0, 0, 0], dtype=np.float32)  # Assume no distortion
            img_points, _ = cv.projectPoints(axis_points_3d, rvec, tvec, K, dist)
            img_points = img_points.reshape(-1, 2).astype(int)

            # Draw axes
            origin = tuple(img_points[0])
            x_end = tuple(img_points[1])
            y_end = tuple(img_points[2])
            z_end = tuple(img_points[3])

            # Draw lines with different colors for each axis
            cv.line(rgb, origin, x_end, (255, 0, 0), 3)  # X-axis: Red
            cv.line(rgb, origin, y_end, (0, 255, 0), 3)  # Y-axis: Green
            cv.line(rgb, origin, z_end, (0, 0, 255), 3)  # Z-axis: Blue

            # Draw origin point
            cv.circle(rgb, origin, 5, (255, 255, 255), -1)

            return rgb

        except Exception as e:
            print(f"[ERROR] _draw_pose_overlay: {e}")
            import traceback
            traceback.print_exc()
            return self.frame_data.get('rgb')

    def _set_image(self, key, img_array):
        """Set image in label"""
        if img_array is None:
            return

        try:
            # Convert to PIL Image
            if img_array.ndim == 2:
                # Grayscale
                pil_img = Image.fromarray(img_array)
            elif img_array.shape[2] == 3:
                # RGB
                pil_img = Image.fromarray(img_array)
            else:
                return

            # Resize to fit display
            pil_img.thumbnail((400, 300), Image.LANCZOS)

            # Create PhotoImage and display
            photo = ImageTk.PhotoImage(pil_img)
            self.image_labels[key].configure(image=photo, text="")
            self.photo_refs[key] = photo

        except Exception as e:
            print(f"[ERROR] _set_image({key}): {e}")

    def _update_info(self):
        """Update info text panels"""
        try:
            # Update bottom info panel
            self.info_text.delete("1.0", tk.END)

            self.info_text.insert(tk.END, "=== FRAME DATA ===\n")
            self.info_text.insert(tk.END, f"Folder: {os.path.basename(self.current_frame_dir)}\n\n")

            # Camera intrinsics
            if 'K' in self.frame_data:
                K = self.frame_data['K']
                self.info_text.insert(tk.END, "=== CAMERA INTRINSICS ===\n")
                self.info_text.insert(tk.END, f"fx: {K[0, 0]:.1f}, fy: {K[1, 1]:.1f}\n")
                self.info_text.insert(tk.END, f"cx: {K[0, 2]:.1f}, cy: {K[1, 2]:.1f}\n\n")

            # Pose
            if 'pose' in self.frame_data:
                T = self.frame_data['pose']
                self.info_text.insert(tk.END, "=== POSE (Transformation Matrix) ===\n")
                self.info_text.insert(tk.END, f"Translation: [{T[0, 3]:.3f}, {T[1, 3]:.3f}, {T[2, 3]:.3f}]\n")

                # Extract rotation angles
                R = T[:3, :3]
                rvec, _ = cv.Rodrigues(R)
                self.info_text.insert(tk.END, f"Rotation Vector: [{rvec[0, 0]:.3f}, {rvec[1, 0]:.3f}, {rvec[2, 0]:.3f}]\n\n")

            # Mesh info
            if 'mesh' in self.frame_data:
                mesh = self.frame_data['mesh']
                self.info_text.insert(tk.END, "=== MESH ===\n")
                self.info_text.insert(tk.END, f"Vertices: {len(mesh.vertices)}\n")
                self.info_text.insert(tk.END, f"Faces: {len(mesh.faces) if hasattr(mesh, 'faces') else 'N/A'}\n\n")

            # Update info display in grid
            info_widget = self.image_labels['info']
            info_widget.delete("1.0", tk.END)

            if 'settings' in self.frame_data:
                info_widget.insert(tk.END, "=== SETTINGS ===\n")
                settings = self.frame_data['settings']
                for key, value in settings.items():
                    info_widget.insert(tk.END, f"{key}: {value}\n")
                info_widget.insert(tk.END, "\n")

            if 'avp_head_pose' in self.frame_data:
                info_widget.insert(tk.END, "=== AVP HEAD POSE ===\n")
                head_pose = self.frame_data['avp_head_pose'].get('head_pose', {})
                if 'position' in head_pose:
                    pos = head_pose['position']
                    info_widget.insert(tk.END, f"Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n")
                if 'rotation' in head_pose:
                    rot = head_pose['rotation']
                    info_widget.insert(tk.END, f"Rotation: [{rot[0]:.3f}, {rot[1]:.3f}, {rot[2]:.3f}]\n")
                info_widget.insert(tk.END, "\n")

            if 'rs_transform' in self.frame_data:
                info_widget.insert(tk.END, "=== REALSENSE TRANSFORM ===\n")
                rs_transform = self.frame_data['rs_transform']
                if 'R_avp_rs_c' in rs_transform:
                    R = np.array(rs_transform['R_avp_rs_c'])
                    info_widget.insert(tk.END, "Rotation Matrix (AVP <- RS):\n")
                    for i in range(3):
                        info_widget.insert(tk.END, f"  [{R[i,0]:7.4f} {R[i,1]:7.4f} {R[i,2]:7.4f}]\n")
                info_widget.insert(tk.END, "\n")

        except Exception as e:
            print(f"[ERROR] _update_info: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    root = tk.Tk()
    app = SavedFrameViewer(root)
    root.mainloop()
