"""
New API Endpoints to Add to main_api.py
========================================

Insert these 5 endpoints before the error handlers (line 1416 in main_api.py)

Phase 3 Endpoints (3):
1. /get_transformed_depth - Transform RS depth to AVP view
2. /get_roi_rgb - Extract ROI from AVP frame
3. /get_roi_binary_mask - Apply HSV color filtering

Phase 6 Endpoints (2):
4. /foundation_pose_request - Forward FoundationPose request with AVP data
5. /transform_depth_rs_to_avp - Transform depth for specific K_avp

"""

# ====================
# PHASE 3 ENDPOINTS
# ====================

@app.route('/get_transformed_depth', methods=['GET'])
def get_transformed_depth():
    """
    Transform RealSense depth map to AVP view.

    Process:
        1. Capture RS depth map and K_rs
        2. Get T_avp_rs transformation from coordinate_manager
        3. Create point cloud from RS depth + K_rs
        4. Transform point cloud to AVP frame
        5. Project to AVP image plane using K_avp
        6. Generate depth colormap in AVP view
        7. Return as base64 JPEG

    Query parameters:
        colormap (optional): OpenCV colormap to use (default: COLORMAP_JET)

    Returns:
        JSON with base64-encoded depth visualization
    """
    try:
        colormap_name = request.args.get('colormap', 'COLORMAP_JET')
        colormap = getattr(cv2, colormap_name, cv2.COLORMAP_JET)

        frame_data, using_cache = get_latest_rs_frame(allow_cache=True)
        if frame_data is None:
            return jsonify({'error': 'RealSense not connected or no frames available'}), 503

        depth_rs = frame_data['depth']
        K_rs = frame_data['K']

        with state_lock:
            ensure_coordinate_manager()
            if coordinate_manager is None:
                return jsonify({'error': 'CoordinateManager not initialized'}), 500

            if not coordinate_manager.is_calibrated():
                depth_normalized = cv2.normalize(depth_rs, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_colormap = cv2.applyColorMap(depth_normalized, colormap)
                _, buffer = cv2.imencode('.jpg', depth_colormap, [cv2.IMWRITE_JPEG_QUALITY, 85])
                depth_b64 = base64.b64encode(buffer).decode('utf-8')

                return jsonify({
                    'depth_colormap': f'data:image/jpeg;base64,{depth_b64}',
                    'timestamp': frame_data.get('timestamp', time.time()),
                    'transformation_applied': False,
                    'message': 'System not calibrated - returning RS depth view',
                    'min_depth': float(np.min(depth_rs[depth_rs > 0])) if np.any(depth_rs > 0) else 0.0,
                    'max_depth': float(np.max(depth_rs))
                }), 200

            T_avp_rs = coordinate_manager.get_T_avp_rs()
            K_avp = avp_intrinsics['K']

            if K_avp is None:
                return jsonify({'error': 'AVP intrinsics not calculated yet'}), 400

        h_rs, w_rs = depth_rs.shape
        u, v = np.meshgrid(np.arange(w_rs), np.arange(h_rs))
        u = u.flatten()
        v = v.flatten()
        z = depth_rs.flatten()

        valid_mask = z > 0
        u = u[valid_mask]
        v = v[valid_mask]
        z = z[valid_mask]

        if len(z) == 0:
            return jsonify({'error': 'No valid depth data'}), 400

        fx_rs, fy_rs = K_rs[0, 0], K_rs[1, 1]
        cx_rs, cy_rs = K_rs[0, 2], K_rs[1, 2]

        X_rs = (u - cx_rs) * z / fx_rs
        Y_rs = (v - cy_rs) * z / fy_rs
        Z_rs = z

        points_rs = np.vstack([X_rs, Y_rs, Z_rs, np.ones_like(Z_rs)])
        points_avp = T_avp_rs @ points_rs

        X_avp = points_avp[0, :]
        Y_avp = points_avp[1, :]
        Z_avp = points_avp[2, :]

        valid_depth = Z_avp > 0
        X_avp = X_avp[valid_depth]
        Y_avp = Y_avp[valid_depth]
        Z_avp = Z_avp[valid_depth]

        if len(Z_avp) == 0:
            return jsonify({'error': 'No points visible in AVP view'}), 400

        fx_avp, fy_avp = K_avp[0, 0], K_avp[1, 1]
        cx_avp, cy_avp = K_avp[0, 2], K_avp[1, 2]

        u_avp = (X_avp * fx_avp / Z_avp) + cx_avp
        v_avp = (Y_avp * fy_avp / Z_avp) + cy_avp

        with state_lock:
            frame, _, metadata = get_avp_frame_for_purpose('general')
            if frame is None:
                h_avp = int(2 * cy_avp)
                w_avp = int(2 * cx_avp)
            else:
                h_avp, w_avp = frame.shape[:2]

        in_bounds = (u_avp >= 0) & (u_avp < w_avp) & (v_avp >= 0) & (v_avp < h_avp)
        u_avp = u_avp[in_bounds]
        v_avp = v_avp[in_bounds]
        Z_avp = Z_avp[in_bounds]

        if len(Z_avp) == 0:
            return jsonify({'error': 'No points project into AVP image bounds'}), 400

        depth_avp = np.zeros((h_avp, w_avp), dtype=np.float32)
        u_int = u_avp.astype(np.int32)
        v_int = v_avp.astype(np.int32)

        for i in range(len(u_int)):
            depth_avp[v_int[i], u_int[i]] = max(depth_avp[v_int[i], u_int[i]], Z_avp[i])

        mask = (depth_avp > 0).astype(np.uint8)
        depth_avp_filled = cv2.inpaint(
            (depth_avp * 1000).astype(np.uint16),
            1 - mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_NS
        ).astype(np.float32) / 1000.0

        depth_normalized = cv2.normalize(depth_avp_filled, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_normalized, colormap)

        _, buffer = cv2.imencode('.jpg', depth_colormap, [cv2.IMWRITE_JPEG_QUALITY, 85])
        depth_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'depth_colormap': f'data:image/jpeg;base64,{depth_b64}',
            'timestamp': frame_data.get('timestamp', time.time()),
            'transformation_applied': True,
            'min_depth': float(np.min(Z_avp)),
            'max_depth': float(np.max(Z_avp)),
            'num_points': int(len(Z_avp)),
            'stale': using_cache
        }), 200

    except Exception as e:
        logger.error(f"Error in get_transformed_depth: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/get_roi_rgb', methods=['GET'])
def get_roi_rgb():
    """
    Get ROI (Region of Interest) RGB image from AVP frame.

    Query parameters:
        x, y, width, height: ROI bounds
        purpose: Frame purpose (default: 'general')

    Returns:
        JSON with base64-encoded cropped RGB image
    """
    try:
        purpose = request.args.get('purpose', 'general')

        with state_lock:
            frame, timestamp, metadata = get_avp_frame_for_purpose(purpose)
            if frame is None:
                return jsonify({'error': f'No AVP frame available for {purpose}'}), 404
            frame = frame.copy()

        h, w = frame.shape[:2]

        x = int(request.args.get('x', 0))
        y = int(request.args.get('y', 0))
        width = int(request.args.get('width', w))
        height = int(request.args.get('height', h))

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        width = max(1, min(width, w - x))
        height = max(1, min(height, h - y))

        roi = frame[y:y+height, x:x+width]

        if roi.size == 0:
            return jsonify({'error': 'Invalid ROI - empty region'}), 400

        _, buffer = cv2.imencode('.jpg', roi, [cv2.IMWRITE_JPEG_QUALITY, 90])
        roi_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'roi_rgb': f'data:image/jpeg;base64,{roi_b64}',
            'roi_x': x,
            'roi_y': y,
            'roi_width': width,
            'roi_height': height,
            'original_width': w,
            'original_height': h,
            'timestamp': timestamp,
            'purpose': purpose
        }), 200

    except ValueError as e:
        logger.error(f"Invalid parameter in get_roi_rgb: {e}")
        return jsonify({'error': f'Invalid parameter: {e}'}), 400
    except Exception as e:
        logger.error(f"Error in get_roi_rgb: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/get_roi_binary_mask', methods=['POST'])
def get_roi_binary_mask():
    """
    Apply HSV color filter to ROI and generate binary mask.

    Expected JSON:
        {
            "x": int, "y": int, "width": int, "height": int,
            "hsv_lower": [h, s, v], "hsv_upper": [h, s, v],
            "purpose": str (optional)
        }

    Returns:
        JSON with base64-encoded binary mask as PNG
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        required_fields = ['x', 'y', 'width', 'height', 'hsv_lower', 'hsv_upper']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        purpose = data.get('purpose', 'general')
        x = int(data['x'])
        y = int(data['y'])
        width = int(data['width'])
        height = int(data['height'])
        hsv_lower = np.array(data['hsv_lower'], dtype=np.uint8)
        hsv_upper = np.array(data['hsv_upper'], dtype=np.uint8)

        if hsv_lower.shape != (3,) or hsv_upper.shape != (3,):
            return jsonify({'error': 'HSV bounds must be arrays of length 3'}), 400

        with state_lock:
            frame, timestamp, metadata = get_avp_frame_for_purpose(purpose)
            if frame is None:
                return jsonify({'error': f'No AVP frame available for {purpose}'}), 404
            frame = frame.copy()

        h, w = frame.shape[:2]

        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        width = max(1, min(width, w - x))
        height = max(1, min(height, h - y))

        roi_bgr = frame[y:y+height, x:x+width]

        if roi_bgr.size == 0:
            return jsonify({'error': 'Invalid ROI - empty region'}), 400

        roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        binary_mask = cv2.inRange(roi_hsv, hsv_lower, hsv_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        mask_pixels = int(np.sum(binary_mask > 0))
        total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
        coverage = (mask_pixels / total_pixels * 100) if total_pixels > 0 else 0.0

        _, buffer = cv2.imencode('.png', binary_mask)
        mask_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'binary_mask': f'data:image/png;base64,{mask_b64}',
            'roi_x': x,
            'roi_y': y,
            'roi_width': width,
            'roi_height': height,
            'mask_pixels': mask_pixels,
            'total_pixels': total_pixels,
            'coverage': float(coverage),
            'original_width': w,
            'original_height': h,
            'timestamp': timestamp,
            'purpose': purpose,
            'hsv_lower': hsv_lower.tolist(),
            'hsv_upper': hsv_upper.tolist()
        }), 200

    except ValueError as e:
        logger.error(f"Invalid parameter in get_roi_binary_mask: {e}")
        return jsonify({'error': f'Invalid parameter: {e}'}), 400
    except Exception as e:
        logger.error(f"Error in get_roi_binary_mask: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ====================
# PHASE 6 ENDPOINTS
# ====================

@app.route('/foundation_pose_request', methods=['POST'])
def foundation_pose_request():
    """
    Forward FoundationPose request to FoundationPose API.

    Expected JSON:
        {
            "roi_rgb": base64_string (JPEG),
            "transformed_depth": base64_string (PNG disparity),
            "avp_intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "mask": base64_string (PNG),
            "mesh_path": "path/to/mesh.ply"
        }

    Returns:
        JSON with pose directly in AVP frame (no transformation needed)
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        required_fields = ['roi_rgb', 'transformed_depth', 'avp_intrinsics', 'mask', 'mesh_path']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing {field}'}), 400

        roi_rgb_b64 = data['roi_rgb']
        transformed_depth_b64 = data['transformed_depth']
        K_avp = np.array(data['avp_intrinsics'], dtype=np.float32)
        mask_b64 = data['mask']
        mesh_path = data['mesh_path']

        # Validate mesh path
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.join(CONFIG["paths"]["models_dir"], mesh_path)

        if not os.path.exists(mesh_path):
            return jsonify({'error': f'Mesh file not found: {mesh_path}'}), 400

        # Decode ROI RGB
        try:
            if ',' in roi_rgb_b64:
                roi_rgb_b64 = roi_rgb_b64.split(',')[1]
            rgb_data = base64.b64decode(roi_rgb_b64)
            nparr = np.frombuffer(rgb_data, np.uint8)
            roi_rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if roi_rgb is None:
                return jsonify({'error': 'Failed to decode ROI RGB'}), 400
        except Exception as e:
            logger.error(f"Error decoding ROI RGB: {e}")
            return jsonify({'error': f'Failed to decode ROI RGB: {e}'}), 400

        # Decode transformed depth
        try:
            if ',' in transformed_depth_b64:
                transformed_depth_b64 = transformed_depth_b64.split(',')[1]
            depth_data = base64.b64decode(transformed_depth_b64)
            nparr = np.frombuffer(depth_data, np.uint8)
            transformed_depth = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if transformed_depth is None:
                return jsonify({'error': 'Failed to decode transformed depth'}), 400

            transformed_depth = transformed_depth.astype(np.float32)
        except Exception as e:
            logger.error(f"Error decoding transformed depth: {e}")
            return jsonify({'error': f'Failed to decode transformed depth: {e}'}), 400

        # Decode mask
        try:
            if ',' in mask_b64:
                mask_b64 = mask_b64.split(',')[1]
            mask_data = base64.b64decode(mask_b64)
            nparr = np.frombuffer(mask_data, np.uint8)
            mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                return jsonify({'error': 'Failed to decode mask'}), 400
        except Exception as e:
            logger.error(f"Error decoding mask: {e}")
            return jsonify({'error': f'Failed to decode mask: {e}'}), 400

        # Call FoundationPose API
        try:
            from foundationpose_client import estimate_pose

            foundationpose_url = CONFIG["network"]["foundationpose_url"]

            pose_result = estimate_pose(
                rgb=roi_rgb,
                depth=transformed_depth,
                mask=mask,
                K=K_avp,
                mesh_path=mesh_path,
                api_url=foundationpose_url
            )

            if pose_result is None:
                return jsonify({
                    'success': False,
                    'error': 'FoundationPose API returned no result'
                }), 500

            return jsonify({
                'success': True,
                'pose_avp': pose_result.tolist(),
                'confidence': 1.0
            }), 200

        except Exception as e:
            logger.error(f"Error calling FoundationPose: {e}", exc_info=True)
            return jsonify({'error': f'FoundationPose failed: {e}'}), 500

    except Exception as e:
        logger.error(f"Error in foundation_pose_request: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/transform_depth_rs_to_avp', methods=['POST'])
def transform_depth_rs_to_avp():
    """
    Transform depth map from RealSense view to AVP view.

    Expected JSON:
        {
            "K_avp": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            "target_width": int,
            "target_height": int
        }

    Returns:
        JSON with transformed depth array in AVP view
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        if 'K_avp' not in data:
            return jsonify({'error': 'Missing K_avp'}), 400

        K_avp = np.array(data['K_avp'], dtype=np.float32)
        target_width = data.get('target_width', 640)
        target_height = data.get('target_height', 480)

        with state_lock:
            if realsense_client is None or not realsense_client.is_running:
                return jsonify({'error': 'RealSenseClient not available'}), 500

            if coordinate_manager is None or not coordinate_manager.is_calibrated():
                return jsonify({'error': 'System not calibrated'}), 400

            frame_data, using_cache = get_latest_rs_frame(allow_cache=True)
            if frame_data is None:
                return jsonify({'error': 'Failed to capture RealSense frame'}), 500

            depth_rs = frame_data['depth']
            K_rs = frame_data['K']
            T_avp_rs = coordinate_manager.get_T_avp_rs()

            # Use the same point cloud transformation logic as get_transformed_depth
            h_rs, w_rs = depth_rs.shape
            u, v = np.meshgrid(np.arange(w_rs), np.arange(h_rs))
            u = u.flatten()
            v = v.flatten()
            z = depth_rs.flatten()

            valid_mask = z > 0.01
            u = u[valid_mask]
            v = v[valid_mask]
            z = z[valid_mask]

            if len(z) == 0:
                return jsonify({'error': 'No valid depth data'}), 400

            fx_rs, fy_rs = K_rs[0, 0], K_rs[1, 1]
            cx_rs, cy_rs = K_rs[0, 2], K_rs[1, 2]

            X_rs = (u - cx_rs) * z / fx_rs
            Y_rs = (v - cy_rs) * z / fy_rs
            Z_rs = z

            points_rs = np.vstack([X_rs, Y_rs, Z_rs, np.ones_like(Z_rs)])
            points_avp = T_avp_rs @ points_rs

            X_avp = points_avp[0, :]
            Y_avp = points_avp[1, :]
            Z_avp = points_avp[2, :]

            valid_depth = Z_avp > 0.01
            X_avp = X_avp[valid_depth]
            Y_avp = Y_avp[valid_depth]
            Z_avp = Z_avp[valid_depth]

            if len(Z_avp) == 0:
                return jsonify({'error': 'No points visible in AVP view'}), 400

            fx_avp, fy_avp = K_avp[0, 0], K_avp[1, 1]
            cx_avp, cy_avp = K_avp[0, 2], K_avp[1, 2]

            u_avp = (X_avp * fx_avp / Z_avp) + cx_avp
            v_avp = (Y_avp * fy_avp / Z_avp) + cy_avp

            in_bounds = (u_avp >= 0) & (u_avp < target_width) & (v_avp >= 0) & (v_avp < target_height)
            u_avp = u_avp[in_bounds]
            v_avp = v_avp[in_bounds]
            Z_avp = Z_avp[in_bounds]

            if len(Z_avp) == 0:
                return jsonify({'error': 'No points project into AVP image bounds'}), 400

            depth_avp = np.zeros((target_height, target_width), dtype=np.float32)
            u_int = u_avp.astype(np.int32)
            v_int = v_avp.astype(np.int32)

            for i in range(len(u_int)):
                if depth_avp[v_int[i], u_int[i]] == 0 or Z_avp[i] < depth_avp[v_int[i], u_int[i]]:
                    depth_avp[v_int[i], u_int[i]] = Z_avp[i]

            mask = (depth_avp > 0).astype(np.uint8)
            depth_avp_filled = cv2.inpaint(
                (depth_avp * 1000).astype(np.uint16),
                1 - mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_NS
            ).astype(np.float32) / 1000.0

        # Encode as PNG
        try:
            disparity = 1.0 / (depth_avp_filled + 1e-6)
            disparity[np.isinf(disparity)] = 0

            disparity_min = np.min(disparity[disparity > 0]) if np.any(disparity > 0) else 1.0
            disparity_max = np.max(disparity)

            if disparity_max - disparity_min < 1e-6:
                disparity_normalized = np.zeros_like(disparity, dtype=np.uint8)
            else:
                disparity_normalized = (
                    (disparity - disparity_min) / (disparity_max - disparity_min) * 255
                ).astype(np.uint8)

            success, encoded = cv2.imencode('.png', disparity_normalized)
            if not success:
                return jsonify({'error': 'Failed to encode transformed depth'}), 500

            depth_b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')

            return jsonify({
                'success': True,
                'transformed_depth': f'data:image/png;base64,{depth_b64}',
                'shape': [target_height, target_width],
                'stale': using_cache
            }), 200

        except Exception as e:
            logger.error(f"Error encoding transformed depth: {e}", exc_info=True)
            return jsonify({'error': f'Failed to encode depth: {e}'}), 500

    except Exception as e:
        logger.error(f"Error in transform_depth_rs_to_avp: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
