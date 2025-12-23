import simd
import ARKit

// Extension to allow initialization from a flat array of 16 Floats, e.g., from UserDefaults
extension simd_float4x4 {
    init?(_ array: [Float]) {
        guard array.count == 16 else { return nil }
        self = simd_float4x4(
            SIMD4<Float>(array[0], array[1], array[2], array[3]),
            SIMD4<Float>(array[4], array[5], array[6], array[7]),
            SIMD4<Float>(array[8], array[9], array[10], array[11]),
            SIMD4<Float>(array[12], array[13], array[14], array[15])
        )
    }
}

/// Utilities for transforming poses between camera space and world space
enum CameraTransformUtils {
    // This will be managed by CalibrationManager
    private static var _calibrationManager: CalibrationManager?

    static func setCalibrationManager(_ manager: CalibrationManager) {
        _calibrationManager = manager
    }

    /// Provides the camera-to-device offset transform.
    /// Reads from CalibrationManager if a calibrated value is available,
    /// otherwise returns a default estimated offset.
    static var currentCameraOffset: simd_float4x4 {
        if let calibratedTransform = _calibrationManager?.calibrationTransformSnapshot {
            return calibratedTransform
        }
        // Default estimated physical offset of AVP main camera from DeviceAnchor origin
        // Coordinate system: x=right, y=down, z=forward (device coordinate system)
        let defaultOffset = SIMD3<Float>(
            0.0,    // x: no lateral offset
            -0.01,  // y: ~1cm down
            0.04    // z: ~4cm forward (TUNABLE during calibration)
        )
        return simd_float4x4(translation: defaultOffset)
    }

    /// Compute camera-to-world transform from device anchor
    /// - Parameter deviceAnchor: Current device anchor from ARKit
    /// - Returns: 4x4 transform from world origin to camera
    static func cameraToWorldTransform(from deviceAnchor: DeviceAnchor) -> simd_float4x4 {
        // Device-to-world transform (head pose)
        let T_world_device = deviceAnchor.originFromAnchorTransform

        // Camera-to-device transform (physical offset)
        let T_device_camera = currentCameraOffset

        // Chain: world ← device ← camera
        return T_world_device * T_device_camera
    }

    /// Compute camera-to-world transform from a cached device transform.
    static func cameraToWorldTransform(from deviceTransform: simd_float4x4) -> simd_float4x4 {
        let T_device_camera = currentCameraOffset
        return deviceTransform * T_device_camera
    }

    /// Transform ArUco camera-space pose to world space
    /// - Parameters:
    ///   - cameraPose: ArUco pose in camera frame (from OpenCV → RealityKit converted)
    ///   - deviceAnchor: Current device anchor
    /// - Returns: ArUco pose in world frame
    static func arucoPoseToWorld(
        cameraPose: simd_float4x4,
        deviceAnchor: DeviceAnchor
    ) -> simd_float4x4 {
        let T_world_camera = cameraToWorldTransform(from: deviceAnchor)
        return T_world_camera * cameraPose
    }

    /// Transform ArUco camera-space pose to world space with cached device transform.
    static func arucoPoseToWorld(
        cameraPose: simd_float4x4,
        deviceTransform: simd_float4x4
    ) -> simd_float4x4 {
        let T_world_camera = cameraToWorldTransform(from: deviceTransform)
        return T_world_camera * cameraPose
    }

    /// Compute the relative transform from ArUco to device
    /// This is used for continuous tracking when the marker is not visible
    /// - Parameters:
    ///   - cameraPose: ArUco pose in camera frame
    ///   - deviceAnchor: Current device anchor
    /// - Returns: Transform from device to ArUco in world space
    static func computeArucoToDeviceTransform(
        cameraPose: simd_float4x4,
        deviceAnchor: DeviceAnchor
    ) -> simd_float4x4 {
        // T_world_aruco = T_world_camera × T_camera_aruco
        let T_world_aruco = arucoPoseToWorld(cameraPose: cameraPose, deviceAnchor: deviceAnchor)

        // T_device_aruco = inv(T_world_device) × T_world_aruco
        let T_world_device = deviceAnchor.originFromAnchorTransform
        let T_device_world = T_world_device.inverse

        return T_device_world * T_world_aruco
    }

    /// Compute the relative transform from ArUco to device using cached transform.
    static func computeArucoToDeviceTransform(
        cameraPose: simd_float4x4,
        deviceTransform: simd_float4x4
    ) -> simd_float4x4 {
        let T_world_aruco = arucoPoseToWorld(cameraPose: cameraPose, deviceTransform: deviceTransform)
        let T_device_world = deviceTransform.inverse
        return T_device_world * T_world_aruco
    }

    /// Estimate ArUco world pose from stored device-relative transform
    /// Used for continuous tracking when marker is not visible
    /// - Parameters:
    ///   - deviceToAruco: Stored transform from device to ArUco
    ///   - deviceAnchor: Current device anchor
    /// - Returns: Estimated ArUco pose in world frame
    static func estimateArucoWorldPose(
        deviceToAruco: simd_float4x4,
        deviceAnchor: DeviceAnchor
    ) -> simd_float4x4 {
        let T_world_device = deviceAnchor.originFromAnchorTransform
        return T_world_device * deviceToAruco
    }

    /// Estimate ArUco world pose using cached device transform.
    static func estimateArucoWorldPose(
        deviceToAruco: simd_float4x4,
        deviceTransform: simd_float4x4
    ) -> simd_float4x4 {
        return deviceTransform * deviceToAruco
    }
}
