import simd
import ARKit

/// Utilities for transforming poses between camera space and world space
enum CameraTransformUtils {
    /// Estimated physical offset of AVP main camera from DeviceAnchor origin
    /// Based on reverse-engineering and community observations
    /// Coordinate system: x=right, y=down, z=forward (device coordinate system)
    static var estimatedCameraOffset = SIMD3<Float>(
        0.0,    // x: no lateral offset
        -0.01,  // y: ~1cm down
        0.04    // z: ~4cm forward (TUNABLE during calibration)
    )

    /// Compute camera-to-world transform from device anchor
    /// - Parameter deviceAnchor: Current device anchor from ARKit
    /// - Returns: 4x4 transform from world origin to camera
    static func cameraToWorldTransform(from deviceAnchor: DeviceAnchor) -> simd_float4x4 {
        // Device-to-world transform (head pose)
        let T_world_device = deviceAnchor.originFromAnchorTransform

        // Camera-to-device transform (physical offset)
        let T_device_camera = simd_float4x4(translation: estimatedCameraOffset)

        // Chain: world ← device ← camera
        return T_world_device * T_device_camera
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
}
