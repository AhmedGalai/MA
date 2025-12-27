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

    /// Compute camera-to-world transform from a cached device transform.
    static func cameraToWorldTransform(from deviceTransform: simd_float4x4) -> simd_float4x4 {
        let T_device_camera = simd_float4x4(translation: estimatedCameraOffset)
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


// import simd
// import ARKit

// /// Utilities for transforming poses between camera space and world space
// enum CameraTransformUtils {
//     /// Estimated physical offset of AVP main camera from DeviceAnchor origin
//     /// Coordinate system: x=right, y=down, z=forward (device coordinate system)
//     static var estimatedCameraOffset = SIMD3<Float>(
//         0.0,    // x: no lateral offset
//         -0.01,  // y: ~1cm down
//         0.04    // z: ~4cm forward (TUNABLE during calibration)
//     )

//     /// Transform that maps CAMERA coordinates into DEVICE coordinates.
//     /// Default is translation-only, derived from `estimatedCameraOffset`.
//     private(set) static var deviceFromCameraTransform: simd_float4x4 = translationMatrix(estimatedCameraOffset)

//     /// Reset to the current `estimatedCameraOffset` (translation-only).
//     static func resetCameraCalibration() {
//         deviceFromCameraTransform = translationMatrix(estimatedCameraOffset)
//     }

//     /// Hard-set translation-only camera offset.
//     static func setEstimatedCameraOffset(_ offset: SIMD3<Float>) {
//         estimatedCameraOffset = offset
//         deviceFromCameraTransform = translationMatrix(offset)
//     }

//     /// Calibrate the translation-only camera offset using:
//     /// - worldFromGizmo: a user-placed world anchor aligned to the ArUco board frame
//     /// - cameraFromAruco: ArUco board pose in camera coordinates (after OpenCV->RealityKit conversion)
//     /// - worldFromDevice: device pose in world (ARKit)
//     ///
//     /// Math:
//     /// worldFromGizmo ≈ worldFromDevice * deviceFromCamera * cameraFromAruco
//     /// => deviceFromCamera ≈ inv(worldFromDevice) * worldFromGizmo * inv(cameraFromAruco)
//     ///
//     /// This function uses translation-only calibration and exponential smoothing.
//     static func calibrateCameraOffsetTranslationOnly(
//         worldFromGizmo: simd_float4x4,
//         cameraFromAruco: simd_float4x4,
//         worldFromDevice: simd_float4x4,
//         smoothing: Float = 0.10
//     ) {
//         let deviceFromWorld = worldFromDevice.inverse
//         let arucoFromCamera = cameraFromAruco.inverse

//         let candidateDeviceFromCamera = deviceFromWorld * worldFromGizmo * arucoFromCamera

//         let t = SIMD3<Float>(
//             candidateDeviceFromCamera.columns.3.x,
//             candidateDeviceFromCamera.columns.3.y,
//             candidateDeviceFromCamera.columns.3.z
//         )

//         // Exponential smoothing
//         estimatedCameraOffset = (1 - smoothing) * estimatedCameraOffset + smoothing * t
//         deviceFromCameraTransform = translationMatrix(estimatedCameraOffset)
//     }

//     /// Compute camera-to-world transform from device anchor
//     /// - Parameter deviceAnchor: Current device anchor from ARKit
//     /// - Returns: 4x4 transform from world origin to camera
//     static func cameraToWorldTransform(from deviceAnchor: DeviceAnchor) -> simd_float4x4 {
//         let worldFromDevice = deviceAnchor.originFromAnchorTransform
//         return worldFromDevice * deviceFromCameraTransform
//     }

//     /// Compute camera-to-world transform from a cached device transform.
//     static func cameraToWorldTransform(from deviceTransform: simd_float4x4) -> simd_float4x4 {
//         return deviceTransform * deviceFromCameraTransform
//     }

//     /// Transform ArUco camera-space pose to world space
//     /// - Parameters:
//     ///   - cameraPose: ArUco pose in camera frame (cameraFromAruco) (OpenCV → RealityKit converted)
//     ///   - deviceAnchor: Current device anchor
//     /// - Returns: ArUco pose in world frame (worldFromAruco)
//     static func arucoPoseToWorld(
//         cameraPose: simd_float4x4,
//         deviceAnchor: DeviceAnchor
//     ) -> simd_float4x4 {
//         let worldFromCamera = cameraToWorldTransform(from: deviceAnchor)
//         return worldFromCamera * cameraPose
//     }

//     /// Transform ArUco camera-space pose to world space with cached device transform.
//     static func arucoPoseToWorld(
//         cameraPose: simd_float4x4,
//         deviceTransform: simd_float4x4
//     ) -> simd_float4x4 {
//         let worldFromCamera = cameraToWorldTransform(from: deviceTransform)
//         return worldFromCamera * cameraPose
//     }

//     /// Compute the relative transform from ArUco to device
//     /// - Returns: deviceFromAruco in world-consistent coordinates
//     static func computeArucoToDeviceTransform(
//         cameraPose: simd_float4x4,
//         deviceAnchor: DeviceAnchor
//     ) -> simd_float4x4 {
//         let worldFromAruco = arucoPoseToWorld(cameraPose: cameraPose, deviceAnchor: deviceAnchor)
//         let worldFromDevice = deviceAnchor.originFromAnchorTransform
//         let deviceFromWorld = worldFromDevice.inverse
//         return deviceFromWorld * worldFromAruco
//     }

//     /// Compute the relative transform from ArUco to device using cached transform.
//     static func computeArucoToDeviceTransform(
//         cameraPose: simd_float4x4,
//         deviceTransform: simd_float4x4
//     ) -> simd_float4x4 {
//         let worldFromAruco = arucoPoseToWorld(cameraPose: cameraPose, deviceTransform: deviceTransform)
//         let deviceFromWorld = deviceTransform.inverse
//         return deviceFromWorld * worldFromAruco
//     }

//     /// Estimate ArUco world pose from stored device-relative transform
//     static func estimateArucoWorldPose(
//         deviceToAruco: simd_float4x4,
//         deviceAnchor: DeviceAnchor
//     ) -> simd_float4x4 {
//         let worldFromDevice = deviceAnchor.originFromAnchorTransform
//         return worldFromDevice * deviceToAruco
//     }

//     /// Estimate ArUco world pose using cached device transform.
//     static func estimateArucoWorldPose(
//         deviceToAruco: simd_float4x4,
//         deviceTransform: simd_float4x4
//     ) -> simd_float4x4 {
//         return deviceTransform * deviceToAruco
//     }

//     private static func translationMatrix(_ t: SIMD3<Float>) -> simd_float4x4 {
//         var m = matrix_identity_float4x4
//         m.columns.3.x = t.x
//         m.columns.3.y = t.y
//         m.columns.3.z = t.z
//         return m
//     }
// }
