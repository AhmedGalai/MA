// import SwiftUI
// import RealityKit
// import simd
// import UIKit
// import ARKit

// struct ImmersiveSpaceView: View {
//     @EnvironmentObject private var appModel: AppModel
//     @EnvironmentObject private var sensorModel: SensorDataModel
//     @EnvironmentObject private var arucoStream: ArucoStreamModel
//     @EnvironmentObject private var calibrationManager: CalibrationManager
//     @EnvironmentObject private var rsPoseModel: RealSensePoseModel
//     @EnvironmentObject private var foundationPoseModel: FoundationPoseModel

//     @State private var boardAxes: Entity?
//     @State private var worldAnchor: AnchorEntity?
//     @State private var rsAxes: Entity?
//     @State private var foundationAxes: Entity?
//     @State private var labelsAdded = false

//     var body: some View {
//         RealityView { content in
//             let anchor = AnchorEntity(.world(transform: matrix_identity_float4x4))
//             content.add(anchor)
//             worldAnchor = anchor

//             let boardAxesEntity = makeBoardAxesEntity(includeLabel: false)
//             boardAxesEntity.name = "BoardAxes"
//             boardAxesEntity.isEnabled = false
//             anchor.addChild(boardAxesEntity)
//             boardAxes = boardAxesEntity

//             let rsAxesEntity = makeRSEntity(includeLabel: false)
//             rsAxesEntity.name = "RealSenseAxes"
//             rsAxesEntity.isEnabled = false
//             anchor.addChild(rsAxesEntity)
//             rsAxes = rsAxesEntity

//             let foundationAxesEntity = makeFoundationPoseEntity(includeLabel: false)
//             foundationAxesEntity.name = "FoundationPoseAxes"
//             foundationAxesEntity.isEnabled = false
//             anchor.addChild(foundationAxesEntity)
//             foundationAxes = foundationAxesEntity
//         } update: { _ in
//             updateBoardAxes()
//             updateRSPose()
//             updateFoundationPose()
//         }
//         .task {
//             await MainActor.run { appModel.setImmersiveSpacePresented(true) }
//             await Task.yield()

//             Task { @MainActor in
//                 guard !labelsAdded else { return }
//                 labelsAdded = true
//                 if let boardAxes {
//                     addLabel(to: boardAxes, text: "aruco", position: [0, 0.14, 0])
//                 }
//                 if let rsAxes {
//                     addLabel(to: rsAxes, text: "Realsense", position: [0, 0.12, 0])
//                 }
//                 if let foundationAxes {
//                     addLabel(to: foundationAxes, text: "foundationpose", position: [0, 0.12, 0])
//                 }
//             }

//             Task(priority: .userInitiated) {
//                 await sensorModel.restartARKitTracking()
//             }
//         }
//         .onDisappear {
//             appModel.setImmersiveSpacePresented(false)
//         }
//     }
    
//     private func updateBoardAxes() {
//         guard let boardAxes else { return }

//         // Use cached device transform from the sensor model to avoid re-querying ARKit here.
//         guard let deviceTransform = sensorModel.latestDeviceTransform else {
//             boardAxes.isEnabled = false
//             return
//         }

//         // Check if ArUco is currently detected
//         if let T_camera_aruco = arucoStream.calibratedBoardTransform {
//             // ArUco detected - compute and store device-to-aruco transform for future tracking
//             let T_device_aruco = CameraTransformUtils.computeArucoToDeviceTransform(
//                 cameraPose: T_camera_aruco,
//                 deviceTransform: deviceTransform
//             )

//             // Transform to world space
//             let T_world_aruco = CameraTransformUtils.arucoPoseToWorld(
//                 cameraPose: T_camera_aruco,
//                 deviceTransform: deviceTransform
//             )

//             boardAxes.isEnabled = true
//             boardAxes.transform = Transform(matrix: T_world_aruco)
//         } else {
//             // No ArUco detected and no stored transform - disable
//             boardAxes.isEnabled = false
//         }
//     }

//     private func updateRSPose() {
//         guard let rsAxes else { return }
//         if let matrix = rsPoseModel.rsPoseMatrix {
//             rsAxes.isEnabled = true
//             rsAxes.transform = Transform(matrix: matrix)
//         } else {
//             rsAxes.isEnabled = false
//         }
//     }

//     private func updateFoundationPose() {
//         guard let foundationAxes else { return }
//         if let matrix = foundationPoseModel.poseMatrix {
//             foundationAxes.isEnabled = true
//             foundationAxes.transform = Transform(matrix: matrix)
//         } else {
//             foundationAxes.isEnabled = true
//             foundationAxes.transform = Transform()
//         }
//     }

//     private func makeAxesEntity(length: Float) -> Entity {
//         let entity = Entity()
//         let radius: Float = 0.004

//         let xMesh = MeshResource.generateBox(size: [length, radius, radius])
//         let yMesh = MeshResource.generateBox(size: [radius, length, radius])
//         let zMesh = MeshResource.generateBox(size: [radius, radius, length])

//         let xMat = SimpleMaterial(color: .red, roughness: 0.2, isMetallic: false)
//         let yMat = SimpleMaterial(color: .green, roughness: 0.2, isMetallic: false)
//         let zMat = SimpleMaterial(color: .blue, roughness: 0.2, isMetallic: false)

//         let xEntity = ModelEntity(mesh: xMesh, materials: [xMat])
//         xEntity.position = [length / 2, 0, 0]

//         let yEntity = ModelEntity(mesh: yMesh, materials: [yMat])
//         yEntity.position = [0, length / 2, 0]

//         let zEntity = ModelEntity(mesh: zMesh, materials: [zMat])
//         zEntity.position = [0, 0, length / 2]

//         entity.addChild(xEntity)
//         entity.addChild(yEntity)
//         entity.addChild(zEntity)
//         return entity
//     }

//     private func makeBoardAxesEntity(includeLabel: Bool) -> Entity {
//         let root = Entity()
//         let scale: Float = 0.6

//         let xArrow = ArrowFactory.makeArrow(color: .red)
//         xArrow.transform.rotation = simd_quatf(angle: .pi / 2, axis: [0, 1, 0])
//         xArrow.scale = SIMD3<Float>(repeating: scale)

//         let yArrow = ArrowFactory.makeArrow(color: .green)
//         yArrow.transform.rotation = simd_quatf(angle: -.pi / 2, axis: [1, 0, 0])
//         yArrow.scale = SIMD3<Float>(repeating: scale)

//         let zArrow = ArrowFactory.makeArrow(color: .blue)
//         zArrow.scale = SIMD3<Float>(repeating: scale)

//         root.addChild(xArrow)
//         root.addChild(yArrow)
//         root.addChild(zArrow)
//         if includeLabel {
//             addLabel(to: root, text: "aruco", position: [0, 0.14, 0])
//         }
//         return root
//     }

//     private func makeRSEntity(includeLabel: Bool) -> Entity {
//         let root = Entity()
//         let scale: Float = 0.45

//         let xArrow = ArrowFactory.makeArrow(color: .red)
//         xArrow.transform.rotation = simd_quatf(angle: .pi / 2, axis: [0, 1, 0])
//         xArrow.scale = SIMD3<Float>(repeating: scale)

//         let yArrow = ArrowFactory.makeArrow(color: .green)
//         yArrow.transform.rotation = simd_quatf(angle: -.pi / 2, axis: [1, 0, 0])
//         yArrow.scale = SIMD3<Float>(repeating: scale)

//         let zArrow = ArrowFactory.makeArrow(color: .blue)
//         zArrow.scale = SIMD3<Float>(repeating: scale)

//         root.addChild(xArrow)
//         root.addChild(yArrow)
//         root.addChild(zArrow)

//         if includeLabel {
//             addLabel(to: root, text: "Realsense", position: [0, 0.12, 0])
//         }
//         return root
//     }

//     private func makeFoundationPoseEntity(includeLabel: Bool) -> Entity {
//         let root = Entity()
//         let scale: Float = 0.45

//         let xArrow = ArrowFactory.makeArrow(color: .red)
//         xArrow.transform.rotation = simd_quatf(angle: .pi / 2, axis: [0, 1, 0])
//         xArrow.scale = SIMD3<Float>(repeating: scale)

//         let yArrow = ArrowFactory.makeArrow(color: .green)
//         yArrow.transform.rotation = simd_quatf(angle: -.pi / 2, axis: [1, 0, 0])
//         yArrow.scale = SIMD3<Float>(repeating: scale)

//         let zArrow = ArrowFactory.makeArrow(color: .blue)
//         zArrow.scale = SIMD3<Float>(repeating: scale)

//         root.addChild(xArrow)
//         root.addChild(yArrow)
//         root.addChild(zArrow)

//         if includeLabel {
//             addLabel(to: root, text: "foundationpose", position: [0, 0.12, 0])
//         }
//         return root
//     }

//     private func addLabel(to entity: Entity, text: String, position: SIMD3<Float>) {
//         let label = makeTextLabel(text: text)
//         label.position = position
//         entity.addChild(label)
//     }

//     private func makeTextLabel(text: String) -> ModelEntity {
//         let mesh: MeshResource
//         switch text {
//         case "aruco":
//             mesh = TextMeshCache.aruco
//         case "Realsense":
//             mesh = TextMeshCache.realsense
//         case "foundationpose":
//             mesh = TextMeshCache.foundationpose
//         default:
//             mesh = TextMeshCache.makeMesh(for: text)
//         }
//         let material = SimpleMaterial(color: .white, roughness: 0.4, isMetallic: false)
//         return ModelEntity(mesh: mesh, materials: [material])
//     }
// }

// private enum TextMeshCache {
//     static let aruco = makeMesh(for: "aruco")
//     static let realsense = makeMesh(for: "Realsense")
//     static let foundationpose = makeMesh(for: "foundationpose")

//     static func makeMesh(for text: String) -> MeshResource {
//         MeshResource.generateText(
//             text,
//             extrusionDepth: 0.002,
//             font: .systemFont(ofSize: 0.08),
//             containerFrame: .zero,
//             alignment: .center,
//             lineBreakMode: .byWordWrapping
//         )
//     }
// }

// extension simd_float4x4 {
//     init(translation: SIMD3<Float>) {
//         self = matrix_identity_float4x4
//         columns.3.x = translation.x
//         columns.3.y = translation.y
//         columns.3.z = translation.z
//     }
// }


import SwiftUI
import RealityKit
import simd
import UIKit
import ARKit
import Foundation

struct ImmersiveSpaceView: View {
    @EnvironmentObject private var appModel: AppModel
    @EnvironmentObject private var sensorModel: SensorDataModel
    @EnvironmentObject private var arucoStream: ArucoStreamModel
    @EnvironmentObject private var calibrationManager: CalibrationManager
    @EnvironmentObject private var rsPoseModel: RealSensePoseModel
    @EnvironmentObject private var foundationPoseModel: FoundationPoseModel

    @State private var boardAxes: Entity?
    @State private var worldAnchor: AnchorEntity?
    @State private var rsAxes: Entity?
    @State private var foundationAxes: Entity?
    @State private var labelsAdded = false

    // --- NEW: user gizmo + calibration bookkeeping
    @State private var gizmoAxes: Entity?
    @State private var gizmoWorldTransform: simd_float4x4?          // worldFromGizmo (aligned to board)
    @State private var lastWorldAruco: simd_float4x4?               // most recent worldFromAruco when visible
    @State private var statusText: String = ""

    var body: some View {
        RealityView { content in
            let anchor = AnchorEntity(.world(transform: matrix_identity_float4x4))
            content.add(anchor)
            worldAnchor = anchor

            let boardAxesEntity = makeBoardAxesEntity(includeLabel: false)
            boardAxesEntity.name = "BoardAxes"
            boardAxesEntity.isEnabled = false
            anchor.addChild(boardAxesEntity)
            boardAxes = boardAxesEntity

            let rsAxesEntity = makeRSEntity(includeLabel: false)
            rsAxesEntity.name = "RealSenseAxes"
            rsAxesEntity.isEnabled = false
            anchor.addChild(rsAxesEntity)
            rsAxes = rsAxesEntity

            let foundationAxesEntity = makeFoundationPoseEntity(includeLabel: false)
            foundationAxesEntity.name = "FoundationPoseAxes"
            foundationAxesEntity.isEnabled = false
            anchor.addChild(foundationAxesEntity)
            foundationAxes = foundationAxesEntity

            // --- NEW: gizmo axes entity (user anchor that should match board frame)
            let gizmoEntity = makeAxesEntity(length: 0.12)
            gizmoEntity.name = "GizmoAxes"
            gizmoEntity.isEnabled = false
            anchor.addChild(gizmoEntity)
            gizmoAxes = gizmoEntity

        } update: { _ in
            updateBoardAxes()
            updateCameraCalibrationIfPossible()
            updateGizmoAxes()
            updateRSPose()
            updateFoundationPose()
        }
        .overlay(alignment: .topLeading) {
            VStack(alignment: .leading, spacing: 10) {
                Text("Cam offset (deviceFromCamera translation)")
                    .font(.caption)

                Text(offsetString(CameraTransformUtils.estimatedCameraOffset))
                    .font(.system(.caption, design: .monospaced))

                if !statusText.isEmpty {
                    Text(statusText)
                        .font(.caption2)
                        .opacity(0.9)
                }

                HStack(spacing: 10) {
                    Button("Set gizmo = current board") {
                        if let lastWorldAruco {
                            gizmoWorldTransform = lastWorldAruco
                            statusText = "Gizmo set from current ArUco world pose."
                        } else {
                            statusText = "No ArUco pose yet. Show the board first."
                        }
                    }

                    Button("Clear gizmo") {
                        gizmoWorldTransform = nil
                        statusText = "Gizmo cleared."
                    }
                }

                HStack(spacing: 10) {
                    Button("Reset cam offset") {
                        // Reset back to whatever is currently in estimatedCameraOffset
                        CameraTransformUtils.resetCameraCalibration()
                        statusText = "Camera calibration reset."
                    }

                    Button("Calibrate once") {
                        calibrateOnce()
                    }
                }
            }
            .padding(12)
            .background(.ultraThinMaterial)
            .cornerRadius(12)
            .padding(12)
        }
        .task {
            await MainActor.run { appModel.setImmersiveSpacePresented(true) }
            await Task.yield()

            Task { @MainActor in
                guard !labelsAdded else { return }
                labelsAdded = true
                if let boardAxes {
                    addLabel(to: boardAxes, text: "aruco", position: [0, 0.14, 0])
                }
                if let rsAxes {
                    addLabel(to: rsAxes, text: "Realsense", position: [0, 0.12, 0])
                }
                if let foundationAxes {
                    addLabel(to: foundationAxes, text: "foundationpose", position: [0, 0.12, 0])
                }
                if let gizmoAxes {
                    addLabel(to: gizmoAxes, text: "gizmo", position: [0, 0.14, 0])
                }
            }

            Task(priority: .userInitiated) {
                await sensorModel.restartARKitTracking()
            }
        }
        .onDisappear {
            appModel.setImmersiveSpacePresented(false)
        }
    }

    // MARK: - ArUco & Board Axes

    private func updateBoardAxes() {
        guard let boardAxes else { return }

        guard let deviceTransform = sensorModel.latestDeviceTransform else {
            boardAxes.isEnabled = false
            return
        }

        if let T_camera_aruco = arucoStream.calibratedBoardTransform {
            // Transform to world space (worldFromAruco)
            let T_world_aruco = CameraTransformUtils.arucoPoseToWorld(
                cameraPose: T_camera_aruco,
                deviceTransform: deviceTransform
            )

            lastWorldAruco = T_world_aruco

            boardAxes.isEnabled = true
            boardAxes.transform = Transform(matrix: T_world_aruco)
        } else {
            boardAxes.isEnabled = false
        }
    }

    // MARK: - NEW: Gizmo display + calibration

    private func updateGizmoAxes() {
        guard let gizmoAxes else { return }
        guard let gizmoWorldTransform else {
            gizmoAxes.isEnabled = false
            return
        }
        gizmoAxes.isEnabled = true
        gizmoAxes.transform = Transform(matrix: gizmoWorldTransform)
    }

    private func calibrateOnce() {
        guard let gizmoWorldTransform else {
            statusText = "Place/set gizmo first."
            return
        }
        guard let deviceTransform = sensorModel.latestDeviceTransform else {
            statusText = "No deviceTransform yet."
            return
        }
        guard let T_camera_aruco = arucoStream.calibratedBoardTransform else {
            statusText = "No ArUco pose right now."
            return
        }

        CameraTransformUtils.calibrateCameraOffsetTranslationOnly(
            worldFromGizmo: gizmoWorldTransform,
            cameraFromAruco: T_camera_aruco,
            worldFromDevice: deviceTransform,
            smoothing: 1.0   // one-shot
        )

        statusText = "Calibrated once. New offset: \(offsetString(CameraTransformUtils.estimatedCameraOffset))"
    }

    private func updateCameraCalibrationIfPossible() {
        // Continuous refinement while board is visible AND gizmo is set.
        guard let gizmoWorldTransform else { return }
        guard let deviceTransform = sensorModel.latestDeviceTransform else { return }
        guard let T_camera_aruco = arucoStream.calibratedBoardTransform else { return }

        CameraTransformUtils.calibrateCameraOffsetTranslationOnly(
            worldFromGizmo: gizmoWorldTransform,
            cameraFromAruco: T_camera_aruco,
            worldFromDevice: deviceTransform,
            smoothing: 0.10
        )
    }

    private func offsetString(_ v: SIMD3<Float>) -> String {
        String(format: "x=% .4f  y=% .4f  z=% .4f", v.x, v.y, v.z)
    }

    // MARK: - Other poses

    private func updateRSPose() {
        guard let rsAxes else { return }
        if let matrix = rsPoseModel.rsPoseMatrix {
            rsAxes.isEnabled = true
            rsAxes.transform = Transform(matrix: matrix)
        } else {
            rsAxes.isEnabled = false
        }
    }

    private func updateFoundationPose() {
        guard let foundationAxes else { return }
        if let matrix = foundationPoseModel.poseMatrix {
            foundationAxes.isEnabled = true
            foundationAxes.transform = Transform(matrix: matrix)
        } else {
            foundationAxes.isEnabled = true
            foundationAxes.transform = Transform()
        }
    }

    // MARK: - Entities

    private func makeAxesEntity(length: Float) -> Entity {
        let entity = Entity()
        let radius: Float = 0.004

        let xMesh = MeshResource.generateBox(size: [length, radius, radius])
        let yMesh = MeshResource.generateBox(size: [radius, length, radius])
        let zMesh = MeshResource.generateBox(size: [radius, radius, length])

        let xMat = SimpleMaterial(color: .red, roughness: 0.2, isMetallic: false)
        let yMat = SimpleMaterial(color: .green, roughness: 0.2, isMetallic: false)
        let zMat = SimpleMaterial(color: .blue, roughness: 0.2, isMetallic: false)

        let xEntity = ModelEntity(mesh: xMesh, materials: [xMat])
        xEntity.position = [length / 2, 0, 0]

        let yEntity = ModelEntity(mesh: yMesh, materials: [yMat])
        yEntity.position = [0, length / 2, 0]

        let zEntity = ModelEntity(mesh: zMesh, materials: [zMat])
        zEntity.position = [0, 0, length / 2]

        entity.addChild(xEntity)
        entity.addChild(yEntity)
        entity.addChild(zEntity)
        return entity
    }

    private func makeBoardAxesEntity(includeLabel: Bool) -> Entity {
        let root = Entity()
        let scale: Float = 0.6

        let xArrow = ArrowFactory.makeArrow(color: .red)
        xArrow.transform.rotation = simd_quatf(angle: .pi / 2, axis: [0, 1, 0])
        xArrow.scale = SIMD3<Float>(repeating: scale)

        let yArrow = ArrowFactory.makeArrow(color: .green)
        yArrow.transform.rotation = simd_quatf(angle: -.pi / 2, axis: [1, 0, 0])
        yArrow.scale = SIMD3<Float>(repeating: scale)

        let zArrow = ArrowFactory.makeArrow(color: .blue)
        zArrow.scale = SIMD3<Float>(repeating: scale)

        root.addChild(xArrow)
        root.addChild(yArrow)
        root.addChild(zArrow)
        if includeLabel {
            addLabel(to: root, text: "aruco", position: [0, 0.14, 0])
        }
        return root
    }

    private func makeRSEntity(includeLabel: Bool) -> Entity {
        let root = Entity()
        let scale: Float = 0.45

        let xArrow = ArrowFactory.makeArrow(color: .red)
        xArrow.transform.rotation = simd_quatf(angle: .pi / 2, axis: [0, 1, 0])
        xArrow.scale = SIMD3<Float>(repeating: scale)

        let yArrow = ArrowFactory.makeArrow(color: .green)
        yArrow.transform.rotation = simd_quatf(angle: -.pi / 2, axis: [1, 0, 0])
        yArrow.scale = SIMD3<Float>(repeating: scale)

        let zArrow = ArrowFactory.makeArrow(color: .blue)
        zArrow.scale = SIMD3<Float>(repeating: scale)

        root.addChild(xArrow)
        root.addChild(yArrow)
        root.addChild(zArrow)

        if includeLabel {
            addLabel(to: root, text: "Realsense", position: [0, 0.12, 0])
        }
        return root
    }

    private func makeFoundationPoseEntity(includeLabel: Bool) -> Entity {
        let root = Entity()
        let scale: Float = 0.45

        let xArrow = ArrowFactory.makeArrow(color: .red)
        xArrow.transform.rotation = simd_quatf(angle: .pi / 2, axis: [0, 1, 0])
        xArrow.scale = SIMD3<Float>(repeating: scale)

        let yArrow = ArrowFactory.makeArrow(color: .green)
        yArrow.transform.rotation = simd_quatf(angle: -.pi / 2, axis: [1, 0, 0])
        yArrow.scale = SIMD3<Float>(repeating: scale)

        let zArrow = ArrowFactory.makeArrow(color: .blue)
        zArrow.scale = SIMD3<Float>(repeating: scale)

        root.addChild(xArrow)
        root.addChild(yArrow)
        root.addChild(zArrow)

        if includeLabel {
            addLabel(to: root, text: "foundationpose", position: [0, 0.12, 0])
        }
        return root
    }

    // MARK: - Labels

    private func addLabel(to entity: Entity, text: String, position: SIMD3<Float>) {
        let label = makeTextLabel(text: text)
        label.position = position
        entity.addChild(label)
    }

    private func makeTextLabel(text: String) -> ModelEntity {
        let mesh: MeshResource
        switch text {
        case "aruco":
            mesh = TextMeshCache.aruco
        case "Realsense":
            mesh = TextMeshCache.realsense
        case "foundationpose":
            mesh = TextMeshCache.foundationpose
        default:
            mesh = TextMeshCache.makeMesh(for: text)
        }
        let material = SimpleMaterial(color: .white, roughness: 0.4, isMetallic: false)
        return ModelEntity(mesh: mesh, materials: [material])
    }
}

private enum TextMeshCache {
    static let aruco = makeMesh(for: "aruco")
    static let realsense = makeMesh(for: "Realsense")
    static let foundationpose = makeMesh(for: "foundationpose")

    static func makeMesh(for text: String) -> MeshResource {
        MeshResource.generateText(
            text,
            extrusionDepth: 0.002,
            font: .systemFont(ofSize: 0.08),
            containerFrame: .zero,
            alignment: .center,
            lineBreakMode: .byWordWrapping
        )
    }
}

extension simd_float4x4 {
    init(translation: SIMD3<Float>) {
        self = matrix_identity_float4x4
        columns.3.x = translation.x
        columns.3.y = translation.y
        columns.3.z = translation.z
    }
}
