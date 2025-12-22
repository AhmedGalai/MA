import SwiftUI
import RealityKit
import simd
import UIKit
import ARKit

struct ImmersiveSpaceView: View {
    @EnvironmentObject private var appModel: AppModel
    @EnvironmentObject private var sensorModel: SensorDataModel
    @EnvironmentObject private var arucoStream: ArucoStreamModel
    @EnvironmentObject private var calibrationManager: CalibrationManager
    @EnvironmentObject private var rsPoseModel: RealSensePoseModel
    @EnvironmentObject private var foundationPoseModel: FoundationPoseModel
    @EnvironmentObject private var logs: LogStore

    @State private var boardAxes: Entity?
    @State private var worldAnchor: AnchorEntity?
    @State private var rsAxes: Entity?
    @State private var foundationAxes: Entity?
    @State private var lastFoundationMessage: String?
    @State private var foundationFallbackActive = false
    @State private var frameCounter: Int = 0

    var body: some View {
        RealityView { content in
            let anchor = AnchorEntity(.world(transform: matrix_identity_float4x4))
            content.add(anchor)
            worldAnchor = anchor

            let boardAxesEntity = makeBoardAxesEntity()
            boardAxesEntity.name = "BoardAxes"
            boardAxesEntity.isEnabled = false
            anchor.addChild(boardAxesEntity)
            boardAxes = boardAxesEntity

            let rsAxesEntity = makeRSEntity()
            rsAxesEntity.name = "RealSenseAxes"
            rsAxesEntity.isEnabled = false
            anchor.addChild(rsAxesEntity)
            rsAxes = rsAxesEntity

            let foundationAxesEntity = makeFoundationPoseEntity()
            foundationAxesEntity.name = "FoundationPoseAxes"
            foundationAxesEntity.isEnabled = false
            anchor.addChild(foundationAxesEntity)
            foundationAxes = foundationAxesEntity
        } update: { _ in
            updateBoardAxes()
            updateRSPose()
            updateFoundationPose()
            frameCounter += 1
        }
        .task {
            appModel.setImmersiveSpacePresented(true)
            await sensorModel.restartARKitTracking()
        }
        .onDisappear {
            appModel.setImmersiveSpacePresented(false)
        }
    }
    
    private func updateBoardAxes() {
        guard let boardAxes else { return }

        // Use cached device transform from the sensor model to avoid re-querying ARKit here.
        guard let deviceTransform = sensorModel.latestDeviceTransform else {
            boardAxes.isEnabled = false
            if frameCounter % 60 == 0 {
                logs.add("⚠️ Waiting for device anchor")
            }
            return
        }

        // Check if ArUco is currently detected
        if let T_camera_aruco = arucoStream.calibratedBoardTransform {
            // ArUco detected - compute and store device-to-aruco transform for future tracking
            let T_device_aruco = CameraTransformUtils.computeArucoToDeviceTransform(
                cameraPose: T_camera_aruco,
                deviceTransform: deviceTransform
            )
            arucoStream.deviceToArucoTransform = T_device_aruco

            // Transform to world space
            let T_world_aruco = CameraTransformUtils.arucoPoseToWorld(
                cameraPose: T_camera_aruco,
                deviceTransform: deviceTransform
            )

            boardAxes.isEnabled = true
            boardAxes.transform = Transform(matrix: T_world_aruco)
            arucoStream.isTracking = true

            // Log position periodically
            if frameCounter % 60 == 0 {
                let pos = T_world_aruco.columns.3
                logs.add("ArUco world pos: [\(String(format: "%.3f", pos.x)), \(String(format: "%.3f", pos.y)), \(String(format: "%.3f", pos.z))]")
            }
        } else if let T_device_aruco = arucoStream.deviceToArucoTransform {
            // ArUco not visible - use continuous tracking with stored transform
            let T_world_aruco = CameraTransformUtils.estimateArucoWorldPose(
                deviceToAruco: T_device_aruco,
                deviceTransform: deviceTransform
            )

            boardAxes.isEnabled = true
            boardAxes.transform = Transform(matrix: T_world_aruco)
            // Keep isTracking true to indicate we're still showing the pose

            if frameCounter % 120 == 0 {
                logs.add("📍 Continuous tracking (ArUco not visible)")
            }
        } else {
            // No ArUco detected and no stored transform - disable
            boardAxes.isEnabled = false
            arucoStream.isTracking = false
        }
    }

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
            foundationFallbackActive = false
        } else {
            foundationAxes.isEnabled = true
            foundationAxes.transform = Transform()
            if !foundationFallbackActive {
                foundationFallbackActive = true
                logs.add("FoundationPose: using anchor fallback")
            }
            if let msg = foundationPoseModel.lastMessage, msg != lastFoundationMessage {
                lastFoundationMessage = msg
                logs.add("FoundationPose: \(msg)")
            }
        }
    }

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

    private func makeBoardAxesEntity() -> Entity {
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
        let label = makeTextLabel(text: "aruco")
        label.position = [0, 0.14, 0]
        root.addChild(label)
        return root
    }

    private func makeRSEntity() -> Entity {
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

        let label = makeTextLabel(text: "Realsense")
        label.position = [0, 0.12, 0]
        root.addChild(label)
        return root
    }

    private func makeFoundationPoseEntity() -> Entity {
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

        let label = makeTextLabel(text: "foundationpose")
        label.position = [0, 0.12, 0]
        root.addChild(label)
        return root
    }

    private func makeTextLabel(text: String) -> ModelEntity {
        let mesh = MeshResource.generateText(
            text,
            extrusionDepth: 0.002,
            font: .systemFont(ofSize: 0.08),
            containerFrame: .zero,
            alignment: .center,
            lineBreakMode: .byWordWrapping
        )
        let material = SimpleMaterial(color: .white, roughness: 0.4, isMetallic: false)
        return ModelEntity(mesh: mesh, materials: [material])
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
