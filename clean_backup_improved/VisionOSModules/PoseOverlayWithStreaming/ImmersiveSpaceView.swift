import SwiftUI
import RealityKit
import simd
import UIKit

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
    @State private var anchorGizmo: Entity?
    @State private var rsAxes: Entity?
    @State private var foundationAxes: Entity?
    @State private var lastFoundationMessage: String?

    var body: some View {
        ZStack(alignment: .bottom) {
            RealityView { content in
                let anchor = AnchorEntity(.world(transform: matrix_identity_float4x4))
                content.add(anchor)
                worldAnchor = anchor

                let boardAxesEntity = makeBoardAxesEntity()
                boardAxesEntity.name = "BoardAxes"
                boardAxesEntity.isEnabled = false
                anchor.addChild(boardAxesEntity)
                boardAxes = boardAxesEntity

                let gizmo = makeAnchorGizmo()
                gizmo.name = "AnchorGizmo"
                anchor.addChild(gizmo)
                anchorGizmo = gizmo

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
                updateWorldAnchorFromSliders()
                updateRSPose()
                updateFoundationPose()
            }

            HStack {
                Button("Calibrate") {
                    let translation = sliderTranslation
                    let rotation = sliderRotation
                    let calibrationMatrix = simd_float4x4(translation: translation) * simd_float4x4(rotation)
                    calibrationManager.saveCalibration(transform: calibrationMatrix)
                }
                .padding()

                Button("Clear Calibration") {
                    calibrationManager.clearCalibration()
                    appModel.anchorTranslation = .zero
                    appModel.anchorEulerDegrees = .zero
                }
                .padding()
            }
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
        if let matrix = arucoStream.calibratedBoardTransform {
            boardAxes.isEnabled = true
            let userTransform = Transform(scale: .one, rotation: sliderRotation, translation: sliderTranslation)
            let combined = Transform(matrix: matrix * userTransform.matrix)
            boardAxes.transform = combined
        } else {
            boardAxes.isEnabled = false
        }
    }

    private func updateWorldAnchorFromSliders() {
        guard let worldAnchor else { return }
        worldAnchor.transform = Transform(scale: .one,
                                          rotation: sliderRotation,
                                          translation: sliderTranslation)
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
        } else {
            foundationAxes.isEnabled = false
            if let msg = foundationPoseModel.lastMessage, msg != lastFoundationMessage {
                lastFoundationMessage = msg
                logs.add("FoundationPose: \(msg)")
            }
        }
    }

    private var sliderTranslation: SIMD3<Float> {
        SIMD3<Float>(
            Float(appModel.anchorTranslation.x),
            Float(appModel.anchorTranslation.y),
            Float(appModel.anchorTranslation.z)
        )
    }

    private var sliderRotation: simd_quatf {
        let degrees = appModel.anchorEulerDegrees
        let radians = SIMD3<Double>(degrees.x, degrees.y, degrees.z) * (.pi / 180.0)
        let pitch = simd_quatf(angle: Float(radians.x), axis: [1, 0, 0])
        let yaw = simd_quatf(angle: Float(radians.y), axis: [0, 1, 0])
        let roll = simd_quatf(angle: Float(radians.z), axis: [0, 0, 1])
        return yaw * pitch * roll
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

    private func makeAnchorGizmo() -> Entity {
        let root = Entity()
        let axes = makeAxesEntity(length: 0.22)
        root.addChild(axes)

        let sphere = ModelEntity(
            mesh: .generateSphere(radius: 0.02),
            materials: [SimpleMaterial(color: .yellow, roughness: 0.4, isMetallic: false)]
        )
        root.addChild(sphere)

        let label = makeTextLabel(text: "aruco")
        label.position = [0, 0.06, 0]
        root.addChild(label)

        return root
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
