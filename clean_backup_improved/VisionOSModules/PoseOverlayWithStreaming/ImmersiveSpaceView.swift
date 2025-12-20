import SwiftUI
import RealityKit
import simd

struct ImmersiveSpaceView: View {
    @EnvironmentObject private var appModel: AppModel
    @EnvironmentObject private var sensorModel: SensorDataModel
    @EnvironmentObject private var arucoStream: ArucoStreamModel
    @EnvironmentObject private var calibrationManager: CalibrationManager

    @State private var boardAxes: Entity?
    @State private var worldAnchor: AnchorEntity?

    @State private var userTranslation: SIMD3<Float> = .zero
    @State private var userRotation: simd_quatf = simd_quatf(angle: 0, axis: [0, 1, 0])

    var body: some View {
        ZStack(alignment: .bottom) {
            RealityView { content in
                let anchor = AnchorEntity(.world)
                content.add(anchor)
                worldAnchor = anchor

                let boardAxesEntity = makeBoardAxesEntity()
                boardAxesEntity.name = "BoardAxes"
                boardAxesEntity.isEnabled = false
                anchor.addChild(boardAxesEntity)
                boardAxes = boardAxesEntity
            } update: { _ in
                updateHeadPose()
                updateBoardAxes()
            }
            .gesture(
                DragGesture()
                    .onChanged { value in
                        let translation = value.translation
                        userTranslation.x += Float(translation.width) * 0.001
                        userTranslation.y -= Float(translation.height) * 0.001
                    }
            )

            HStack {
                Button("Calibrate") {
                    let calibrationMatrix = simd_float4x4(translation: userTranslation) * simd_float4x4(userRotation)
                    calibrationManager.saveCalibration(transform: calibrationMatrix)
                }
                .padding()

                Button("Clear Calibration") {
                    calibrationManager.clearCalibration()
                    userTranslation = .zero
                    userRotation = simd_quatf(angle: 0, axis: [0, 1, 0])
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
    
    private func updateHeadPose() {
        guard let worldAnchor else { return }
        let headPose = Transform(
            scale: .one,
            rotation: sensorModel.headOrientation,
            translation: sensorModel.headPosition
        )
        worldAnchor.transform = headPose.inverse
    }

    private func updateBoardAxes() {
        guard let boardAxes else { return }
        if let matrix = arucoStream.calibratedBoardTransform {
            boardAxes.isEnabled = true
            let userTransform = Transform(scale: .one, rotation: userRotation, translation: userTranslation)
            boardAxes.transform = Transform(matrix: matrix) * userTransform
        } else {
            boardAxes.isEnabled = false
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
        return root
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
