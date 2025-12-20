import SwiftUI
import RealityKit
import simd

struct ImmersiveSpaceView: View {
    @EnvironmentObject private var appModel: AppModel
    @EnvironmentObject private var sensorModel: SensorDataModel
    @EnvironmentObject private var arucoStream: ArucoStreamModel

    @State private var boardAxes: Entity?
    @State private var headAxes: Entity?
    @State private var headAnchor: AnchorEntity?

    var body: some View {
        RealityView { content in
            let anchor = AnchorEntity(.head)
            //let anchor = AnchorEntity(.camera)
            content.add(anchor)
            headAnchor = anchor

            let boardAxesEntity = makeBoardAxesEntity()
            boardAxesEntity.name = "BoardAxes"
            boardAxesEntity.isEnabled = false
            anchor.addChild(boardAxesEntity)
            boardAxes = boardAxesEntity

            let headAxesEntity = makeAxesEntity(length: 0.12)
            headAxesEntity.name = "HeadAxes"
            headAxesEntity.isEnabled = false
            anchor.addChild(headAxesEntity)
            headAxes = headAxesEntity
        } update: { _ in
            updateBoardAxes()
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
        if let matrix = arucoStream.latestBoardTransform {
            boardAxes.isEnabled = true
            boardAxes.transform = Transform(matrix: matrix)
        } else {
            boardAxes.isEnabled = false
        }
    }

    private func updateHeadAxes() {
        guard let headAxes else { return }
        let orientation = sensorModel.headOrientation
        headAxes.transform.rotation = simd_quatf(
            ix: Float(orientation.imag.x),
            iy: Float(orientation.imag.y),
            iz: Float(orientation.imag.z),
            r: Float(orientation.real)
        )
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
