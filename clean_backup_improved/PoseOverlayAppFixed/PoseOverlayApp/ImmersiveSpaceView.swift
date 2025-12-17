import SwiftUI
import RealityKit
import simd
import UIKit

struct ImmersiveSpaceView: View {
    @EnvironmentObject private var settings: ArrowSettings
    @EnvironmentObject private var logs: LogStore
    @EnvironmentObject private var sensorModel: SensorDataModel
    @EnvironmentObject private var appModel: AppModel

    @State private var pollTask: Task<Void, Never>?
    @State private var overlayContainer: Entity?
    @State private var axesEntity: Entity?

    var body: some View {
        RealityView { content in
            let headAnchor = AnchorEntity(.head)
            content.add(headAnchor)

            let container = Entity()
            container.name = "PoseOverlayRoot"
            container.position = [0, 0, -1.0]
            headAnchor.addChild(container)
            overlayContainer = container

            let axes = makeAxesEntity()
            axes.name = "HeadAxes"
            headAnchor.addChild(axes)
            axesEntity = axes

            pollTask = makePollingTask(with: container)
            Task {
                await sensorModel.restartARKitTracking()
            }
        } update: { _ in
            if let axes = axesEntity {
                let orientation = sensorModel.headOrientation
                let rotation = simd_quatf(ix: Float(orientation.imag.x),
                                          iy: Float(orientation.imag.y),
                                          iz: Float(orientation.imag.z),
                                          r: Float(orientation.real))
                axes.transform.rotation = rotation
            }
        }
        .onAppear {
            appModel.setImmersiveSpacePresented(true)
            logs.add("Immersive space became visible")
        }
        .onDisappear {
            pollTask?.cancel()
            pollTask = nil
            overlayContainer = nil
            axesEntity = nil
            appModel.setImmersiveSpacePresented(false)
            logs.add("Immersive space dismissed")
        }
    }

    private func makePollingTask(with container: Entity) -> Task<Void, Never> {
        Task { [weak container] in
            while !Task.isCancelled {
                let context = await MainActor.run { (appModel.baseURL, appModel.selectedModel) }
                guard let container, let base = context.0 else {
                    try? await Task.sleep(nanoseconds: 2_000_000_000)
                    continue
                }
                do {
                    let transforms = try await PoseService.fetchTransforms(
                        baseURL: base,
                        modelName: context.1
                    )
                    await MainActor.run {
                        update(container: container, with: transforms)
                    }
                } catch {
                    logs.add("Pose fetch failed: \(error.localizedDescription)")
                }
                try? await Task.sleep(nanoseconds: 2_000_000_000)
            }
        }
    }

    @MainActor
    private func update(container: Entity, with transforms: [simd_float4x4]) {
        for child in container.children where child.name.hasPrefix("poseArrow_") {
            child.removeFromParent()
        }
        for (index, matrix) in transforms.enumerated() {
            let arrow = ArrowFactory.makeArrow(color: settings.color)
            arrow.name = "poseArrow_\(index)"
            arrow.transform = Transform(matrix: matrix)
            container.addChild(arrow)
        }
        logs.add("Rendered \(transforms.count) pose(s)")
    }

    private func makeAxesEntity() -> Entity {
        let entity = Entity()
        entity.name = "Axes"

        let length: Float = 0.4
        let radius: Float = 0.005

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
}
