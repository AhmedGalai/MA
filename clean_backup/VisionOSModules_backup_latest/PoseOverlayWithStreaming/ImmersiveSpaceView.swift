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
    @State private var containerAnchor: AnchorEntity?
    @State private var worldAnchor: AnchorEntity?
    @State private var rsPoseEntity: Entity?
    @State private var arucoPoseEntity: Entity?

    var body: some View {
        RealityView { content in
            let headAnchor = AnchorEntity(.head)
            headAnchor.name = "PoseOverlayHeadAnchor"
            headAnchor.position = [0, 0, 0]
            content.add(headAnchor)

            let container = Entity()
            container.name = "PoseOverlayRoot"
            container.position = [0, 0, -1.0]
            headAnchor.addChild(container)
            overlayContainer = container
            containerAnchor = headAnchor

            let axes = makeAxesEntity()
            axes.name = "HeadAxes"
            headAnchor.addChild(axes)
            axesEntity = axes

            let world = AnchorEntity(world: .zero)
            content.add(world)
            worldAnchor = world
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
            Task {
                await sensorModel.restartARKitTracking()
            }
        }
        .onDisappear {
            pollTask?.cancel()
            pollTask = nil
            overlayContainer = nil
            axesEntity = nil
            containerAnchor = nil
            worldAnchor = nil
            rsPoseEntity = nil
            arucoPoseEntity = nil
            appModel.setImmersiveSpacePresented(false)
            logs.add("Immersive space dismissed")
        }
        .task(id: overlayContainer?.id) {
            pollTask?.cancel()
            guard let container = overlayContainer else { return }
            pollTask = makePollingTask(with: container)
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
                    async let debugPoses = fetchDebugPoses(baseURL: base)
                    await MainActor.run {
                        update(container: container, with: transforms)
                    }
                    if let debug = try? await debugPoses {
                        await MainActor.run {
                            updateDebugPoses(worldRS: debug.rsPose, aruco: debug.arucoPose)
                        }
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

    @MainActor
    private func updateDebugPoses(worldRS: simd_float4x4?, aruco: simd_float4x4?) {
        guard let anchor = worldAnchor else { return }
        func place(_ matrix: simd_float4x4, name: String, color: UIColor) -> Entity {
            let entity = makePoseMarker(color: color)
            entity.name = name
            entity.transform = Transform(matrix: matrix)
            return entity
        }

        if let rsEntity = rsPoseEntity { rsEntity.removeFromParent() }
        if let arucoEntity = arucoPoseEntity { arucoEntity.removeFromParent() }

        if let worldRS {
            let e = place(worldRS, name: "rsPose", color: .yellow)
            anchor.addChild(e)
            rsPoseEntity = e
        } else {
            rsPoseEntity = nil
        }

        if let aruco {
            let e = place(aruco, name: "arucoPose", color: .cyan)
            anchor.addChild(e)
            arucoPoseEntity = e
        } else {
            arucoPoseEntity = nil
        }
    }

    private func makePoseMarker(color: UIColor) -> Entity {
        let root = Entity()
        let arrow = ArrowFactory.makeArrow(color: Color(uiColor: color))
        arrow.position = [0, 0.05, 0]
        root.addChild(arrow)

        let cube = makeWireCube(size: 0.12, thickness: 0.003, color: color)
        root.addChild(cube)
        return root
    }

    private func makeWireCube(size: Float, thickness: Float, color: UIColor) -> Entity {
        let half = size / 2
        let positions: [(SIMD3<Float>, SIMD3<Float>)] = [
            (.init(-half, -half, -half), .init( half, -half, -half)),
            (.init( half, -half, -half), .init( half, -half,  half)),
            (.init( half, -half,  half), .init(-half, -half,  half)),
            (.init(-half, -half,  half), .init(-half, -half, -half)),
            (.init(-half,  half, -half), .init( half,  half, -half)),
            (.init( half,  half, -half), .init( half,  half,  half)),
            (.init( half,  half,  half), .init(-half,  half,  half)),
            (.init(-half,  half,  half), .init(-half,  half, -half)),
            (.init(-half, -half, -half), .init(-half,  half, -half)),
            (.init( half, -half, -half), .init( half,  half, -half)),
            (.init( half, -half,  half), .init( half,  half,  half)),
            (.init(-half, -half,  half), .init(-half,  half,  half))
        ]

        let material = SimpleMaterial(color: .init(Color(uiColor: color)), isMetallic: false)
        let root = Entity()

        for (start, end) in positions {
            let delta = end - start
            let length = simd_length(delta)
            let box = MeshResource.generateBox(size: [thickness, length, thickness])
            let model = ModelEntity(mesh: box, materials: [material])
            model.position = (start + end) / 2
            model.look(at: end, from: start, relativeTo: nil as Entity?)
            root.addChild(model)
        }
        return root
    }

    private func fetchDebugPoses(baseURL: URL) async throws -> (rsPose: simd_float4x4?, arucoPose: simd_float4x4?) {
        struct TransformResponse: Decodable {
            let T_world_rs: [[Double]]?
            let T_world_aruco: [[Double]]?
        }

        let url = baseURL.appendingPathComponent("get_transformation")
        let (data, _) = try await URLSession.shared.data(from: url)
        let response = try JSONDecoder().decode(TransformResponse.self, from: data)

        let rsMatrix = response.T_world_rs.flatMap(MatrixUtils.simdMatrix(from:)).map(MatrixUtils.convertOpenCVToRealityKit)
        let arucoMatrix = response.T_world_aruco.flatMap(MatrixUtils.simdMatrix(from:)).map(MatrixUtils.convertOpenCVToRealityKit)
        return (rsMatrix, arucoMatrix)
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
