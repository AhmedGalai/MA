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
//     @EnvironmentObject private var calibrationModel: CalibrationModel
//     @EnvironmentObject private var logStore: LogStore

//     @State private var boardAxes: Entity?
//     @State private var worldAnchor: AnchorEntity?
//     @State private var rsAxes: Entity?
//     @State private var foundationAxes: Entity?
//     @State private var gizmo: Entity?
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
            
//             let gizmoEntity = makeGizmoEntity()
//             gizmoEntity.name = "Gizmo"
//             anchor.addChild(gizmoEntity)
//             gizmo = gizmoEntity

//         } update: { _ in
//             updateBoardAxes()
//             updateRSPose()
//             updateFoundationPose()
            
//             if let gizmo {
//                 gizmo.transform.matrix = calibrationModel.transform
//             }
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
//         .onReceive(NotificationCenter.default.publisher(for: .saveCalibration)) { _ in
//             saveCalibration()
//         }
//     }

//     private func saveCalibration() {
//         guard let gizmo = gizmo,
//               let T_camera_object = arucoStream.latestPose?.realityTransform,
//               let T_world_device = sensorModel.latestDeviceTransform else {
//             logStore.add("Missing data for calibration")
//             return
//         }

//         let T_world_gizmo = gizmo.transform.matrix
        
//         let T_device_world = T_world_device.inverse
//         let T_object_camera = T_camera_object.inverse
        
//         let newOffset = T_device_world * T_world_gizmo * T_object_camera
        
//         calibrationManager.saveCalibration(transform: newOffset)
//         logStore.add("Saved new camera offset to UserDefaults")
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

//     private func makeGizmoEntity() -> Entity {
//         let root = Entity()
//         let scale: Float = 0.5

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
        
//         addLabel(to: root, text: "Gizmo", position: [0, 0.12, 0])

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
//         case "Gizmo":
//             mesh = TextMeshCache.gizmo
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
//     static let gizmo = makeMesh(for: "Gizmo")

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

struct ImmersiveSpaceView: View {
    @EnvironmentObject private var appModel: AppModel
    @EnvironmentObject private var sensorModel: SensorDataModel
    @EnvironmentObject private var arucoStream: ArucoStreamModel
    @EnvironmentObject private var calibrationManager: CalibrationManager
    @EnvironmentObject private var rsPoseModel: RealSensePoseModel
    @EnvironmentObject private var foundationPoseModel: FoundationPoseModel
    @EnvironmentObject private var calibrationModel: CalibrationModel
    @EnvironmentObject private var logStore: LogStore

    @State private var boardAxes: Entity?
    @State private var worldAnchor: AnchorEntity?
    @State private var rsAxes: Entity?
    @State private var foundationAxesAVP: Entity?
    @State private var foundationAxesRS: Entity?
    @State private var gizmo: Entity?
    @State private var labelsAdded = false
    @State private var smoothedArucoTransform: simd_float4x4?

    var body: some View {
        RealityView { content in
            // IMPORTANT: make sure CameraTransformUtils can see the calibration manager
            CameraTransformUtils.setCalibrationManager(calibrationManager)

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

            let foundationAxesAVPEntity = makeFoundationPoseEntity(label: "foundationpose_avp", includeLabel: false)
            foundationAxesAVPEntity.name = "FoundationPoseAxesAVP"
            foundationAxesAVPEntity.isEnabled = false
            anchor.addChild(foundationAxesAVPEntity)
            foundationAxesAVP = foundationAxesAVPEntity

            let foundationAxesRSEntity = makeFoundationPoseEntity(label: "foundationpose_rs", includeLabel: false)
            foundationAxesRSEntity.name = "FoundationPoseAxesRS"
            foundationAxesRSEntity.isEnabled = false
            anchor.addChild(foundationAxesRSEntity)
            foundationAxesRS = foundationAxesRSEntity

            let gizmoEntity = makeGizmoEntity()
            gizmoEntity.name = "Gizmo"
            anchor.addChild(gizmoEntity)
            gizmo = gizmoEntity

        } update: { _ in
            updateBoardAxes()
            updateRSPose()
            updateFoundationPose()

            // sliders drive gizmo transform (local relative to worldAnchor)
            if let gizmo {
                gizmo.transform.matrix = calibrationModel.transform
            }
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
                if let foundationAxesAVP {
                    addLabel(to: foundationAxesAVP, text: "foundationpose_avp", position: [0, 0.12, 0])
                }
                if let foundationAxesRS {
                    addLabel(to: foundationAxesRS, text: "foundationpose_rs", position: [0, 0.12, 0])
                }
            }

            Task(priority: .userInitiated) {
                await sensorModel.restartARKitTracking()
            }
        }
        .onDisappear {
            appModel.setImmersiveSpacePresented(false)
        }
        .onReceive(NotificationCenter.default.publisher(for: .saveCalibration)) { _ in
            saveCalibration()
        }
    }

    private func saveCalibration() {
        guard let gizmo = gizmo else {
            logStore.add("Missing gizmo for calibration")
            return
        }
        guard let cameraFromAruco = arucoStream.calibratedBoardTransform else {
            logStore.add("Missing ArUco pose (calibratedBoardTransform) for calibration")
            return
        }
        guard let worldFromDevice = sensorModel.latestDeviceTransform else {
            logStore.add("Missing device transform for calibration")
            return
        }

        // Use WORLD transform (robust even if parent anchor moves)
        let worldFromGizmo = gizmo.transformMatrix(relativeTo: nil)

        // deviceFromCamera = inv(worldFromDevice) * worldFromGizmo * inv(cameraFromAruco)
        let newOffset = CameraTransformUtils.computeDeviceFromCameraOffset(
            worldFromGizmo: worldFromGizmo,
            cameraFromAruco: cameraFromAruco,
            worldFromDevice: worldFromDevice
        )

        calibrationManager.saveCalibration(transform: newOffset)
        logStore.add("Saved new camera offset to UserDefaults")
    }

    private func updateBoardAxes() {
        guard let boardAxes else { return }
        guard let deviceTransform = sensorModel.latestDeviceTransform else {
            boardAxes.isEnabled = false
            smoothedArucoTransform = nil
            return
        }

        if let cameraFromAruco = arucoStream.calibratedBoardTransform {
            let worldFromAruco = CameraTransformUtils.arucoPoseToWorld(
                cameraPose: cameraFromAruco,
                deviceTransform: deviceTransform
            )
            let filtered = smoothTransform(
                previous: smoothedArucoTransform,
                target: worldFromAruco,
                alpha: calibrationModel.arucoSmoothingAlpha
            )
            smoothedArucoTransform = filtered

            boardAxes.isEnabled = true
            boardAxes.transform = Transform(matrix: filtered)
        } else {
            boardAxes.isEnabled = false
            smoothedArucoTransform = nil
        }
    }

    private func updateRSPose() {
        guard let rsAxes else { return }
        guard let deviceTransform = sensorModel.latestDeviceTransform,
              let cameraFromRS = rsPoseModel.rsPoseMatrix else {
            rsAxes.isEnabled = false
            return
        }

        let worldFromCamera = CameraTransformUtils.cameraToWorldTransform(from: deviceTransform)
        let worldFromRS = worldFromCamera * cameraFromRS
        rsAxes.isEnabled = true
        rsAxes.transform = Transform(matrix: worldFromRS)
    }

    private func updateFoundationPose() {
        updateFoundationPoseAVP()
        updateFoundationPoseRS()
    }

    private func updateFoundationPoseAVP() {
        guard let foundationAxesAVP else { return }
        guard let deviceTransform = sensorModel.latestDeviceTransform else {
            foundationAxesAVP.isEnabled = false
            return
        }
        guard let cameraFromObject = foundationPoseModel.poseMatrix else {
            return
        }

        let worldFromCamera = CameraTransformUtils.cameraToWorldTransform(from: deviceTransform)
        let worldFromObject = worldFromCamera * cameraFromObject
        foundationAxesAVP.isEnabled = true
        foundationAxesAVP.transform = Transform(matrix: worldFromObject)
    }

    private func updateFoundationPoseRS() {
        guard let foundationAxesRS else { return }
        guard let deviceTransform = sensorModel.latestDeviceTransform else {
            foundationAxesRS.isEnabled = false
            return
        }
        guard let cameraFromRS = rsPoseModel.rsPoseMatrix,
              let rsCameraFromObject = foundationPoseModel.rsPoseMatrix else {
            return
        }

        let cameraFromObject = cameraFromRS * rsCameraFromObject
        let worldFromCamera = CameraTransformUtils.cameraToWorldTransform(from: deviceTransform)
        let worldFromObject = worldFromCamera * cameraFromObject
        foundationAxesRS.isEnabled = true
        foundationAxesRS.transform = Transform(matrix: worldFromObject)
    }

    private func smoothTransform(previous: simd_float4x4?, target: simd_float4x4, alpha: Float) -> simd_float4x4 {
        guard let previous else { return target }
        let prevRot = simd_quatf(previous)
        let targetRot = simd_quatf(target)
        let rot = simd_slerp(prevRot, targetRot, alpha)

        let prevPos = SIMD3<Float>(previous.columns.3.x, previous.columns.3.y, previous.columns.3.z)
        let targetPos = SIMD3<Float>(target.columns.3.x, target.columns.3.y, target.columns.3.z)
        let pos = prevPos + (targetPos - prevPos) * alpha

        var result = simd_float4x4(rot)
        result.columns.3 = SIMD4<Float>(pos.x, pos.y, pos.z, 1.0)
        return result
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

    private func makeFoundationPoseEntity(label: String, includeLabel: Bool) -> Entity {
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
            addLabel(to: root, text: label, position: [0, 0.12, 0])
        }
        return root
    }

    private func makeGizmoEntity() -> Entity {
        let root = Entity()
        let scale: Float = 0.5

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

        addLabel(to: root, text: "Gizmo", position: [0, 0.12, 0])

        return root
    }

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
        case "foundationpose_avp":
            mesh = TextMeshCache.foundationposeAVP
        case "foundationpose_rs":
            mesh = TextMeshCache.foundationposeRS
        case "Gizmo":
            mesh = TextMeshCache.gizmo
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
    static let foundationposeAVP = makeMesh(for: "foundationpose_avp")
    static let foundationposeRS = makeMesh(for: "foundationpose_rs")
    static let gizmo = makeMesh(for: "Gizmo")

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
