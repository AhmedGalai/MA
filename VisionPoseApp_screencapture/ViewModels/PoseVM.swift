import SwiftUI
import CoreGraphics

@MainActor
final class PoseVM: ObservableObject {
    // Screen capture
    @Published var preview: CGImage?
    // Thumbs
    @Published var leftSnap: CGImage?
    @Published var rightSnap: CGImage?
    // Outputs
    @Published var depthImage: CGImage?
    @Published var overlayImage: CGImage?
    @Published var maskedImage: CGImage?
    // Models/UI
    @Published var modelNames: [String] = []
    @Published var selectedModel: String = ""
    @Published var rateHz: Double = 1.0
    @Published var log: String = ""
    // ROI
    @Published var roiCenter: CGPoint?
    @Published var roiRadius: CGFloat?

    // Intrinsics
    var K: [[Double]] = [[700,0,320],[0,700,240],[0,0,1]]

    // Mesh
    var mesh = MeshData()

    // Loop
    private var timer: Timer?
    private var running = false

    func startCapture() {
        ScreenCapture.shared.start()
    }

    func stopCapture() {
        ScreenCapture.shared.stop()
    }

    func bindCapture(_ img: CGImage?) {
        self.preview = img
    }

    func appendLog(_ s: String) {
        log.append("[\(Date().formatted(date: .omitted, time: .standard))] \(s)\n")
    }

    func fetchModels() async {
        do {
            let names = try await APIClient.shared.fetchModels()
            self.modelNames = names.sorted()
            if selectedModel.isEmpty || !modelNames.contains(selectedModel) {
                selectedModel = modelNames.first ?? ""
            }
            if !selectedModel.isEmpty {
                try await fetchMeshForSelected()
            }
            appendLog("Models: \(modelNames.joined(separator: ", "))")
        } catch {
            appendLog("fetchModels error: \(error)")
        }
    }

    func fetchMeshForSelected() async throws {
        guard !selectedModel.isEmpty else { return }
        let data = try await APIClient.shared.fetchModelMesh(name: selectedModel)
        do {
            self.mesh = try parsePLYAscii(data)
            appendLog("Loaded mesh \(selectedModel) V=\(mesh.vertices.count) E=\(mesh.edges.count)")
        } catch {
            appendLog("PLY parse failed (\(selectedModel)): \(error)")
            throw error
        }
    }

    func captureLeft() {
        guard let img = preview else { return }
        leftSnap = img
    }

    func captureRight() {
        guard let img = preview else { return }
        rightSnap = img
    }

    func sendIntrinsics() async {
        guard let L = leftSnap, let R = rightSnap,
              let lj = ImageUtils.jpegData(from: L), let rj = ImageUtils.jpegData(from: R) else {
            appendLog("Capture left and right first")
            return
        }
        do {
            let Kresp = try await APIClient.shared.postIntrinsics(leftJPEG: lj, rightJPEG: rj)
            self.K = Kresp
            appendLog("Got intrinsics K")
        } catch {
            appendLog("intrinsics error: \(error)")
        }
    }

    func startEstimation() {
        guard !running else { return }
        running = true
        let interval = max(0.1, 1.0 / rateHz)
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            Task { await self?.tick() }
        }
        appendLog("Estimation started @\(rateHz, specifier: "%.2f") Hz")
    }

    func stopEstimation() {
        running = false
        timer?.invalidate()
        timer = nil
        appendLog("Estimation stopped")
    }

    private func currentMaskPNG(for width: Int, height: Int) -> Data? {
        guard let c = roiCenter, let r = roiRadius else { return nil }
        return ImageUtils.makeMaskPNG(width: width, height: height, center: c, radius: r)
    }

    private func maskedPreview(_ img: CGImage) -> CGImage? {
        let m = currentMaskPNG(for: img.width, height: img.height)
        return ImageUtils.applyMask(rgb: img, maskPNG: m)
    }

    private func overlay(_ img: CGImage, with pose: PoseResponse) -> CGImage? {
        var out: CGImage? = img
        for T in pose.transformation_matrix {
            out = OverlayRenderer.drawOverlay(base: out ?? img, mesh: mesh, K: K, T: T, edgeColor: .systemGreen, thickness: 2)
        }
        return out
    }

    private func tick() async {
        guard running, let frame = preview,
              let rgbJPEG = ImageUtils.jpegData(from: frame) else { return }
        do {
            // depth
            let depthPNG = try await APIClient.shared.postDepth(rgbJPEG: rgbJPEG)
            if let dimg = ImageUtils.cgImage(from: depthPNG) { self.depthImage = dimg }
            // masked preview
            self.maskedImage = self.maskedPreview(frame)
            // pose
            let maskPNG = currentMaskPNG(for: frame.width, height: frame.height)
            let pose = try await APIClient.shared.postPose(K: K, rgbJPEG: rgbJPEG, depthPNG: depthPNG, maskPNG: maskPNG, modelName: selectedModel, depthscale: 0.001)
            if let over = overlay(frame, with: pose) { self.overlayImage = over }
            appendLog("Pose updated")
        } catch {
            appendLog("tick error: \(error)")
        }
    }
}

