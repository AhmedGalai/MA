import Foundation
import simd

@MainActor
final class FoundationPoseModel: ObservableObject {
    @Published var poseMatrix: simd_float4x4?
    @Published var lastUpdate: Date?
    @Published var lastMessage: String?

    private var baseURL: URL?
    private var pollTask: Task<Void, Never>?

    func updateBaseURL(_ url: URL?) {
        baseURL = url
        if pollTask != nil {
            startPolling()
        }
    }

    func startPolling() {
        pollTask?.cancel()
        guard baseURL != nil else { return }
        pollTask = Task { await pollLoop() }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    private func pollLoop() async {
        while !Task.isCancelled {
            await fetchOnce()
            try? await Task.sleep(for: .seconds(1.0))
        }
    }

    private func fetchOnce() async {
        guard let baseURL else { return }
        do {
            let url = baseURL.appendingPathComponent("get_foundationpose_pose")
            let (data, resp) = try await URLSession.shared.data(from: url)
            guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                throw URLError(.badServerResponse)
            }
            let decoded = try JSONDecoder().decode(FoundationPoseResponse.self, from: data)
            if let matrix = decoded.pose_matrix {
                poseMatrix = MatrixUtils.convertOpenCVToRealityKit(MatrixUtils.simdMatrix(from: matrix))
                lastMessage = nil
            } else {
                poseMatrix = nil
                lastMessage = decoded.message ?? "FoundationPose not available"
            }
            lastUpdate = Date()
        } catch {
            poseMatrix = nil
            lastMessage = error.localizedDescription
        }
    }
}

private struct FoundationPoseResponse: Decodable {
    let pose_matrix: [[Double]]?
    let message: String?
    let success: Bool?
}
