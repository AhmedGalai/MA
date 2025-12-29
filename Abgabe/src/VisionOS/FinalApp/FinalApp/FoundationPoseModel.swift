import Foundation
import simd

@MainActor
final class FoundationPoseModel: ObservableObject {
    @Published var poseMatrix: simd_float4x4?
    @Published var rsPoseMatrix: simd_float4x4?
    @Published var lastUpdate: Date?
    @Published var lastMessage: String?
    @Published var rsLastMessage: String?

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
        let avpURL = baseURL.appendingPathComponent("get_foundationpose_pose")
        let rsURL = baseURL.appendingPathComponent("get_foundationpose_rs_pose")

        var avpPose: simd_float4x4?
        var avpMessage: String?
        var rsPose: simd_float4x4?
        var rsMessage: String?

        do {
            (avpPose, avpMessage) = try await fetchPose(url: avpURL)
        } catch {
            avpPose = nil
            avpMessage = error.localizedDescription
        }

        do {
            (rsPose, rsMessage) = try await fetchPose(url: rsURL)
        } catch {
            rsPose = nil
            rsMessage = error.localizedDescription
        }

        poseMatrix = avpPose
        lastMessage = avpMessage
        rsPoseMatrix = rsPose
        rsLastMessage = rsMessage
        lastUpdate = Date()
    }

    private func fetchPose(url: URL) async throws -> (simd_float4x4?, String?) {
        let (data, resp) = try await URLSession.shared.data(from: url)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        let decoded = try JSONDecoder().decode(FoundationPoseResponse.self, from: data)
        if let matrix = decoded.pose_matrix {
            return (MatrixUtils.convertOpenCVToRealityKit(MatrixUtils.simdMatrix(from: matrix)), nil)
        }
        return (nil, decoded.message ?? "FoundationPose not available")
    }
}

private struct FoundationPoseResponse: Decodable {
    let pose_matrix: [[Double]]?
    let message: String?
    let success: Bool?
}
