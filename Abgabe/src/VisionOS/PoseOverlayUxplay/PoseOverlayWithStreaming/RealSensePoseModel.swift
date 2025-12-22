import Foundation
import simd

@MainActor
final class RealSensePoseModel: ObservableObject {
    @Published var rsPoseMatrix: simd_float4x4?
    @Published var lastUpdate: Date?
    @Published var lastError: String?

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
            let url = baseURL.appendingPathComponent("get_rs_pose_in_avp")
            let (data, resp) = try await URLSession.shared.data(from: url)
            guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                throw URLError(.badServerResponse)
            }
            let decoded = try JSONDecoder().decode(RSPoseResponse.self, from: data)
            if let matrix = decoded.T_avp_rs {
                let simdMatrix = MatrixUtils.simdMatrix(from: matrix)
                rsPoseMatrix = MatrixUtils.convertOpenCVToRealityKit(simdMatrix)
                lastError = nil
                lastUpdate = Date()
            } else {
                lastError = decoded.message ?? "RS pose unavailable"
            }
        } catch {
            lastError = error.localizedDescription
        }
    }
}

private struct RSPoseResponse: Decodable {
    let T_avp_rs: [[Double]]?
    let message: String?
}
