import Foundation

struct HeadPosePayload: Encodable, Sendable {
    let position: [Double]
    let rotation: [Double]
    let quaternion: [Double]
    let timestamp: Double
    let confidence: Double
    let metadata: [String: String]
}

enum HeadPoseService {
    enum UploadError: Error, LocalizedError {
        case badResponse(Int, String)
        var errorDescription: String? {
            switch self {
            case .badResponse(let status, let body):
                return "Head pose upload failed (\(status)): \(body)"
            }
        }
    }

    static func send(baseURL: URL, payload: HeadPosePayload) async throws {
        let url = baseURL.appendingPathComponent("head_pose")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(payload)

        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse,
              (200..<300).contains(http.statusCode) else {
            let status = (resp as? HTTPURLResponse)?.statusCode ?? -1
            let snippet = String(data: data, encoding: .utf8) ?? "<non-utf8>"
            throw UploadError.badResponse(status, snippet)
        }
    }
}
