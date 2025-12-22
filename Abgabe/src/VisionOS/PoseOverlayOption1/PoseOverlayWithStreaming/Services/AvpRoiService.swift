import Foundation

enum AvpRoiService {
    struct APIError: Error, LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    static func update(baseURL: URL, roi: AvpRoiConfigPayload) async throws {
        let url = baseURL.appendingPathComponent("avp_roi_config")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(roi)
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse else {
            throw APIError(message: "Failed to post AVP ROI config (invalid response)")
        }
        guard (200..<300).contains(http.statusCode) else {
            let snippet = String(data: data, encoding: .utf8) ?? "<non-utf8>"
            throw APIError(message: "Failed to post AVP ROI config (status \(http.statusCode)): \(snippet)")
        }
    }
}
