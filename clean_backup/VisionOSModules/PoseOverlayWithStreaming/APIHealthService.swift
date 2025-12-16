import Foundation

struct APIHealthService {
    struct Response: Decodable {
        let status: String
        let rs_connected: Bool?
        let calibrated: Bool?
    }

    static func check(baseURL: URL) async throws -> Response {
        var req = URLRequest(url: baseURL.appendingPathComponent("health"))
        req.httpMethod = "GET"
        req.timeoutInterval = 3
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
            throw URLError(.init(rawValue: URLError.badServerResponse.rawValue),
                           userInfo: [NSLocalizedDescriptionKey: "Health check failed (\(code))"])
        }
        return try JSONDecoder().decode(Response.self, from: data)
    }
}
