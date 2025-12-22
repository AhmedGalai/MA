import Foundation

enum ModelService {
    struct APIError: Error, LocalizedError {
        let message: String
        var errorDescription: String? { message }
    }

    private struct ModelListResponse: Decodable {
        struct Item: Decodable {
            let name: String?
        }
        let models: [Item]
    }

    private struct LegacyModelListResponse: Decodable {
        let models: [String]
    }

    static func fetchModelList(baseURL: URL) async throws -> [String] {
        let url = baseURL.appendingPathComponent("models")
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        req.setValue("application/json", forHTTPHeaderField: "Accept")
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse else {
            throw APIError(message: "Failed to fetch models (invalid response)")
        }
        guard (200..<300).contains(http.statusCode) else {
            let snippet = String(data: data, encoding: .utf8) ?? "<non-utf8>"
            throw APIError(message: "Failed to fetch models (status \(http.statusCode)): \(snippet)")
        }

        if let response = try? JSONDecoder().decode(ModelListResponse.self, from: data) {
            let names = response.models.compactMap { $0.name?.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
            if !names.isEmpty { return names }
        }
        if let legacy = try? JSONDecoder().decode(LegacyModelListResponse.self, from: data) {
            return legacy.models
        }
        throw APIError(message: "Malformed /models payload")
    }

    static func selectModel(baseURL: URL, name: String) async throws {
        let url = baseURL.appendingPathComponent("select_model")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let payload = ["model_name": name]
        req.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse else {
            throw APIError(message: "Failed to post selected model (invalid response)")
        }
        guard (200..<300).contains(http.statusCode) else {
            let snippet = String(data: data, encoding: .utf8) ?? "<non-utf8>"
            throw APIError(message: "Failed to post selected model (status \(http.statusCode)): \(snippet)")
        }
    }
}
