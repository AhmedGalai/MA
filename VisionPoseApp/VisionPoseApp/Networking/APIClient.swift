import Foundation
import UIKit

class APIClient {
    static let baseURL = "http://localhost:8000"

    static func fetchModels(completion: @escaping ([String]) -> Void) {
        guard let url = URL(string: "\(baseURL)/models") else { return }
        URLSession.shared.dataTask(with: url) { data, _, _ in
            if let data = data, let list = try? JSONDecoder().decode([String].self, from: data) {
                completion(list)
            }
        }.resume()
    }

    static func sendIntrinsics(left: UIImage, right: UIImage, completion: @escaping ([[Double]]) -> Void) {
        // TODO: encode as base64 + send
        completion([[700,0,320],[0,700,240],[0,0,1]]) // stub
    }

    static func requestPose(modelFile: String, completion: @escaping (String) -> Void) {
        guard let url = URL(string: "\(baseURL)/pose") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        let body = ["mesh": modelFile]
        req.httpBody = try? JSONSerialization.data(withJSONObject: body)
        req.addValue("application/json", forHTTPHeaderField: "Content-Type")

        URLSession.shared.dataTask(with: req) { data, _, _ in
            if let data = data, let resp = String(data: data, encoding: .utf8) {
                completion(resp)
            }
        }.resume()
    }
}

