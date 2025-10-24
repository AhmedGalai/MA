import Foundation

fileprivate let API_BASE = URL(string: "http://localhost:8000")!

struct ModelName: Decodable { let name: String }
struct ModelsResponse: Decodable { let models: [ModelName] }
struct ModelMeshResponse: Decodable { let name: String; let mesh: String }
struct IntrinsicsResponse: Decodable { let camera_matrix: [[Double]]; let debug_image: String? }
struct DepthResponse: Decodable { let depth: String }
struct PoseResponse: Decodable {
    let status: String?
    let transformation_matrix: [[[Double]]]
    let debug: [String: Double]?
}

actor APIClient {
    static let shared = APIClient()
    private let session = URLSession(configuration: .default)

    func fetchModels() async throws -> [String] {
        let url = API_BASE.appendingPathComponent("models")
        let (data, _) = try await session.data(from: url)
        let decoded = try JSONDecoder().decode(ModelsResponse.self, from: data)
        return decoded.models.map { $0.name }
    }

    func fetchModelMesh(name: String) async throws -> Data {
        var comps = URLComponents(url: API_BASE.appendingPathComponent("model"), resolvingAgainstBaseURL: false)!
        comps.queryItems = [URLQueryItem(name: "name", value: name)]
        let (data, _) = try await session.data(from: comps.url!)
        let decoded = try JSONDecoder().decode(ModelMeshResponse.self, from: data)
        guard let raw = Data(base64Encoded: decoded.mesh) else { throw NSError(domain: "b64", code: -1) }
        return raw
    }

    func postIntrinsics(leftJPEG: Data, rightJPEG: Data) async throws -> [[Double]] {
        struct Req: Encodable { let left: String; let right: String }
        let req = Req(left: leftJPEG.base64EncodedString(), right: rightJPEG.base64EncodedString())
        let url = API_BASE.appendingPathComponent("intrinsics")
        var r = URLRequest(url: url); r.httpMethod = "POST"
        r.addValue("application/json", forHTTPHeaderField: "Content-Type")
        r.httpBody = try JSONEncoder().encode(req)
        let (data, _) = try await session.data(for: r)
        let decoded = try JSONDecoder().decode(IntrinsicsResponse.self, from: data)
        return decoded.camera_matrix
    }

    func postDepth(rgbJPEG: Data) async throws -> Data {
        struct Req: Encodable { let rgb: String }
        let req = Req(rgb: rgbJPEG.base64EncodedString())
        let url = API_BASE.appendingPathComponent("depth")
        var r = URLRequest(url: url); r.httpMethod = "POST"
        r.addValue("application/json", forHTTPHeaderField: "Content-Type")
        r.httpBody = try JSONEncoder().encode(req)
        let (data, _) = try await session.data(for: r)
        let decoded = try JSONDecoder().decode(DepthResponse.self, from: data)
        guard let raw = Data(base64Encoded: decoded.depth) else { throw NSError(domain: "b64", code: -2) }
        return raw
    }

    func postPose(K: [[Double]], rgbJPEG: Data, depthPNG: Data, maskPNG: Data?, modelName: String, depthscale: Double) async throws -> PoseResponse {
        struct ImageItem: Encodable { let filename: String; let rgb: String; let depth: String }
        struct Req: Encodable {
            let camera_matrix: [[Double]]
            let images: [ImageItem]
            let mesh: String // empty; middle API fills from model
            let model: String
            let mask: String?
            let depthscale: Double
        }
        let img = ImageItem(filename: "snap", rgb: rgbJPEG.base64EncodedString(), depth: depthPNG.base64EncodedString())
        let req = Req(camera_matrix: K, images: [img], mesh: "", model: modelName, mask: maskPNG?.base64EncodedString(), depthscale: depthscale)
        let url = API_BASE.appendingPathComponent("pose")
        var r = URLRequest(url: url); r.httpMethod = "POST"
        r.addValue("application/json", forHTTPHeaderField: "Content-Type")
        r.httpBody = try JSONEncoder().encode(req)
        let (data, _) = try await session.data(for: r)
        return try JSONDecoder().decode(PoseResponse.self, from: data)
    }
}

