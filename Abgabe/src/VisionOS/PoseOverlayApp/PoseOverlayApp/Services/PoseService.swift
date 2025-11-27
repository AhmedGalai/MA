import Foundation
import simd

enum PoseService {
    enum PoseError: Error, LocalizedError {
        case missingRGBFrame
        case missingIntrinsics
        case invalidResponse(String)
        case noPoseAvailable(String)

        var errorDescription: String? {
            switch self {
            case .missingRGBFrame:
                return "No RGB frame available from pipeline"
            case .missingIntrinsics:
                return "Camera intrinsics are unavailable"
            case .invalidResponse(let detail):
                return detail
            case .noPoseAvailable(let status):
                return status.isEmpty ? "No pose available yet" : status
            }
        }
    }

    private struct IntrinsicsResponse: Decodable {
        let K: [[Double]]
    }

    private struct FrameResponse: Decodable {
        let frame: String
    }

    private struct MaskResponse: Decodable {
        let mask: String
    }

    private struct DisparityResponse: Decodable {
        let disparity: String
    }

    private struct PoseRequestPayload: Encodable {
        let rgbFrame: String
        let cameraMatrix: [[Double]]
        let depthMap: String?
        let mask: String?
        let modelName: String?

        enum CodingKeys: String, CodingKey {
            case rgbFrame = "rgb_frame"
            case cameraMatrix = "camera_matrix"
            case depthMap = "depth_map"
            case mask
            case modelName = "model_name"
        }
    }

    private struct AVPPoseEnvelope: Decodable {
        let success: Bool?
        let pose: PoseResponse?
        let error: String?
    }

    static func fetchTransforms(baseURL: URL, modelName: String?) async throws -> [simd_float4x4] {
        let snapshot = try await fetchSnapshot(baseURL: baseURL)
        let matrices = try await requestPose(baseURL: baseURL,
                                             snapshot: snapshot,
                                             modelName: modelName)
        return matrices
            .map(MatrixUtils.simdMatrix(from:))
            .map(MatrixUtils.convertOpenCVToRealityKit)
    }

    // MARK: - Snapshot
    private struct PipelineSnapshot {
        let rgbFrame: String
        let cameraMatrix: [[Double]]
        let mask: String?
        let depthMap: String?
    }

    private static func fetchSnapshot(baseURL: URL) async throws -> PipelineSnapshot {
        async let rgbTask = fetchRGBFrame(baseURL: baseURL)
        async let intrinsicsTask = fetchIntrinsics(baseURL: baseURL)
        async let maskTask = fetchOptionalMask(baseURL: baseURL)
        async let depthTask = fetchOptionalDisparity(baseURL: baseURL)

        guard let rgb = try await rgbTask else {
            throw PoseError.missingRGBFrame
        }
        guard let intrinsics = try await intrinsicsTask else {
            throw PoseError.missingIntrinsics
        }
        let snapshot = PipelineSnapshot(
            rgbFrame: rgb,
            cameraMatrix: intrinsics,
            mask: try await maskTask,
            depthMap: try await depthTask
        )
        return snapshot
    }

    private static func fetchRGBFrame(baseURL: URL) async throws -> String? {
        let url = baseURL.appendingPathComponent("rgb_frame")
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        req.timeoutInterval = 2
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
            return nil
        }
        return try JSONDecoder().decode(FrameResponse.self, from: data).frame
    }

    private static func fetchIntrinsics(baseURL: URL) async throws -> [[Double]]? {
        let url = baseURL.appendingPathComponent("intrinsics")
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        req.timeoutInterval = 2
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
            return nil
        }
        return try JSONDecoder().decode(IntrinsicsResponse.self, from: data).K
    }

    private static func fetchOptionalMask(baseURL: URL) async throws -> String? {
        let url = baseURL.appendingPathComponent("mask")
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        req.timeoutInterval = 2
        do {
            let (data, resp) = try await URLSession.shared.data(for: req)
            guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
                return nil
            }
            return try JSONDecoder().decode(MaskResponse.self, from: data).mask
        } catch {
            return nil
        }
    }

    private static func fetchOptionalDisparity(baseURL: URL) async throws -> String? {
        let url = baseURL.appendingPathComponent("disparity")
        var req = URLRequest(url: url)
        req.httpMethod = "GET"
        req.timeoutInterval = 2
        do {
            let (data, resp) = try await URLSession.shared.data(for: req)
            guard let http = resp as? HTTPURLResponse, http.statusCode == 200 else {
                return nil
            }
            return try JSONDecoder().decode(DisparityResponse.self, from: data).disparity
        } catch {
            return nil
        }
    }

    // MARK: - Pose Request
    private static func requestPose(baseURL: URL,
                                    snapshot: PipelineSnapshot,
                                    modelName: String?) async throws -> [Matrix4x4DTO] {
        let payload = PoseRequestPayload(
            rgbFrame: snapshot.rgbFrame,
            cameraMatrix: snapshot.cameraMatrix,
            depthMap: snapshot.depthMap,
            mask: snapshot.mask,
            modelName: modelName
        )

        let url = baseURL.appendingPathComponent("avp_pose")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(payload)

        let (data, resp) = try await URLSession.shared.data(for: req)
        guard let http = resp as? HTTPURLResponse else {
            throw PoseError.invalidResponse("Invalid response from /avp_pose")
        }
        guard (200..<300).contains(http.statusCode) else {
            let snippet = String(data: data, encoding: .utf8) ?? "<non-utf8>"
            throw PoseError.invalidResponse("Pose request failed (\(http.statusCode)): \(snippet)")
        }

        let envelope = try JSONDecoder().decode(AVPPoseEnvelope.self, from: data)
        if let err = envelope.error {
            throw PoseError.invalidResponse(err)
        }
        guard let pose = envelope.pose else {
            throw PoseError.noPoseAvailable("Pose unavailable")
        }
        let matrices = pose.transformation_matrix
        guard !matrices.isEmpty else {
            throw PoseError.noPoseAvailable(pose.status ?? "Pose unavailable")
        }
        return matrices
    }
}
