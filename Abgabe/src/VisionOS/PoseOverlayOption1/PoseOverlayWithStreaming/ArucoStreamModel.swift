import Foundation
import SwiftUI
import simd
import UIKit

@MainActor
final class ArucoStreamModel: ObservableObject {
    @Published var annotatedImage: UIImage?
    @Published var lastTimestamp: Date?
    @Published var status: String = "Idle"
    @Published var latestPose: ArucoBoardPose?
    @Published var calibratedBoardTransform: simd_float4x4?

    /// Stored device-to-ArUco transform for continuous tracking when marker is not visible
    @Published var deviceToArucoTransform: simd_float4x4?

    /// Whether we're currently tracking (true when ArUco detected, remains true for continuous tracking)
    @Published var isTracking: Bool = false

    private var latestBoardTransform: simd_float4x4? {
        didSet {
            applyCalibration()
        }
    }

    private let calibrationManager: CalibrationManager

    private var pollTask: Task<Void, Never>?
    
    init(calibrationManager: CalibrationManager) {
        self.calibrationManager = calibrationManager
    }

    func startStreaming(baseURL: URL) {
        stopStreaming()
        status = "Connecting…"
        pollTask = Task { [weak self] in
            guard let self else { return }
            await pollLoop(baseURL: baseURL)
        }
    }

    func stopStreaming() {
        pollTask?.cancel()
        pollTask = nil
        status = "Stopped"
    }

    private func pollLoop(baseURL: URL) async {
        let url = baseURL.appendingPathComponent("aruco")
        let decoder = JSONDecoder()

        while !Task.isCancelled {
            do {
                let (data, response) = try await URLSession.shared.data(from: url)
                guard let http = response as? HTTPURLResponse,
                      (200..<300).contains(http.statusCode) else {
                    status = "HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1)"
                    try? await Task.sleep(for: .milliseconds(800))
                    continue
                }

                let payload = try decoder.decode(ArucoAPIResponse.self, from: data)
                apply(payload)
                status = payload.pose?.detected == true ? "Live (detected)" : "Live (searching)"
            } catch {
                status = "Error: \(error.localizedDescription)"
            }

            try? await Task.sleep(for: .milliseconds(250))
        }
    }

    private func apply(_ payload: ArucoAPIResponse) {
        if let dataURL = payload.rgb,
           let image = Self.decodeImage(dataURL) {
            annotatedImage = image
            if let ts = payload.timestamp {
                lastTimestamp = Date(timeIntervalSince1970: ts)
            }
        }

        if let posePayload = payload.pose {
            let pose = ArucoBoardPose(from: posePayload)
            latestPose = pose
            latestBoardTransform = pose.realityTransform
        } else {
            latestPose = nil
            latestBoardTransform = nil
        }
    }
    
    private func applyCalibration() {
        guard let latest = latestBoardTransform, let calibration = calibrationManager.calibrationTransform else {
            calibratedBoardTransform = latestBoardTransform
            return
        }
        calibratedBoardTransform = latest * calibration
    }

    private static func decodeImage(_ dataURL: String) -> UIImage? {
        guard let commaIndex = dataURL.firstIndex(of: ",") else { return nil }
        let b64 = String(dataURL[dataURL.index(after: commaIndex)...])
        guard let data = Data(base64Encoded: b64) else { return nil }
        return UIImage(data: data)
    }
}

struct ArucoAPIResponse: Decodable {
    let rgb: String?
    let timestamp: Double?
    let pose: ArucoPoseResponse?
}

struct ArucoPoseResponse: Decodable {
    let detected: Bool
    let marker_ids: [Int]?
    let board_pose_camera_T_4x4: [[Double]]?
    let quaternion_xyzw: [Double]?
    let rvec: [Double]?
    let tvec: [Double]?
    let num_markers: Int?
    let K: [[Double]]?
}

struct ArucoBoardPose {
    let detected: Bool
    let markerIDs: [Int]
    let numMarkers: Int
    let boardPose: [[Double]]?
    let intrinsics: [[Double]]?

    init(from response: ArucoPoseResponse) {
        detected = response.detected
        markerIDs = response.marker_ids ?? []
        numMarkers = response.num_markers ?? markerIDs.count
        boardPose = response.board_pose_camera_T_4x4
        intrinsics = response.K
    }

    var isDetected: Bool {
        detected || boardPose != nil || numMarkers > 0
    }

    var realityTransform: simd_float4x4? {
        guard let boardPose,
              boardPose.count == 4,
              boardPose.allSatisfy({ $0.count == 4 }) else { return nil }
        let cvBoardToCam = MatrixUtils.simdMatrix(from: boardPose)
        // OpenCV pose is board->camera; RealityKit wants the entity pose in camera space.
        return MatrixUtils.convertOpenCVToRealityKit(cvBoardToCam)
    }
}
