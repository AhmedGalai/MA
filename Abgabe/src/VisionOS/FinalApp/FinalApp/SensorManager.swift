import SwiftUI
import ARKit
import AVFoundation
import Combine
import CoreImage
import CoreVideo
import QuartzCore

@MainActor
class SensorManager: ObservableObject {
    // Head tracking data
    @Published var headPosition: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    @Published var headRotation: SIMD3<Float> = SIMD3<Float>(0, 0, 0)
    @Published var headTrackingState: String = "Not Started"

    // Camera data
    @Published var cameraFrameCount: Int = 0
    @Published var cameraResolution: String = "N/A"
    @Published var cameraFPS: Float = 0.0
    @Published var cameraState: String = "Not Started"
    @Published var latestCameraImage: CGImage?

    // Device data
    @Published var deviceOrientation: String = "Unknown"
    // API / Pose overlay data
    @Published var apiHost: String = "127.0.0.1"
    @Published var apiPort: String = "8000"
    @Published var apiBaseURL: URL?
    @Published var healthStatus: String = "Unknown"
    @Published var lastHealthCheck: Date?
    @Published var headPoseUploadCount: Int = 0
    @Published var headPoseUploadError: String?

    // ARKit session
    private let arSession = ARKitSession()
    private var worldTracking: WorldTrackingProvider?
    private var cameraProvider: CameraFrameProvider?
    private var updateTimer: Timer?
    private var cameraTask: Task<Void, Never>?

    // Camera frame tracking
    private var frameCount: Int = 0
    private var lastFrameTime: Date = Date()

    private var lastHeadPoseUploadAttempt: Date = .distantPast
    private var headPoseUploadsEnabled = true

    init() {}

    // MARK: - API helpers
    func setAPI(host: String, port: String) {
        apiHost = SensorManager.sanitize(host: host)
        apiPort = SensorManager.sanitize(port: port)
        apiBaseURL = URL(string: "http://\(apiHost):\(apiPort)")
    }

    func setHeadPoseUploadsEnabled(_ enabled: Bool) {
        headPoseUploadsEnabled = enabled
    }

    func performHealthCheck() async {
        guard let base = apiBaseURL else {
            healthStatus = "Invalid URL"
            return
        }
        do {
            let res = try await APIHealthService.check(baseURL: base)
            healthStatus = res.status
            lastHealthCheck = Date()
        } catch {
            healthStatus = "Error: \(error.localizedDescription)"
        }
    }

    // MARK: - Restart ARKit (called from ImmersiveSpace)
    func restartARKitTracking() async {
        await startHeadTrackingAndCamera()
    }

    // MARK: - Head Tracking and Camera (ARKit)
    private func startHeadTrackingAndCamera() async {
        do {
            headTrackingState = "Requesting authorization..."
            cameraState = "Requesting authorization..."

            // Create NEW instances (can't re-run stopped providers)
            worldTracking = WorldTrackingProvider()
            cameraProvider = CameraFrameProvider()

            guard let worldTracking = worldTracking else {
                headTrackingState = "Failed to create provider"
                return
            }

            // Request authorization for both
            let authorizationResult = await arSession.requestAuthorization(for: [.worldSensing, .cameraAccess])

            guard authorizationResult[.worldSensing] == .allowed else {
                headTrackingState = "Not Authorized"
                return
            }

            var providers: [any DataProvider] = [worldTracking]

            // Add camera provider if authorized
            if authorizationResult[.cameraAccess] == .allowed, let cameraProvider = cameraProvider {
                providers.append(cameraProvider)
                cameraState = "Starting..."
            } else {
                cameraState = "Not Authorized"
            }

            // Run the session with both providers
            try await arSession.run(providers)
            headTrackingState = "Active"

            // Wait for tracking to stabilize
            try? await Task.sleep(for: .milliseconds(500))

            // Start timer to poll device anchor at 60Hz
            updateTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    await self?.updateHeadPose()
                }
            }

            // Start camera frame processing if authorized
            if authorizationResult[.cameraAccess] == .allowed, let cameraProvider = cameraProvider {
                cameraTask = Task { @MainActor in
                    await self.processCameraFrames(from: cameraProvider)
                }
            }

        } catch {
            headTrackingState = "Error: \(error.localizedDescription)"
            cameraState = "Error: \(error.localizedDescription)"
        }
    }

    private func updateHeadPose() async {
        guard let worldTracking = worldTracking,
              let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: CACurrentMediaTime()) else {
            return
        }

        let transform = deviceAnchor.originFromAnchorTransform

        // Extract position
        headPosition = SIMD3<Float>(
            transform.columns.3.x,
            transform.columns.3.y,
            transform.columns.3.z
        )

        // Extract rotation (Euler angles from rotation matrix)
        let rotationMatrix = simd_float3x3(
            SIMD3(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z),
            SIMD3(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z),
            SIMD3(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
        )
        let orientationQuat = simd_quatf(rotationMatrix)

        // Convert to Euler angles
        let pitch = asin(-rotationMatrix[2][0])
        let yaw = atan2(rotationMatrix[1][0], rotationMatrix[0][0])
        let roll = atan2(rotationMatrix[2][1], rotationMatrix[2][2])

        headRotation = SIMD3<Float>(
            pitch * 180.0 / .pi,
            yaw * 180.0 / .pi,
            roll * 180.0 / .pi
        )
        queueHeadPoseUpload(position: headPosition, rotation: headRotation, quaternion: orientationQuat)
    }


    private func processCameraFrames(from provider: CameraFrameProvider) async {
        frameCount = 0
        lastFrameTime = Date()

        // Get available camera formats
        let availableFormats = CameraVideoFormat.supportedVideoFormats(for: .main, cameraPositions: [.left])

        guard let format = availableFormats.first else {
            cameraState = "No camera formats available"
            return
        }

        guard let cameraFrameUpdates = cameraProvider?.cameraFrameUpdates(for: format) else {
            cameraState = "Failed to get camera frame updates"
            return
        }

        for await cameraFrame in cameraFrameUpdates {
            frameCount += 1

            // Update FPS every second
            let now = Date()
            let timeDiff = now.timeIntervalSince(lastFrameTime)
            if timeDiff >= 1.0 {
                cameraFPS = Float(frameCount) / Float(timeDiff)
                frameCount = 0
                lastFrameTime = now
            }

            // Get camera sample for left camera
            if let sample = cameraFrame.sample(for: .left) {
                let pixelBuffer = sample.pixelBuffer
                let width = CVPixelBufferGetWidth(pixelBuffer)
                let height = CVPixelBufferGetHeight(pixelBuffer)
                cameraResolution = "\(width)x\(height)"

                // Convert to CGImage for display
                latestCameraImage = pixelBufferToCGImage(pixelBuffer)
            }
        }
    }

    private func queueHeadPoseUpload(position: SIMD3<Float>,
                                     rotation: SIMD3<Float>,
                                     quaternion: simd_quatf) {
        guard headPoseUploadsEnabled, let base = apiBaseURL else { return }
        let now = Date()
        guard now.timeIntervalSince(lastHeadPoseUploadAttempt) >= 0.2 else { return }
        lastHeadPoseUploadAttempt = now

        let payload = HeadPosePayload(
            position: [Double(position.x), Double(position.y), Double(position.z)],
            rotation: [Double(rotation.x), Double(rotation.y), Double(rotation.z)],
            quaternion: [Double(quaternion.imag.x), Double(quaternion.imag.y), Double(quaternion.imag.z), Double(quaternion.real)],
            timestamp: now.timeIntervalSince1970,
            confidence: 1.0,
            metadata: [
                "source": "visionOS",
                "module": "FinalApp"
            ]
        )

        Task.detached(priority: .background) { [weak self] in
            do {
                try await HeadPoseService.send(baseURL: base, payload: payload)
                await MainActor.run {
                    self?.headPoseUploadCount += 1
                    self?.headPoseUploadError = nil
                }
            } catch {
                await MainActor.run {
                    self?.headPoseUploadError = error.localizedDescription
                }
            }
        }
    }

    private func pixelBufferToCGImage(_ pixelBuffer: CVPixelBuffer) -> CGImage? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()

        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            return nil
        }

        return cgImage
    }


    // MARK: - Cleanup
    func stopMonitoring() {
        updateTimer?.invalidate()
        updateTimer = nil
        cameraTask?.cancel()
        cameraTask = nil
        arSession.stop()
    }

    // MARK: - Helpers
    private static func sanitize(host: String) -> String {
        var raw = host.trimmingCharacters(in: .whitespacesAndNewlines)
        raw = raw.replacingOccurrences(of: "http://", with: "")
        raw = raw.replacingOccurrences(of: "https://", with: "")
        raw = raw.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        return raw.isEmpty ? "127.0.0.1" : raw
    }

    private static func sanitize(port: String) -> String {
        let trimmed = port.trimmingCharacters(in: .whitespacesAndNewlines)
        if let value = Int(trimmed), value > 0 && value < 65536 { return "\(value)" }
        return "8000"
    }
}
