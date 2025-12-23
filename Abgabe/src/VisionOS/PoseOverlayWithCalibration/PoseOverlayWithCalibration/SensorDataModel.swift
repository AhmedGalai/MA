import Foundation
import CoreMotion
import simd
import ARKit
import Combine
import QuartzCore

@MainActor
final class SensorDataModel: ObservableObject {
    @Published var headPosition: SIMD3<Double> = .zero
    @Published var headOrientation: simd_quatd = simd_quatd(ix: 0, iy: 0, iz: 0, r: 1)
    @Published var headEulerDegrees: SIMD3<Double> = .zero
    @Published var lastPoseUpdate: Date = .distantPast
    @Published private(set) var latestDeviceTransform: simd_float4x4?

    @Published var userAcceleration: CMAcceleration = .init()
    @Published var rotationRate: CMRotationRate = .init()
    @Published var gravity: CMAcceleration = .init()
    @Published var attitude: CMAttitude?
    @Published var lastMotionUpdate: Date = .distantPast

    @Published var statusMessage: String = "Starting…"
    @Published var frameCount: Int = 0  // Visual indicator of updates
    @Published var lastHeadPoseUpload: Date = .distantPast
    @Published var headPoseUploadError: String?
    @Published var headPoseUploadCount: Int = 0

    // CoreMotion (fallback for iOS)
    private let motionManager = CMMotionManager()
    private let motionQueue = OperationQueue.main

    // ARKit for visionOS
    private let arkitSession = ARKitSession()
    let worldTracking = WorldTrackingProvider()  // Exposed for camera transform queries
    private var updateTimer: Timer?
    private var updateTask: Task<Void, Never>?

    private var started = false
    private var apiBaseURL: URL?
    private var lastHeadPoseUploadAttempt: Date = .distantPast

    func setAPIBaseURL(_ url: URL?) {
        apiBaseURL = url
    }

    func start() {
        NSLog("📱 [SensorDataModel] START CALLED")
        guard !started else {
            NSLog("📱 [SensorDataModel] Already started, ignoring")
            statusMessage = "Already started"
            return
        }
        started = true
        statusMessage = "Initializing sensors"

        #if os(visionOS)
        NSLog("📱 [SensorDataModel] Using ARKit for visionOS")
        NSLog("📱 [SensorDataModel] Waiting for ImmersiveSpace to open...")
        statusMessage = "Tap 'Show 3D View' to enable ARKit"
        #else
        NSLog("📱 [SensorDataModel] Using CoreMotion for iOS")
        startMotionUpdates()
        #endif
    }

    func restartARKitTracking() async {
        NSLog("📱 [SensorDataModel] restartARKitTracking() called from ImmersiveSpace")
        Task.detached(priority: .userInitiated) { [weak self] in
            await self?.startARKitTrackingInBackground()
        }
    }

    func stop() {
        started = false
        updateTimer?.invalidate()
        updateTimer = nil
        updateTask?.cancel()
        updateTask = nil
        motionManager.stopDeviceMotionUpdates()
        statusMessage = "Stopped"
        latestDeviceTransform = nil
    }

    // MARK: - ARKit Tracking (visionOS)

    private func startARKitTrackingInBackground() async {
        NSLog("📱 [SensorDataModel] Starting ARKitSession...")

        do {
            let (session, tracking) = await MainActor.run { (arkitSession, worldTracking) }
            await MainActor.run {
                updateTimer?.invalidate()
                updateTimer = nil
                updateTask?.cancel()
                updateTask = nil
            }
            // Request authorization first (this shows the permission prompt)
            NSLog("📱 [SensorDataModel] Requesting ARKit authorization...")
            let auth = await session.requestAuthorization(for: [.worldSensing])

            guard auth[.worldSensing] == .allowed else {
                NSLog("📱 [SensorDataModel] ❌ WorldSensing not authorized: \(auth[.worldSensing])")
                NSLog("📱 [SensorDataModel] Falling back to CoreMotion...")
                await MainActor.run {
                    statusMessage = "ARKit denied - trying CoreMotion"
                    self.startMotionUpdates()
                }
                return
            }

            NSLog("📱 [SensorDataModel] ✓ ARKit authorized")

            // Run ARKit session with world tracking.
            try await session.run([tracking])
            NSLog("📱 [SensorDataModel] ✓ ARKitSession running")

            // Wait a moment for tracking to stabilize
            try? await Task.sleep(for: .milliseconds(500))

            // Check if world tracking is running
            NSLog("📱 [SensorDataModel] WorldTracking state: \(tracking.state)")

            if tracking.state != .running {
                NSLog("📱 [SensorDataModel] ⚠️ WorldTracking not running yet: \(tracking.state), continuing anyway...")
            }

            // Poll device anchor off the main thread to avoid UI stalls.
            await MainActor.run {
                self.updateTask = Task.detached(priority: .userInitiated) { [weak self] in
                    while !Task.isCancelled {
                        if let deviceAnchor = tracking.queryDeviceAnchor(atTimestamp: CACurrentMediaTime()) {
                            let sample = Self.makeARKitSample(from: deviceAnchor.originFromAnchorTransform)
                            await self?.applyARKitSample(sample)
                        } else if let lastPoseUpdate = await self?.lastPoseUpdate,
                                  lastPoseUpdate == .distantPast {
                            NSLog("📱 [SensorDataModel] ❌ queryDeviceAnchor returned nil (tracking not ready?)")
                        }
                        try? await Task.sleep(for: .milliseconds(33))
                    }
                }
                self.statusMessage = "Tracking head pose"
            }
            NSLog("📱 [SensorDataModel] ✓ Started 60Hz polling task")

        } catch {
            NSLog("📱 [SensorDataModel] ❌ ARKit error: \(error.localizedDescription)")
            await MainActor.run {
                statusMessage = "ARKit error: \(error.localizedDescription)"
            }
        }
    }

    private struct ARKitSample {
        let position: SIMD3<Double>
        let orientation: simd_quatd
        let eulerDegrees: SIMD3<Double>
        let transform: simd_float4x4
    }

    nonisolated private static func makeARKitSample(from transform: simd_float4x4) -> ARKitSample {
        let position = SIMD3<Double>(
            Double(transform.columns.3.x),
            Double(transform.columns.3.y),
            Double(transform.columns.3.z)
        )

        let rotationMatrix = simd_float3x3(
            SIMD3<Float>(transform.columns.0.x, transform.columns.0.y, transform.columns.0.z),
            SIMD3<Float>(transform.columns.1.x, transform.columns.1.y, transform.columns.1.z),
            SIMD3<Float>(transform.columns.2.x, transform.columns.2.y, transform.columns.2.z)
        )

        let orientation = simd_quatf(rotationMatrix)
        let orientationD = simd_quatd(
            ix: Double(orientation.imag.x),
            iy: Double(orientation.imag.y),
            iz: Double(orientation.imag.z),
            r: Double(orientation.real)
        )
        let euler = eulerDegrees(from: orientationD)

        return ARKitSample(
            position: position,
            orientation: orientationD,
            eulerDegrees: euler,
            transform: transform
        )
    }

    private func applyARKitSample(_ sample: ARKitSample) {
        // First update - log it
        if self.lastPoseUpdate == .distantPast {
            NSLog("📱 [SensorDataModel] ✓✓✓ RECEIVING ARKit UPDATES! ✓✓✓")
            NSLog("📱 [SensorDataModel] Position: [\(sample.position.x), \(sample.position.y), \(sample.position.z)]")
        }

        self.headPosition = sample.position
        self.headOrientation = sample.orientation
        self.headEulerDegrees = sample.eulerDegrees
        self.latestDeviceTransform = sample.transform
        self.lastPoseUpdate = Date()
        self.lastMotionUpdate = Date()
        self.frameCount += 1  // Increment frame counter for visual feedback

        // Log every 60 frames (1 second at 60Hz)
        if self.frameCount % 60 == 0 {
            NSLog("📱 [SensorDataModel] Frame \(self.frameCount) - Pos: [\(String(format: "%.3f", sample.position.x)), \(String(format: "%.3f", sample.position.y)), \(String(format: "%.3f", sample.position.z))]")
        }

        // Simulate device motion data from head movement (for compatibility)
        // Note: These are approximations since ARKit doesn't provide raw IMU data
        self.userAcceleration = CMAcceleration(x: 0, y: 0, z: 0)
        self.rotationRate = CMRotationRate(x: 0, y: 0, z: 0)
        self.gravity = CMAcceleration(x: 0, y: -1, z: 0) // Simulated gravity

        self.queueHeadPoseUpload(position: sample.position,
                                 eulerDegrees: sample.eulerDegrees,
                                 orientation: sample.orientation)
    }

    // MARK: - CoreMotion (iOS fallback)

    private func startMotionUpdates() {
        NSLog("📱 [SensorDataModel] Checking device motion availability...")

        guard motionManager.isDeviceMotionAvailable else {
            NSLog("📱 [SensorDataModel] ❌ Device motion NOT available")
            statusMessage = "Device motion unavailable (simulator?)"
            return
        }

        NSLog("📱 [SensorDataModel] ✓ Device motion available")
        #if os(iOS)
        NSLog("📱 [SensorDataModel] Accelerometer: %d, Gyro: %d, Magnetometer: %d",
              motionManager.isAccelerometerAvailable,
              motionManager.isGyroAvailable,
              motionManager.isMagnetometerAvailable)
        #else
        NSLog("📱 [SensorDataModel] Accelerometer: %d, Gyro: %d (visionOS - no magnetometer API)",
              motionManager.isAccelerometerAvailable,
              motionManager.isGyroAvailable)
        #endif

        motionManager.deviceMotionUpdateInterval = 1.0 / 60.0

        // Try reference frames in order of preference
        // Note: visionOS may have limited reference frame support
        var referenceFrames: [(CMAttitudeReferenceFrame, String)] = [
            (.xArbitraryZVertical, "xArbitraryZVertical"),
            (.xArbitraryCorrectedZVertical, "xArbitraryCorrectedZVertical")
        ]

        #if os(iOS)
        referenceFrames.append((.xMagneticNorthZVertical, "xMagneticNorthZVertical"))
        referenceFrames.append((.xTrueNorthZVertical, "xTrueNorthZVertical"))
        #endif

        let availableFrames = CMMotionManager.availableAttitudeReferenceFrames()
        NSLog("📱 [SensorDataModel] Available reference frames: %lu", availableFrames.rawValue)

        var selectedFrame: CMAttitudeReferenceFrame = .xArbitraryZVertical
        for (frame, name) in referenceFrames {
            if availableFrames.contains(frame) {
                NSLog("📱 [SensorDataModel] ✓ Using reference frame: %@", name)
                selectedFrame = frame
                break
            } else {
                NSLog("📱 [SensorDataModel] ✗ Reference frame unavailable: %@", name)
            }
        }

        NSLog("📱 [SensorDataModel] Starting motion updates at 60Hz...")
        motionManager.startDeviceMotionUpdates(using: selectedFrame, to: motionQueue) { [weak self] motion, error in
            guard let self else { return }

            if let error = error {
                NSLog("📱 [SensorDataModel] ❌ Motion update error: %@", error.localizedDescription)
                self.statusMessage = "Error: \(error.localizedDescription)"
                return
            }

            guard let motion = motion else {
                NSLog("📱 [SensorDataModel] ⚠️ Received nil motion data")
                return
            }

            // First update - log it
            if self.lastMotionUpdate == .distantPast {
                NSLog("📱 [SensorDataModel] ✓✓✓ RECEIVING CoreMotion UPDATES! ✓✓✓")
            }

            self.userAcceleration = motion.userAcceleration
            self.rotationRate = motion.rotationRate
            self.gravity = motion.gravity
            self.attitude = motion.attitude
            self.lastMotionUpdate = Date()
            self.statusMessage = "Tracking head pose"
            self.frameCount += 1  // Increment frame counter

            // Log every 60 frames
            if self.frameCount % 60 == 0 {
                NSLog("📱 [SensorDataModel] CoreMotion Frame \(self.frameCount)")
            }

            let q = motion.attitude.quaternion
            let orientation = simd_quatd(ix: q.x, iy: q.y, iz: q.z, r: q.w)
            self.headOrientation = orientation
            let euler = Self.eulerDegrees(from: orientation)
            self.headEulerDegrees = euler
            self.lastPoseUpdate = Date()
            self.headPosition = SIMD3<Double>(
                motion.gravity.x,
                motion.gravity.y,
                motion.gravity.z
            )
            self.queueHeadPoseUpload(position: self.headPosition,
                                     eulerDegrees: euler,
                                     orientation: orientation)
        }

        NSLog("📱 [SensorDataModel] Motion updates started (waiting for first callback...)")

        // Fallback: If no updates after 2 seconds, start test mode with fake data
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) { [weak self] in
            guard let self = self else { return }
            if self.frameCount == 0 {
                NSLog("📱 [SensorDataModel] ⚠️ No CoreMotion updates after 2s, starting TEST MODE with fake data")
                self.startTestMode()
            }
        }
    }

    // MARK: - Test Mode (Fallback)

    private func startTestMode() {
        NSLog("📱 [SensorDataModel] Starting test mode - generating fake sensor data")
        statusMessage = "TEST MODE - Fake Data"

        // Generate fake data at 60Hz
        updateTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            guard let self = self else { return }

            let time = Date().timeIntervalSince1970
            let slowWave = sin(time * 0.5)  // Slow oscillation
            let fastWave = sin(time * 2.0)  // Fast oscillation

            // Fake position that changes over time
            self.headPosition = SIMD3<Double>(
                slowWave * 0.1,     // X: -0.1 to 0.1
                1.6,                 // Y: Fixed at head height
                fastWave * 0.05     // Z: Small movement
            )

            // Fake orientation (slowly rotating)
            let angle = time * 0.3
            let orientation = simd_quatd(angle: angle, axis: SIMD3<Double>(0, 1, 0))
            self.headOrientation = orientation
            let euler = Self.eulerDegrees(from: orientation)
            self.headEulerDegrees = euler

            // Fake motion data
            self.userAcceleration = CMAcceleration(x: slowWave * 0.1, y: 0, z: fastWave * 0.05)
            self.rotationRate = CMRotationRate(x: 0, y: fastWave * 0.2, z: 0)
            self.gravity = CMAcceleration(x: 0, y: -1, z: 0)

            self.lastPoseUpdate = Date()
            self.lastMotionUpdate = Date()
            self.frameCount += 1
            self.queueHeadPoseUpload(position: self.headPosition,
                                     eulerDegrees: euler,
                                     orientation: orientation)

            // Log first update and every 60 frames
            if self.frameCount == 1 {
                NSLog("📱 [SensorDataModel] ✓✓✓ TEST MODE ACTIVE - FAKE DATA UPDATING! ✓✓✓")
            }
            if self.frameCount % 60 == 0 {
                NSLog("📱 [SensorDataModel] TEST Frame \(self.frameCount) - Pos: [\(String(format: "%.3f", self.headPosition.x)), \(String(format: "%.3f", self.headPosition.y)), \(String(format: "%.3f", self.headPosition.z))]")
            }
        }
    }

    private func queueHeadPoseUpload(position: SIMD3<Double>,
                                     eulerDegrees: SIMD3<Double>,
                                     orientation: simd_quatd) {
        guard let baseURL = apiBaseURL else { return }
        let now = Date()
        guard now.timeIntervalSince(lastHeadPoseUploadAttempt) >= 0.15 else { return }
        lastHeadPoseUploadAttempt = now

        let payload = HeadPosePayload(
            position: [position.x, position.y, position.z],
            rotation: [eulerDegrees.x, eulerDegrees.y, eulerDegrees.z],
            quaternion: [orientation.imag.x, orientation.imag.y, orientation.imag.z, orientation.real],
            timestamp: now.timeIntervalSince1970,
            confidence: 1.0,
            metadata: [
                "source": "visionOS",
                "notes": "PoseOverlayWithCalibration"
            ]
        )

        Task.detached(priority: .background) { [weak self] in
            do {
                try await HeadPoseService.send(baseURL: baseURL, payload: payload)
                await MainActor.run {
                    self?.lastHeadPoseUpload = Date()
                    self?.headPoseUploadError = nil
                    self?.headPoseUploadCount += 1
                }
            } catch {
                await MainActor.run {
                    self?.headPoseUploadError = error.localizedDescription
                }
            }
        }
    }

    nonisolated static func eulerDegrees(from q: simd_quatd) -> SIMD3<Double> {
        let ysqr = q.imag.y * q.imag.y

        let t0 = 2.0 * (q.real * q.imag.x + q.imag.y * q.imag.z)
        let t1 = 1.0 - 2.0 * (q.imag.x * q.imag.x + ysqr)
        let roll = atan2(t0, t1)

        let t2 = max(-1.0, min(1.0, 2.0 * (q.real * q.imag.y - q.imag.z * q.imag.x)))
        let pitch = asin(t2)

        let t3 = 2.0 * (q.real * q.imag.z + q.imag.x * q.imag.y)
        let t4 = 1.0 - 2.0 * (ysqr + q.imag.z * q.imag.z)
        let yaw = atan2(t3, t4)

        return SIMD3<Double>(roll, pitch, yaw) * (180.0 / .pi)
    }
}
