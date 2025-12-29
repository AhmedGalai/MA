import SwiftUI
import UIKit
import simd

// Periodically polls the Python API and mirrors the debugging layout from debug_viewer.py.
@MainActor
final class DebugDashboardModel: ObservableObject {
    struct HealthState {
        var ok = false
        var rsConnected = false
        var calibrated = false
        var lastUpdated: Date?
    }

    enum DashboardError: LocalizedError {
        case badHTTPStatus(Int, String)

        var errorDescription: String? {
            switch self {
            case .badHTTPStatus(let code, let message):
                return "HTTP \(code): \(message)"
            }
        }
    }

    struct FrameState {
        var image: UIImage?
        var subtitle: String = ""
        var details: String = ""
        var timestamp: Date?
        var poseMatrix: [[Double]]?
    }

    struct IntrinsicsState {
        var rs: [[Double]]?
        var avp: [[Double]]?
    }

    struct TransformState {
        var avpRS: [[Double]]?
        var worldRS: [[Double]]?
        var worldAVP: [[Double]]?
        var avpBoard: [[Double]]?
    }

    struct PoseInAVPState {
        var matrix: [[Double]]?
        var headPoseAge: Double?
    }

    @Published var health = HealthState()
    @Published var rgbFrame = FrameState()
    @Published var depthFrame = FrameState()
    @Published var rsArucoFrame = FrameState()
    @Published var avpFrame = FrameState()
    @Published var avpArucoFrame = FrameState()
    @Published var avpRSOverlayFrame = FrameState()
    @Published var avpMaskFrame = FrameState()
    @Published var transformedDepthFrame = FrameState()
    @Published var fpAVPonAVPFrame = FrameState()
    @Published var fpAVPonRSFrame = FrameState()
    @Published var fpRSonAVPFrame = FrameState()
    @Published var fpRSonRSFrame = FrameState()
    @Published var intrinsics = IntrinsicsState()
    @Published var transforms = TransformState()
    @Published var poseInAVP = PoseInAVPState()
    @Published var avpBoardPose: [[Double]]?
    @Published var foundationPoseMatrix: [[Double]]?
    @Published var foundationPoseMessage: String?
    @Published var lastError: String?
    @Published var lastUpdate: Date?
    @Published var avpLastFetchTime: Date?

    @Published var saveNextFoundationPose = false
    @Published var processingStride: Int = 1

    @Published var selectedView: String = "overlay"

    private var baseURL: URL?
    private var pollTask: Task<Void, Never>?

    func updateBaseURL(_ url: URL?) {
        baseURL = url
        if pollTask != nil { startPolling() }
    }

    func manualRefresh() {
        Task { await fetchOnce() }
    }

    func toggleSaveNextFoundationPose() {
        saveNextFoundationPose.toggle()
        postSaveNextFoundationPose(enabled: saveNextFoundationPose)
    }

    func startPolling() {
        pollTask?.cancel()
        guard baseURL != nil else { return }
        pollTask = Task { await pollLoop() }
    }

    func stop() {
        pollTask?.cancel()
        pollTask = nil
    }

    private func pollLoop() async {
        while !Task.isCancelled {
            await fetchOnce()
            try? await Task.sleep(for: .seconds(1.5))
        }
    }

    private func fetchOnce() async {
        guard let baseURL else { return }
        var errors: [String] = []

        async let healthTask = fetchHealth(baseURL: baseURL)
        async let rgbdTask = fetchRGBD(baseURL: baseURL)
        async let rsArucoTask = fetchRSAruco(baseURL: baseURL)
        async let avpLatestTask = fetchAVPLatest(baseURL: baseURL)
        async let avpArucoTask = fetchAVPAruco(baseURL: baseURL)
        async let avpRSOverlayTask = fetchAVPRSOverlay(baseURL: baseURL)
        async let avpMaskTask = fetchAVPMask(baseURL: baseURL)
        async let transformedDepthTask = fetchTransformedDepth(baseURL: baseURL)
        async let intrinsicsTask = fetchIntrinsics(baseURL: baseURL)
        async let transformTask = fetchTransforms(baseURL: baseURL)
        async let poseInAVPTask = fetchPoseInAVP(baseURL: baseURL)
        async let foundationPoseTask = fetchFoundationPose(baseURL: baseURL)
        async let strideTask = fetchProcessingStride(baseURL: baseURL)

        do { self.health = try await healthTask } catch { errors.append(error.localizedDescription) }
        do {
            let rgbd = try await rgbdTask
            self.rgbFrame = rgbd.rgb
            self.depthFrame = rgbd.depth
        } catch { errors.append(error.localizedDescription) }
        do { self.rsArucoFrame = try await rsArucoTask } catch { errors.append(error.localizedDescription) }
        do { self.avpFrame = try await avpLatestTask } catch { errors.append(error.localizedDescription) }
        do {
            let avpAruco = try await avpArucoTask
            self.avpArucoFrame = avpAruco
            var t = self.transforms
            t.avpBoard = avpAruco.poseMatrix
            self.transforms = t
            self.avpBoardPose = avpAruco.poseMatrix
        } catch { errors.append(error.localizedDescription) }
        do { self.avpRSOverlayFrame = try await avpRSOverlayTask } catch { errors.append(error.localizedDescription) }
        do { self.avpMaskFrame = try await avpMaskTask } catch { errors.append(error.localizedDescription) }
        do { self.transformedDepthFrame = try await transformedDepthTask } catch { errors.append(error.localizedDescription) }
        do { self.intrinsics = try await intrinsicsTask } catch { errors.append(error.localizedDescription) }

        // Only fetch the currently selected FP view (on-demand)
        await fetchSelectedFPViewIfNeeded()
        do { self.transforms = try await transformTask } catch { errors.append(error.localizedDescription) }
        do { self.poseInAVP = try await poseInAVPTask } catch { errors.append(error.localizedDescription) }
        do {
            let fp = try await foundationPoseTask
            foundationPoseMatrix = fp.poseMatrix
            foundationPoseMessage = fp.message
        } catch { errors.append(error.localizedDescription) }
        do { processingStride = try await strideTask } catch { errors.append(error.localizedDescription) }

        self.lastError = errors.isEmpty ? nil : errors.joined(separator: " | ")
        self.avpLastFetchTime = Date()
        lastUpdate = Date()
    }

    private func fetchAVPData() async {
        guard let baseURL else { return }
        var errors: [String] = []

        async let avpLatestTask = fetchAVPLatest(baseURL: baseURL)
        async let avpArucoTask = fetchAVPAruco(baseURL: baseURL)
        async let transformedDepthTask = fetchTransformedDepth(baseURL: baseURL)
        async let avpRSOverlayTask = fetchAVPRSOverlay(baseURL: baseURL)
        async let avpMaskTask = fetchAVPMask(baseURL: baseURL)
        async let foundationPoseTask = fetchFoundationPose(baseURL: baseURL)
        async let strideTask = fetchProcessingStride(baseURL: baseURL)

        do { self.avpFrame = try await avpLatestTask } catch { errors.append(error.localizedDescription) }
        do {
            let avpAruco = try await avpArucoTask
            self.avpArucoFrame = avpAruco
            var t = self.transforms
            t.avpBoard = avpAruco.poseMatrix
            self.transforms = t
            self.avpBoardPose = avpAruco.poseMatrix
        } catch { errors.append(error.localizedDescription) }
        do { self.transformedDepthFrame = try await transformedDepthTask } catch { errors.append(error.localizedDescription) }
        do { self.avpRSOverlayFrame = try await avpRSOverlayTask } catch { errors.append(error.localizedDescription) }
        do { self.avpMaskFrame = try await avpMaskTask } catch { errors.append(error.localizedDescription) }
        do {
            let fp = try await foundationPoseTask
            foundationPoseMatrix = fp.poseMatrix
            foundationPoseMessage = fp.message
        } catch { errors.append(error.localizedDescription) }
        do { processingStride = try await strideTask } catch { errors.append(error.localizedDescription) }

        self.avpLastFetchTime = Date()
        self.lastError = errors.isEmpty ? nil : errors.joined(separator: " | ")
    }
}

// MARK: - Networking
private extension DebugDashboardModel {
    struct HealthResponse: Decodable { let status: String; let rs_connected: Bool; let calibrated: Bool }
    struct RGBDResponse: Decodable { let rgb: String; let depth: String; let timestamp: Double }
    struct ArucoResponse: Decodable {
        let rgb: String
        let markers_detected: Int?
        let marker_ids: [Int]?
        let timestamp: Double?
        let intrinsics_calculated: Bool?
        let K: [[Double]]?
        let samples_collected: Int?
        let pose_matrix: [[Double]]?
    }
    struct AVPFrameResponse: Decodable { let rgb: String; let timestamp: Double?; let age_seconds: Double?; let width: Int?; let height: Int? }
    struct IntrinsicsResponse: Decodable {
        struct CameraIntrinsics: Decodable { let K: [[Double]]?; let calculated: Bool?; let method: String?; let timestamp: Double? }
        let rs: CameraIntrinsics
        let avp: CameraIntrinsics
    }
    struct TransformResponse: Decodable {
        let T_avp_rs: [[Double]]?
        let T_world_rs: [[Double]]?
        let T_world_avp: [[Double]]?
        let calibrated: Bool?
        let message: String?
    }
    struct PoseInAVPResponse: Decodable {
        let position: [Double]?
        let quaternion: [Double]?
        let T_avp_rs: [[Double]]?
        let head_pose_age: Double?
        let calibrated: Bool?
        let message: String?
    }
    struct FoundationPoseResponse: Decodable {
        let pose_matrix: [[Double]]?
        let message: String?
        let success: Bool?
    }
    struct ProcessingStrideResponse: Decodable {
        let stride: Int
    }
    struct ErrorResponse: Decodable { let error: String }

    func fetchHealth(baseURL: URL) async throws -> HealthState {
        let data = try await performRequest(baseURL: baseURL, path: "health")
        let response = try JSONDecoder().decode(HealthResponse.self, from: data)
        return HealthState(ok: response.status == "ok",
                           rsConnected: response.rs_connected,
                           calibrated: response.calibrated,
                           lastUpdated: Date())
    }

    func fetchRGBD(baseURL: URL) async throws -> (rgb: FrameState, depth: FrameState) {
        let data = try await performRequest(baseURL: baseURL, path: "get_rgbd_frame")
        let response = try JSONDecoder().decode(RGBDResponse.self, from: data)
        let ts = Date(timeIntervalSince1970: response.timestamp)
        let rgbImage = await decodeImage(from: response.rgb)
        let depthImage = await decodeImage(from: response.depth)
        let rgb = FrameState(image: rgbImage,
                             subtitle: "RGB",
                             details: "timestamp \(response.timestamp)",
                             timestamp: ts)
        let depth = FrameState(image: depthImage,
                               subtitle: "Depth",
                               details: "timestamp \(response.timestamp)",
                               timestamp: ts)
        return (rgb, depth)
    }

    func fetchRSAruco(baseURL: URL) async throws -> FrameState {
        let data = try await performRequest(baseURL: baseURL, path: "get_aruco_frame")
        let response = try JSONDecoder().decode(ArucoResponse.self, from: data)
        let ts = response.timestamp.map { Date(timeIntervalSince1970: $0) }
        let markers = response.markers_detected ?? 0
        let ids = (response.marker_ids ?? []).map(String.init).joined(separator: ", ")
        let subtitle = "RS ArUco • \(markers) markers"
        let details = ids.isEmpty ? "No IDs" : ids
        return FrameState(image: await decodeImage(from: response.rgb),
                          subtitle: subtitle,
                          details: details,
                          timestamp: ts)
    }

    func fetchAVPLatest(baseURL: URL) async throws -> FrameState {
        var comps = URLComponents(url: baseURL.appendingPathComponent("get_avp_latest_frame"), resolvingAgainstBaseURL: false)!
        comps.queryItems = [URLQueryItem(name: "purpose", value: "roi_selection")]
        let data = try await performRequest(url: comps.url!)
        let response = try JSONDecoder().decode(AVPFrameResponse.self, from: data)
        let ts = response.timestamp.map { Date(timeIntervalSince1970: $0) }
        let age = response.age_seconds.map { String(format: "%.2fs age", $0) } ?? "—"
        let size = [response.width, response.height].compactMap { $0 }.map(String.init).joined(separator: "×")
        let subtitle = "Latest AVP frame"
        let details = [age, size].filter { !$0.isEmpty }.joined(separator: " • ")
        return FrameState(image: await decodeImage(from: response.rgb),
                          subtitle: subtitle,
                          details: details,
                          timestamp: ts)
    }

    func fetchAVPAruco(baseURL: URL) async throws -> FrameState {
        var comps = URLComponents(url: baseURL.appendingPathComponent("get_avp_aruco_frame"), resolvingAgainstBaseURL: false)!
        comps.queryItems = [URLQueryItem(name: "purpose", value: "aruco_calibration")]
        let data = try await performRequest(url: comps.url!)
        let response = try JSONDecoder().decode(ArucoResponse.self, from: data)
        let ts = response.timestamp.map { Date(timeIntervalSince1970: $0) }
        let markers = response.markers_detected ?? 0
        let ids = (response.marker_ids ?? []).map(String.init).joined(separator: ", ")
        let subtitle = "AVP ArUco • \(markers) markers"
        let details = ids.isEmpty ? "No IDs" : ids
        return FrameState(image: await decodeImage(from: response.rgb),
                          subtitle: subtitle,
                          details: details,
                          timestamp: ts,
                          poseMatrix: response.pose_matrix)
    }

    func fetchAVPRSOverlay(baseURL: URL) async throws -> FrameState {
        let data = try await performRequest(baseURL: baseURL, path: "get_avp_rs_overlay")
        let response = try JSONDecoder().decode(AVPFrameResponse.self, from: data)
        let ts = response.timestamp.map { Date(timeIntervalSince1970: $0) }
        let subtitle = "AVP + RS Overlay"
        return FrameState(image: await decodeImage(from: response.rgb),
                          subtitle: subtitle,
                          details: "",
                          timestamp: ts)
    }

    func fetchAVPMask(baseURL: URL) async throws -> FrameState {
        let data = try await performRequest(baseURL: baseURL, path: "get_avp_mask_frame")
        let response = try JSONDecoder().decode(AVPFrameResponse.self, from: data)
        let ts = response.timestamp.map { Date(timeIntervalSince1970: $0) }
        let subtitle = "AVP Mask"
        return FrameState(image: await decodeImage(from: response.rgb),
                          subtitle: subtitle,
                          details: "",
                          timestamp: ts)
    }

    func fetchIntrinsics(baseURL: URL) async throws -> IntrinsicsState {
        let data = try await performRequest(baseURL: baseURL, path: "get_intrinsics")
        let response = try JSONDecoder().decode(IntrinsicsResponse.self, from: data)
        return IntrinsicsState(rs: response.rs.K, avp: response.avp.K)
    }

    func fetchTransforms(baseURL: URL) async throws -> TransformState {
        let data = try await performRequest(baseURL: baseURL, path: "get_transformation")
        let response = try JSONDecoder().decode(TransformResponse.self, from: data)
        return TransformState(avpRS: response.T_avp_rs,
                              worldRS: response.T_world_rs,
                              worldAVP: response.T_world_avp)
    }

    func fetchPoseInAVP(baseURL: URL) async throws -> PoseInAVPState {
        let data = try await performRequest(baseURL: baseURL, path: "get_rs_pose_in_avp")
        let response = try JSONDecoder().decode(PoseInAVPResponse.self, from: data)
        return PoseInAVPState(matrix: response.T_avp_rs,
                              headPoseAge: response.head_pose_age)
    }

    func fetchFoundationPose(baseURL: URL) async throws -> (poseMatrix: [[Double]]?, message: String?) {
        let data = try await performRequest(baseURL: baseURL, path: "get_foundationpose_pose")
        let response = try JSONDecoder().decode(FoundationPoseResponse.self, from: data)
        return (poseMatrix: response.pose_matrix, message: response.message)
    }

    func fetchProcessingStride(baseURL: URL) async throws -> Int {
        let data = try await performRequest(baseURL: baseURL, path: "processing_stride")
        let response = try JSONDecoder().decode(ProcessingStrideResponse.self, from: data)
        return max(1, response.stride)
    }

    func fetchTransformedDepth(baseURL: URL) async throws -> FrameState {
        let data = try await performRequest(baseURL: baseURL, path: "get_transformed_depth")
        struct Response: Decodable {
            let depth_colormap: String
            let timestamp: Double
            let transformation_applied: Bool
            let min_depth: Double?
            let max_depth: Double?
        }
        let response = try JSONDecoder().decode(Response.self, from: data)
        let ts = Date(timeIntervalSince1970: response.timestamp)
        let transformedInfo = response.transformation_applied ? "Transformed" : "RS view (uncalibrated)"
        let depthRange = [response.min_depth, response.max_depth].compactMap { $0 }.map { String(format: "%.2fm", $0) }.joined(separator: " - ")
        let subtitle = "Transformed Depth"
        let details = [transformedInfo, depthRange].filter { !$0.isEmpty }.joined(separator: " • ")
        return FrameState(image: await decodeImage(from: response.depth_colormap),
                          subtitle: subtitle,
                          details: details,
                          timestamp: ts)
    }

    func fetchSelectedFPViewIfNeeded() async {
        guard baseURL != nil else { return }

        // Only fetch if an FP view is selected
        let fpViews = ["fp_avp_on_avp", "fp_avp_on_rs", "fp_rs_on_avp", "fp_rs_on_rs"]
        guard fpViews.contains(selectedView) else { return }

        // Fetch only the selected view
        do {
            let frame = try await fetchMJPEGFrame(baseURL: baseURL!, view: selectedView)
            switch selectedView {
            case "fp_avp_on_avp": self.fpAVPonAVPFrame = frame
            case "fp_avp_on_rs": self.fpAVPonRSFrame = frame
            case "fp_rs_on_avp": self.fpRSonAVPFrame = frame
            case "fp_rs_on_rs": self.fpRSonRSFrame = frame
            default: break
            }
        } catch {
            // Silently fail - user can manually refresh if needed
        }
    }

    func fetchMJPEGFrame(baseURL: URL, view: String) async throws -> FrameState {
        return try await withThrowingTaskGroup(of: FrameState.self) { group in
            group.addTask {
                var comps = URLComponents(url: baseURL.appendingPathComponent("mjpeg"), resolvingAgainstBaseURL: false)!
                comps.queryItems = [
                    URLQueryItem(name: "view", value: view),
                    URLQueryItem(name: "_ts", value: String(Date().timeIntervalSince1970))
                ]

                // MJPEG stream returns multipart data, we need to extract first frame
                let (asyncBytes, response) = try await URLSession.shared.bytes(from: comps.url!)
                guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                    throw DashboardError.badHTTPStatus((response as? HTTPURLResponse)?.statusCode ?? -1, "Failed to fetch MJPEG")
                }

                // Read MJPEG stream with improved efficiency
                var buffer = Data()
                let maxBytes = 8_000_000 // Allow larger JPEGs from high-res UxPlay feeds.
                var lastCheckSize = 0

                for try await byte in asyncBytes {
                    buffer.append(byte)

                    // Check size limit
                    if buffer.count > maxBytes {
                        throw DashboardError.badHTTPStatus(-1, "MJPEG frame too large")
                    }

                    // Only check for JPEG markers every 1KB to reduce overhead
                    if buffer.count - lastCheckSize >= 1024 {
                        lastCheckSize = buffer.count

                        // Check if we have a complete JPEG
                        if let jpegStart = buffer.range(of: Data([0xFF, 0xD8])),
                           let jpegEnd = buffer.range(of: Data([0xFF, 0xD9]), in: jpegStart.upperBound..<buffer.endIndex) {
                            // Extract the JPEG data
                            let jpegData = buffer[jpegStart.lowerBound...jpegEnd.upperBound]
                            let image = await Task.detached(priority: .utility) {
                                UIImage(data: jpegData)
                            }.value

                            return FrameState(image: image,
                                            subtitle: view,
                                            details: "Fetched from MJPEG stream",
                                            timestamp: Date())
                        }
                    }
                }

                throw DashboardError.badHTTPStatus(-1, "No complete JPEG frame found")
            }

            // Add timeout task
            group.addTask {
                try await Task.sleep(for: .seconds(3))
                throw DashboardError.badHTTPStatus(-1, "MJPEG fetch timeout")
            }

            // Return first result (either frame or timeout)
            let result = try await group.next()!
            group.cancelAll()
            return result
        }
    }

    func performRequest(baseURL: URL, path: String) async throws -> Data {
        let url = baseURL.appendingPathComponent(path)
        return try await performRequest(url: url)
    }

    func performRequest(url: URL) async throws -> Data {
        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }
        guard (200..<300).contains(http.statusCode) else {
            if let err = try? JSONDecoder().decode(ErrorResponse.self, from: data) {
                throw DashboardError.badHTTPStatus(http.statusCode, err.error)
            }
            throw DashboardError.badHTTPStatus(http.statusCode, "Unexpected response")
        }
        return data
    }

    func decodeImage(from dataURLString: String?) async -> UIImage? {
        guard let dataURLString else { return nil }
        let base64Part: String
        if let commaIndex = dataURLString.firstIndex(of: ",") {
            base64Part = String(dataURLString[dataURLString.index(after: commaIndex)...])
        } else {
            base64Part = dataURLString
        }
        return await Task.detached(priority: .utility) {
            guard let data = Data(base64Encoded: base64Part, options: .ignoreUnknownCharacters) else {
                return nil
            }
            return UIImage(data: data)
        }.value
    }

    func postSaveNextFoundationPose(enabled: Bool) {
        guard let baseURL else { return }
        var request = URLRequest(url: baseURL.appendingPathComponent("foundationpose_save_next"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let payload: [String: Any] = ["enabled": enabled]
        request.httpBody = try? JSONSerialization.data(withJSONObject: payload)
        Task {
            _ = try? await URLSession.shared.data(for: request)
        }
    }

    func postProcessingStride(_ stride: Int) {
        guard let baseURL else { return }
        var request = URLRequest(url: baseURL.appendingPathComponent("processing_stride"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let payload: [String: Any] = ["stride": stride]
        request.httpBody = try? JSONSerialization.data(withJSONObject: payload)
        Task {
            _ = try? await URLSession.shared.data(for: request)
        }
    }
}

// MARK: - View
struct DebugDashboardView: View {
    @EnvironmentObject private var appModel: AppModel
    @EnvironmentObject private var sensorModel: SensorDataModel
    @EnvironmentObject private var logs: LogStore
    @StateObject private var model = DebugDashboardModel()
    @State private var autoRefresh = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    connectionStatus
                    viewSelectionSection
                    selectedFrameSection
                    matricesSection
                    processingSection
                    logsSection
                    sensorSection
                    if let error = model.lastError {
                        Text(error)
                            .font(.callout)
                            .foregroundStyle(.red)
                            .padding(.vertical, 4)
                    }
                }
                .padding()
            }
            .navigationTitle("Debug Dashboard")
            .toolbar {
                ToolbarItemGroup(placement: .navigationBarTrailing) {
                    Toggle("Live", isOn: $autoRefresh)
                        .toggleStyle(.switch)
                        .labelsHidden()
                        .onChange(of: autoRefresh) { live in
                            if live { model.startPolling() } else { model.stop() }
                        }
                    Button("Refresh now") { model.manualRefresh() }
                }
            }
            .task {
                model.updateBaseURL(appModel.baseURL)
                autoRefresh = false
            }
            .onChange(of: appModel.baseURL) { model.updateBaseURL($0) }
            .onDisappear { model.stop() }
        }
    }

    private var connectionStatus: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading) {
                    Text(appModel.baseURL?.absoluteString ?? "No API URL")
                        .font(.headline)
                    if let ts = model.lastUpdate {
                        Text("Last poll: \(ts.formatted(date: .omitted, time: .standard))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer()
                Circle()
                    .fill(model.health.ok ? .green : .red)
                    .frame(width: 14, height: 14)
            }
            HStack(spacing: 12) {
                statusPill(label: "API", active: model.health.ok)
                statusPill(label: "RealSense", active: model.health.rsConnected)
                statusPill(label: "Calibrated", active: model.health.calibrated)
            }
        }
        .padding()
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var matricesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Matrices & Intrinsics")
                .font(.headline)
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                MatrixCard(title: "RS Intrinsics", matrix: model.intrinsics.rs)
                MatrixCard(title: "AVP Intrinsics", matrix: model.intrinsics.avp)
                MatrixCard(title: "T_avp_rs", matrix: model.transforms.avpRS)
                MatrixCard(title: "T_world_rs", matrix: model.transforms.worldRS)
                MatrixCard(title: "T_world_avp", matrix: model.transforms.worldAVP)
                MatrixCard(title: "AVP ArUco Pose", matrix: model.transforms.avpBoard ?? model.avpBoardPose)
                MatrixCard(title: "RS Pose in AVP", matrix: model.poseInAVP.matrix)
                MatrixCard(title: "FoundationPose", matrix: model.foundationPoseMatrix)
            }
            if let age = model.poseInAVP.headPoseAge {
                Text("Head pose age: \(String(format: "%.2f s", age))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let msg = model.foundationPoseMessage, model.foundationPoseMatrix == nil {
                Text("FoundationPose: \(msg)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var processingSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Processing")
                .font(.headline)
            VStack(alignment: .leading, spacing: 4) {
                Text("Process every \(model.processingStride) frame(s)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Slider(value: Binding(
                    get: { Double(model.processingStride) },
                    set: { model.processingStride = max(1, Int($0)) }
                ), in: 1...10, step: 1)
                .onChange(of: model.processingStride) { _, newValue in
                    model.postProcessingStride(newValue)
                }
            }
        }
        .padding()
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }


    private var selectedFrameSection: some View {
        let selection = selectedFrame
        return VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Live Frame")
                    .font(.headline)
                Spacer()
                Button(model.saveNextFoundationPose ? "Save next request ✓" : "Save next request") {
                    model.toggleSaveNextFoundationPose()
                }
                .buttonStyle(.bordered)
            }
            if let avpFetchTime = model.avpLastFetchTime {
                let age = Date().timeIntervalSince(avpFetchTime)
                Text("AVP frames age: \(String(format: "%.1fs", age))")
                    .font(.caption)
                    .foregroundColor(age < 5 ? .secondary : .orange)
            }
            FrameCard(title: selection.title, state: selection.state)
        }
        .padding()
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var viewSelectionSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Live View Selection")
                .font(.headline)
            Picker("View", selection: $model.selectedView) {
                Text("AVP ArUco Overlay").tag("overlay")
                Text("AVP Raw (UxPlay)").tag("raw")
                Text("AVP ROI Mask").tag("mask")
                Text("RS RGB").tag("rs_rgb")
                Text("RS Depth").tag("rs_depth")
                Text("RS ArUco Overlay").tag("rs_aruco")
                Text("RS ROI").tag("rs_roi")
                Text("AVP + RS Pose Overlay").tag("avp_rs")
                Text("RS Depth → AVP").tag("avp_depth")
            }
            .pickerStyle(.menu)

            Text("Selected: \(viewLabel(for: model.selectedView))")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding()
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func viewLabel(for view: String) -> String {
        switch view {
        case "overlay": return "AVP ArUco Overlay"
        case "raw": return "AVP Raw (UxPlay)"
        case "mask": return "AVP ROI Mask"
        case "rs_rgb": return "RS RGB"
        case "rs_depth": return "RS Depth"
        case "rs_aruco": return "RS ArUco Overlay"
        case "rs_roi": return "RS ROI"
        case "avp_rs": return "AVP + RS Pose Overlay"
        case "avp_depth": return "RS Depth → AVP"
        default: return view
        }
    }

    private var selectedFrame: (title: String, state: DebugDashboardModel.FrameState) {
        switch model.selectedView {
        case "overlay":
            return ("AVP ArUco Overlay", model.avpArucoFrame)
        case "raw":
            return ("AVP Raw (UxPlay)", model.avpFrame)
        case "mask":
            return ("AVP ROI Mask", model.avpMaskFrame)
        case "rs_rgb":
            return ("RS RGB", model.rgbFrame)
        case "rs_depth":
            return ("RS Depth", model.depthFrame)
        case "rs_aruco":
            return ("RS ArUco Overlay", model.rsArucoFrame)
        case "rs_roi":
            return ("RS ROI", DebugDashboardModel.FrameState(subtitle: "Not available in dashboard"))
        case "avp_rs":
            return ("AVP + RS Pose Overlay", model.avpRSOverlayFrame)
        case "avp_depth":
            return ("RS Depth → AVP", model.transformedDepthFrame)
        default:
            return (viewLabel(for: model.selectedView), DebugDashboardModel.FrameState(subtitle: "No data"))
        }
    }

    private var logsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Logs").font(.headline)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(logs.lines.suffix(120).enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(.caption, design: .monospaced))
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(.vertical, 4)
            }
            .frame(maxHeight: 220)
        }
        .padding()
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private var sensorSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Head Pose (ARKit)")
                .font(.headline)
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Position (m)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(vectorString(sensorModel.headPosition))
                        .font(.body.monospacedDigit())
                }
                Spacer()
                VStack(alignment: .leading, spacing: 4) {
                    Text("Euler (deg)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(vectorString(sensorModel.headEulerDegrees))
                        .font(.body.monospacedDigit())
                }
                Spacer()
                VStack(alignment: .leading, spacing: 4) {
                    Text("Frames")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("\(sensorModel.frameCount)")
                        .font(.body.monospacedDigit())
                }
            }
            if sensorModel.lastPoseUpdate > .distantPast {
                Text("Last ARKit update \(sensorModel.lastPoseUpdate.formatted(date: .omitted, time: .standard))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    private func statusPill(label: String, active: Bool) -> some View {
        HStack(spacing: 6) {
            Circle()
                .fill(active ? .green : .red)
                .frame(width: 10, height: 10)
            Text(label)
                .font(.caption)
                .foregroundStyle(.primary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.ultraThinMaterial)
        .clipShape(Capsule())
    }

    private func vectorString(_ v: SIMD3<Double>) -> String {
        "\(String(format: "%.3f", v.x)), \(String(format: "%.3f", v.y)), \(String(format: "%.3f", v.z))"
    }
}

// MARK: - Subviews
private struct FrameCard: View {
    let title: String
    let state: DebugDashboardModel.FrameState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            if let image = state.image {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(minHeight: 200)
                    .cornerRadius(10)
                    .shadow(radius: 2)
            } else {
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.secondary.opacity(0.15))
                    Text("No frame")
                        .foregroundStyle(.secondary)
                }
                .frame(height: 200)
            }
            Text(state.subtitle)
                .font(.subheadline)
            Text(state.details)
                .font(.caption)
                .foregroundStyle(.secondary)
            if let ts = state.timestamp {
                Text(ts.formatted(date: .omitted, time: .standard))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

private struct MatrixCard: View {
    let title: String
    let matrix: [[Double]]?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.subheadline)
            if let matrix {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(0..<matrix.count, id: \.self) { row in
                        Text(matrix[row].map { String(format: "%.3f", $0) }.joined(separator: "  "))
                            .font(.caption.monospacedDigit())
                    }
                }
            } else {
                Text("No data")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}
