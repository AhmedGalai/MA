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
    @Published var intrinsics = IntrinsicsState()
    @Published var transforms = TransformState()
    @Published var poseInAVP = PoseInAVPState()
    @Published var avpBoardPose: [[Double]]?
    @Published var lastError: String?
    @Published var lastUpdate: Date?
    @Published var avpLastFetchTime: Date?

    // HSV parameters (mean color + std deviation)
    @Published var hsvMeanH: Int = 90
    @Published var hsvMeanS: Int = 255
    @Published var hsvMeanV: Int = 255
    @Published var hsvMeanColor: Color = .cyan
    @Published var hsvStdH: Int = 10
    @Published var hsvStdS: Int = 40
    @Published var hsvStdV: Int = 40

    private var baseURL: URL?
    private var pollTask: Task<Void, Never>?
    private var suppressHSVPost = false

    func updateBaseURL(_ url: URL?) {
        baseURL = url
        if pollTask != nil { startPolling() }
    }

    func manualRefresh() {
        Task { await fetchOnce() }
    }

    func fetchAVPFramesManually() {
        Task { await fetchAVPData() }
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
        async let hsvTask = fetchHSVConfig(baseURL: baseURL)

        do { self.health = try await healthTask } catch { errors.append(error.localizedDescription) }
        do {
            let rgbd = try await rgbdTask
            self.rgbFrame = rgbd.rgb
            self.depthFrame = rgbd.depth
        } catch { errors.append(error.localizedDescription) }
        do { self.rsArucoFrame = try await rsArucoTask } catch { errors.append(error.localizedDescription) }
        do { self.avpFrame = try await avpLatestTask } catch { errors.append(error.localizedDescription) }
        do { self.avpArucoFrame = try await avpArucoTask } catch { errors.append(error.localizedDescription) }
        do { self.avpRSOverlayFrame = try await avpRSOverlayTask } catch { errors.append(error.localizedDescription) }
        do { self.avpMaskFrame = try await avpMaskTask } catch { errors.append(error.localizedDescription) }
        do { self.transformedDepthFrame = try await transformedDepthTask } catch { errors.append(error.localizedDescription) }
        do { self.intrinsics = try await intrinsicsTask } catch { errors.append(error.localizedDescription) }
        do { self.transforms = try await transformTask } catch { errors.append(error.localizedDescription) }
        do { self.poseInAVP = try await poseInAVPTask } catch { errors.append(error.localizedDescription) }
        do { applyHSVConfig(try await hsvTask) } catch { errors.append(error.localizedDescription) }

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
        async let hsvTask = fetchHSVConfig(baseURL: baseURL)

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
        do { applyHSVConfig(try await hsvTask) } catch { errors.append(error.localizedDescription) }

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
    struct HSVConfigResponse: Decodable {
        let mean_h: Int
        let mean_s: Int
        let mean_v: Int
        let std_h: Int
        let std_s: Int
        let std_v: Int
        let enabled: Bool
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
        let rgbImage = decodeImage(from: response.rgb)
        let depthImage = decodeImage(from: response.depth)
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
        return FrameState(image: decodeImage(from: response.rgb),
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
        return FrameState(image: decodeImage(from: response.rgb),
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
        return FrameState(image: decodeImage(from: response.rgb),
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
        return FrameState(image: decodeImage(from: response.rgb),
                          subtitle: subtitle,
                          details: "",
                          timestamp: ts)
    }

    func fetchAVPMask(baseURL: URL) async throws -> FrameState {
        let data = try await performRequest(baseURL: baseURL, path: "get_avp_mask_frame")
        let response = try JSONDecoder().decode(AVPFrameResponse.self, from: data)
        let ts = response.timestamp.map { Date(timeIntervalSince1970: $0) }
        let subtitle = "AVP Mask"
        return FrameState(image: decodeImage(from: response.rgb),
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
        return FrameState(image: decodeImage(from: response.depth_colormap),
                          subtitle: subtitle,
                          details: details,
                          timestamp: ts)
    }

    func fetchHSVConfig(baseURL: URL) async throws -> HSVConfigResponse {
        let data = try await performRequest(baseURL: baseURL, path: "hsv_config")
        return try JSONDecoder().decode(HSVConfigResponse.self, from: data)
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

    func decodeImage(from dataURLString: String?) -> UIImage? {
        guard let dataURLString else { return nil }
        let base64Part: String
        if let commaIndex = dataURLString.firstIndex(of: ",") {
            base64Part = String(dataURLString[dataURLString.index(after: commaIndex)...])
        } else {
            base64Part = dataURLString
        }
        guard let data = Data(base64Encoded: base64Part, options: .ignoreUnknownCharacters) else { return nil }
        return UIImage(data: data)
    }

    func applyHSVConfig(_ config: HSVConfigResponse) {
        suppressHSVPost = true
        hsvMeanH = config.mean_h
        hsvMeanS = config.mean_s
        hsvMeanV = config.mean_v
        hsvStdH = config.std_h
        hsvStdS = config.std_s
        hsvStdV = config.std_v
        hsvMeanColor = colorFromOpenCVHSV(h: config.mean_h, s: config.mean_s, v: config.mean_v)
        DispatchQueue.main.async { [weak self] in
            self?.suppressHSVPost = false
        }
    }

    func postHSVConfig(meanH: Int, meanS: Int, meanV: Int, stdH: Int, stdS: Int, stdV: Int) {
        guard let baseURL else { return }
        var request = URLRequest(url: baseURL.appendingPathComponent("hsv_config"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let payload: [String: Any] = [
            "mean_h": meanH,
            "mean_s": meanS,
            "mean_v": meanV,
            "std_h": stdH,
            "std_s": stdS,
            "std_v": stdV
        ]
        request.httpBody = try? JSONSerialization.data(withJSONObject: payload)
        Task {
            _ = try? await URLSession.shared.data(for: request)
        }
    }

    func updateHSVFromColor(_ color: Color) {
        guard !suppressHSVPost else { return }
        guard let (h, s, v) = openCVHSV(from: color) else { return }
        hsvMeanH = h
        hsvMeanS = s
        hsvMeanV = v
        postHSVConfig(meanH: h, meanS: s, meanV: v, stdH: hsvStdH, stdS: hsvStdS, stdV: hsvStdV)
    }

    func updateHSVStd() {
        guard !suppressHSVPost else { return }
        postHSVConfig(meanH: hsvMeanH, meanS: hsvMeanS, meanV: hsvMeanV, stdH: hsvStdH, stdS: hsvStdS, stdV: hsvStdV)
    }

    func openCVHSV(from color: Color) -> (Int, Int, Int)? {
        let uiColor = UIColor(color)
        var h: CGFloat = 0
        var s: CGFloat = 0
        var v: CGFloat = 0
        var a: CGFloat = 0
        guard uiColor.getHue(&h, saturation: &s, brightness: &v, alpha: &a) else { return nil }
        return (Int(round(h * 179.0)),
                Int(round(s * 255.0)),
                Int(round(v * 255.0)))
    }

    func colorFromOpenCVHSV(h: Int, s: Int, v: Int) -> Color {
        let hue = Double(max(0, min(179, h))) / 179.0
        let sat = Double(max(0, min(255, s))) / 255.0
        let val = Double(max(0, min(255, v))) / 255.0
        return Color(hue: hue, saturation: sat, brightness: val)
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
                    matricesSection
                    framesSection
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
            MatrixCard(title: "RS Intrinsics", matrix: model.intrinsics.rs)
            MatrixCard(title: "AVP Intrinsics", matrix: model.intrinsics.avp)
            MatrixCard(title: "T_avp_rs", matrix: model.transforms.avpRS)
            MatrixCard(title: "T_world_rs", matrix: model.transforms.worldRS)
            MatrixCard(title: "T_world_avp", matrix: model.transforms.worldAVP)
            MatrixCard(title: "AVP ArUco Pose", matrix: model.transforms.avpBoard ?? model.avpBoardPose)
            MatrixCard(title: "RS Pose in AVP", matrix: model.poseInAVP.matrix)
            if let age = model.poseInAVP.headPoseAge {
                Text("Head pose age: \(String(format: "%.2f s", age))")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .padding()
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }


    private var framesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Frames")
                    .font(.headline)
                Spacer()
                Button("Fetch AVP Frames") {
                    model.fetchAVPFramesManually()
                }
                .buttonStyle(.bordered)
                .disabled(model.health.ok == false)
            }
            if let avpFetchTime = model.avpLastFetchTime {
                let age = Date().timeIntervalSince(avpFetchTime)
                Text("AVP frames age: \(String(format: "%.1fs", age))")
                    .font(.caption)
                    .foregroundColor(age < 5 ? .secondary : .orange)
            }
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                FrameCard(title: "RS RGB", state: model.rgbFrame)
                FrameCard(title: "RS Depth", state: model.depthFrame)
                FrameCard(title: "RS ArUco", state: model.rsArucoFrame)
                FrameCard(title: "AVP Raw", state: model.avpFrame)
                FrameCard(title: "AVP ArUco", state: model.avpArucoFrame)
                FrameCard(title: "AVP Mask", state: model.avpMaskFrame)
                FrameCard(title: "AVP + RS", state: model.avpRSOverlayFrame)
                FrameCard(title: "Transformed Depth", state: model.transformedDepthFrame)
            }
        }
        .padding()
        .background(.thinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 12))
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
    }
}
