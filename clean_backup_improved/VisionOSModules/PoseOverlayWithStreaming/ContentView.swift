import SwiftUI

struct ContentView: View {
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    @EnvironmentObject private var appModel: AppModel
    @EnvironmentObject private var sensorModel: SensorDataModel
    @EnvironmentObject private var arucoStream: ArucoStreamModel

    @AppStorage("poseoverlay.apiHost") private var storedHost = "127.0.0.1"
    @AppStorage("poseoverlay.apiPort") private var storedPort = "8000"

    @State private var hostField = ""
    @State private var portField = ""
    @State private var statusText = "Not connected"
    @State private var isOpeningImmersive = false

    private let timestampFormatter: DateFormatter = {
        let f = DateFormatter()
        f.timeStyle = .medium
        f.dateStyle = .none
        return f
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Aruco Monitor")
                .font(.title2)
                .bold()

            connectionControls
            feedView
            statusSection
            Spacer(minLength: 0)
        }
        .padding()
        .task { await initializeConnection() }
    }

    // MARK: - Sections
    private var connectionControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Host").font(.caption).foregroundStyle(.secondary)
                    TextField("127.0.0.1", text: $hostField)
                        .textInputAutocapitalization(.never)
                        .disableAutocorrection(true)
                        .textFieldStyle(.roundedBorder)
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Port").font(.caption).foregroundStyle(.secondary)
                    TextField("8000", text: $portField)
                        .keyboardType(.numberPad)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 120)
                }
                Button("Connect") { Task { await applyConnection() } }
                    .buttonStyle(.borderedProminent)
            }

            HStack(spacing: 12) {
                Button(appModel.immersiveSpacePresented ? "Close Immersive Space" : "Open Immersive Space") {
                    Task { await toggleImmersiveSpace() }
                }
                .buttonStyle(.bordered)
                .disabled(isOpeningImmersive)

                if let base = appModel.baseURL {
                    Text(base.absoluteString)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                } else {
                    Text("Set a valid URL to start")
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
            }
        }
    }

    private var feedView: some View {
        ZStack(alignment: .bottomLeading) {
            if let image = arucoStream.annotatedImage {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: .infinity)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.primary.opacity(0.1), lineWidth: 1)
                    )
            } else {
                VStack(spacing: 12) {
                    ProgressView()
                    Text("Waiting for RGB feed…")
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, minHeight: 280)
                .background(.ultraThinMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 12))
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(feedStatusLine)
                    .font(.footnote)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.ultraThinMaterial)
                    .clipShape(Capsule())
                if let ts = arucoStream.lastTimestamp {
                    Text("Updated \(timestampFormatter.string(from: ts))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(.ultraThinMaterial)
                        .clipShape(Capsule())
                }
            }
            .padding(12)
        }
        .frame(maxHeight: 420)
    }

    private var statusSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(statusText)
                .font(.body)
            Text("Stream: \(arucoStream.status)")
                .font(.footnote)
                .foregroundStyle(.secondary)
            if let pose = arucoStream.latestPose {
                Text("Detection: \(pose.isDetected ? "Found \(pose.numMarkers) markers" : "Searching")")
                    .font(.footnote)
                if !pose.markerIDs.isEmpty {
                    Text("IDs: \(pose.markerIDs.map(String.init).joined(separator: ", "))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Text("Head pose uploads: \(sensorModel.headPoseUploadCount)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    // MARK: - Actions
    private func initializeConnection() async {
        hostField = storedHost
        portField = storedPort
        await applyConnection()
        await MainActor.run {
            sensorModel.start()
        }
    }

    private func applyConnection() async {
        let host = sanitizeHost(hostField)
        let port = sanitizePort(portField)
        let urlString = "http://\(host):\(port)"
        guard let base = URL(string: urlString) else {
            statusText = "Invalid URL"
            return
        }

        storedHost = host
        storedPort = port
        hostField = host
        portField = port

        await MainActor.run {
            appModel.updateBaseURL(urlString)
            statusText = "Connecting to \(urlString)"
        }

        sensorModel.setAPIBaseURL(base)
        await MainActor.run {
            arucoStream.startStreaming(baseURL: base)
            statusText = "Connected"
        }
    }

    private func toggleImmersiveSpace() async {
        guard !isOpeningImmersive else { return }
        isOpeningImmersive = true
        defer { isOpeningImmersive = false }

        if appModel.immersiveSpacePresented {
            await dismissImmersiveSpace()
            await MainActor.run { appModel.setImmersiveSpacePresented(false) }
        } else {
            do {
                let result = try await openImmersiveSpace(id: "PoseSpace")
                if case .opened = result {
                    await MainActor.run { appModel.setImmersiveSpacePresented(true) }
                }
            } catch {
                await MainActor.run { statusText = "Immersive space error: \(error.localizedDescription)" }
            }
        }
    }

    private var feedStatusLine: String {
        if let pose = arucoStream.latestPose, pose.isDetected {
            return "ArUco detected (\(pose.numMarkers) markers)"
        }
        return "No detection yet"
    }

    private func sanitizeHost(_ raw: String) -> String {
        var host = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if host.hasPrefix("http://") {
            host = String(host.dropFirst(7))
        } else if host.hasPrefix("https://") {
            host = String(host.dropFirst(8))
        }
        host = host.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        return host.isEmpty ? "127.0.0.1" : host
    }

    private func sanitizePort(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let value = Int(trimmed), value > 0 && value < 65536 {
            return "\(value)"
        }
        return "8000"
    }
}
