import SwiftUI

struct ContentView: View {
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    @Environment(\.openWindow) private var openWindow
    @EnvironmentObject private var appModel: AppModel
    @EnvironmentObject private var sensorModel: SensorDataModel
    @EnvironmentObject private var arucoStream: ArucoStreamModel
    @EnvironmentObject private var rsPoseModel: RealSensePoseModel
    @EnvironmentObject private var foundationPoseModel: FoundationPoseModel
    @EnvironmentObject private var logs: LogStore

    @AppStorage("poseoverlay.apiHost") private var storedHost = "127.0.0.1"
    @AppStorage("poseoverlay.apiPort") private var storedPort = "8000"

    @State private var hostField = ""
    @State private var portField = ""
    @State private var statusText = "Not connected"
    @State private var isOpeningImmersive = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Aruco Monitor")
                .font(.title2)
                .bold()

            connectionControls
            modelSection
            statusSection
            windowsSection
            Spacer(minLength: 0)
        }
        .padding()
        .task { initializeConnection() }
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

    private var modelSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("3D Model")
                .font(.headline)
            HStack(spacing: 12) {
                Menu {
                    ForEach(appModel.availableModels, id: \.self) { name in
                        Button(name) { handleModelSelection(name) }
                    }
                } label: {
                    HStack(spacing: 6) {
                        Text(modelMenuLabel)
                            .foregroundStyle(appModel.availableModels.isEmpty ? .secondary : .primary)
                        Image(systemName: "chevron.up.chevron.down")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
                .disabled(appModel.isLoadingModels || appModel.availableModels.isEmpty)

                Button("Reload") { Task { await loadModels() } }
                    .disabled(appModel.isLoadingModels)
            }
            if let err = appModel.lastModelError {
                Text(err)
                    .font(.footnote)
                    .foregroundStyle(.red)
            }
        }
    }

    private var windowsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Windows")
                .font(.headline)
            HStack(spacing: 12) {
                Button("ROI Window") { openWindow(id: "roi") }
                Button("Debug Viewer") { openWindow(id: "debug") }
                Button("Help") { openWindow(id: "help") }
            }
        }
    }

    // MARK: - Actions
    private func initializeConnection() {
        hostField = storedHost
        portField = storedPort
        Task {
            await applyConnection()
            await MainActor.run {
                sensorModel.start()
            }
            await loadModelsIfNeeded()
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
        rsPoseModel.updateBaseURL(base)
        rsPoseModel.startPolling()
        foundationPoseModel.updateBaseURL(base)
        foundationPoseModel.startPolling()
        logs.add("API base set to \(urlString)")
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

    private var modelMenuLabel: String {
        if appModel.isLoadingModels { return "Loading…" }
        if let selection = appModel.selectedModel, !selection.isEmpty { return selection }
        if appModel.availableModels.isEmpty { return "No models" }
        return "Select model"
    }

    private func handleModelSelection(_ name: String) {
        appModel.setSelectedModel(name)
        Task { await postSelectionIfNeeded() }
    }

    private func loadModelsIfNeeded() async {
        guard appModel.availableModels.isEmpty else { return }
        await loadModels()
    }

    private func loadModels() async {
        guard let base = appModel.baseURL else {
            appModel.lastModelError = "Set a valid API URL first"
            return
        }
        await MainActor.run {
            appModel.isLoadingModels = true
            appModel.lastModelError = nil
        }
        do {
            let list = try await ModelService.fetchModelList(baseURL: base)
            await MainActor.run {
                appModel.setAvailableModels(list)
                appModel.isLoadingModels = false
                if appModel.selectedModel == nil, let first = list.first {
                    appModel.setSelectedModel(first)
                }
            }
            if let selected = appModel.selectedModel {
                try? await ModelService.selectModel(baseURL: base, name: selected)
            }
        } catch {
            await MainActor.run {
                appModel.isLoadingModels = false
                appModel.lastModelError = error.localizedDescription
            }
        }
    }

    private func postSelectionIfNeeded() async {
        guard let name = appModel.selectedModel,
              let base = appModel.baseURL else { return }
        do {
            try await ModelService.selectModel(baseURL: base, name: name)
        } catch {
            await MainActor.run { appModel.lastModelError = error.localizedDescription }
        }
    }
}
