import SwiftUI

struct ContentView: View {
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    @Environment(\.openWindow) private var openWindow
    @EnvironmentObject private var settings: ArrowSettings
    @EnvironmentObject private var logs: LogStore
    @EnvironmentObject private var appModel: AppModel
    @EnvironmentObject private var sensorModel: SensorDataModel

    @AppStorage("poseoverlay.apiHost") private var storedHost = "127.0.0.1"
    @AppStorage("poseoverlay.apiPort") private var storedPort = "8000"

    @State private var hostField = ""
    @State private var portField = ""
    @State private var errorText: String?
    @State private var hasScheduledImmersivePresentation = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("Pose Overlay Control")
                    .font(.title2)
                    .bold()

                apiSection
                modelSection
                appearanceSection
                roiSection
                calibrationSection
                windowsSection

                if let errorText {
                    Text(errorText)
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
            }
            .padding()
        }
        .task {
            hostField = storedHost
            portField = storedPort
            await MainActor.run {
                appModel.updateBaseURL(buildURLString())
            }
            await sensorModel.setAPIBaseURL(appModel.baseURL)
            await MainActor.run {
                sensorModel.start()
            }
            await loadModelsIfNeeded()
            requestInitialImmersiveSpace()
        }
        .onAppear { requestInitialImmersiveSpace() }
        .onDisappear {
            hasScheduledImmersivePresentation = false
            Task { await dismissImmersiveSpaceIfNeeded() }
        }
    }

    // MARK: - UI Sections
    private var apiSection: some View {
        GroupBox("Main API") {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Host").font(.caption).foregroundStyle(.secondary)
                        TextField("127.0.0.1", text: $hostField)
                            .textInputAutocapitalization(.never)
                            .disableAutocorrection(true)
                            .keyboardType(.URL)
                            .textFieldStyle(.roundedBorder)
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Port").font(.caption).foregroundStyle(.secondary)
                        TextField("8000", text: $portField)
                            .keyboardType(.numberPad)
                            .textFieldStyle(.roundedBorder)
                    }
                }
                HStack {
                    Button("Apply") {
                        Task { await applyBaseURL(triggerReload: true) }
                    }
                    Button("Ping") {
                        Task { await healthCheck() }
                    }
                    Spacer()
                    if let url = appModel.baseURL {
                        Text(url.absoluteString)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    } else {
                        Text("Invalid URL")
                            .font(.footnote)
                            .foregroundStyle(.red)
                    }
                }
            }
        }
    }

    private var modelSection: some View {
        GroupBox("Model Selection") {
            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("3D Model")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        Menu {
                            ForEach(appModel.availableModels, id: \.self) { name in
                                Button {
                                    handleModelSelection(name)
                                } label: {
                                    if appModel.selectedModel == name {
                                        Label(name, systemImage: "checkmark")
                                    } else {
                                        Text(name)
                                    }
                                }
                            }
                        } label: {
                            HStack(spacing: 6) {
                                Text(modelMenuLabel)
                                    .foregroundStyle(appModel.availableModels.isEmpty ? .secondary : .primary)
                                    .lineLimit(1)
                                Spacer(minLength: 0)
                                Image(systemName: "chevron.up.chevron.down")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(.horizontal, 12)
                            .padding(.vertical, 10)
                            .frame(minWidth: 180, alignment: .leading)
                            .background(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(.quaternary)
                            )
                        }
                        .disabled(appModel.isLoadingModels || appModel.availableModels.isEmpty)
                        .accessibilityLabel("3D Model")
                    }
                    Button("Reload") {
                        Task { await loadModels() }
                    }
                    .disabled(appModel.isLoadingModels)
                }
                if appModel.isLoadingModels {
                    ProgressView().progressViewStyle(.circular)
                }
                if let err = appModel.lastModelError {
                    Text(err)
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
            }
        }
    }

    private var appearanceSection: some View {
        GroupBox("Appearance") {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Arrow")
                    Spacer()
                    ColorPicker("", selection: $settings.color, supportsOpacity: false)
                        .labelsHidden()
                }
                HStack {
                    Text("ROI Tint")
                    Spacer()
                    ColorPicker("", selection: $settings.roiColor, supportsOpacity: true)
                        .labelsHidden()
                }
            }
        }
    }

    private var roiSection: some View {
        GroupBox("ROI Size") {
            VStack(alignment: .leading, spacing: 8) {
                Slider(value: Binding(
                    get: { Double(settings.roiRadius) },
                    set: { newVal in
                        let r = CGFloat(newVal)
                        settings.roiRadius = max(12, min(r, 400))
                        logs.add("ROI radius set: \(Int(settings.roiRadius))")
                    }
                ), in: 12...400, step: 1)
                HStack {
                    Text("12").font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Text("\(Int(settings.roiRadius)) pt").font(.caption)
                    Spacer()
                    Text("400").font(.caption).foregroundStyle(.secondary)
                }
            }
        }
    }

    private var calibrationSection: some View {
        GroupBox("Calibration") {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 12) {
                    Button("Send ArUco Frame") {
                        Task { await sendCalibrationFrame(purpose: "aruco_calibration") }
                    }
                    .disabled(appModel.baseURL == nil)

                    Button("Send ROI Frame") {
                        Task { await sendCalibrationFrame(purpose: "roi_selection") }
                    }
                    .disabled(appModel.baseURL == nil)
                }
                Text("Capture and send current view to API for calibration")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var windowsSection: some View {
        GroupBox("Windows") {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Button("Open ROI Window") { openWindow(id: "roi") }
                    Button("Open Logs Window") { openWindow(id: "logs") }
                }
                Button("Open Sensor Monitor") { openWindow(id: "sensors") }
            }
        }
    }

    // MARK: - Actions
    private func applyBaseURL(triggerReload: Bool) async {
        let normalizedHost = sanitizeHost(hostField)
        let normalizedPort = sanitizePort(portField)
        let urlString = "http://\(normalizedHost):\(normalizedPort)"

        storedHost = normalizedHost
        storedPort = normalizedPort
        hostField = normalizedHost
        portField = normalizedPort

        await MainActor.run {
            appModel.updateBaseURL(urlString)
        }
        await sensorModel.setAPIBaseURL(appModel.baseURL)
        logs.add("API base set to \(urlString)")
        if triggerReload { await loadModels() }
    }

    private func loadModelsIfNeeded() async {
        guard appModel.availableModels.isEmpty else { return }
        await loadModels()
    }

    private var modelMenuLabel: String {
        if appModel.isLoadingModels { return "Loading…" }
        if let selection = appModel.selectedModel, !selection.isEmpty { return selection }
        if appModel.availableModels.isEmpty { return "No models found" }
        return "Select a model"
    }

    private func handleModelSelection(_ rawName: String) {
        let trimmed = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed != appModel.selectedModel else { return }
        appModel.setSelectedModel(trimmed)
        Task { await postSelectionIfNeeded() }
    }

    private func buildURLString() -> String {
        let host = sanitizeHost(hostField)
        let port = sanitizePort(portField)
        return "http://\(host):\(port)"
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

    private func loadModels() async {
        guard let base = appModel.baseURL else {
            await MainActor.run { appModel.lastModelError = "Set a valid API URL first" }
            return
        }
        await MainActor.run {
            appModel.isLoadingModels = true
            appModel.lastModelError = nil
        }
        logs.add("GET /models → \(base.absoluteString)")
        do {
            let list = try await ModelService.fetchModelList(baseURL: base)
            await MainActor.run {
                appModel.setAvailableModels(list)
                appModel.isLoadingModels = false
                if appModel.selectedModel == nil, let first = list.first {
                    appModel.setSelectedModel(first)
                }
            }
            if let name = appModel.selectedModel, !name.isEmpty {
                try? await ModelService.selectModel(baseURL: base, name: name)
            }
            logs.add("GET /models ✓ \(list.count) models")
        } catch {
            await MainActor.run {
                appModel.isLoadingModels = false
                appModel.lastModelError = error.localizedDescription
                errorText = error.localizedDescription
            }
            logs.add("GET /models ✗ \(error.localizedDescription)")
        }
    }

    private func postSelectionIfNeeded() async {
        guard let name = appModel.selectedModel,
              !name.isEmpty,
              let base = appModel.baseURL else { return }
        logs.add("POST /select_model → \(name)")
        do {
            try await ModelService.selectModel(baseURL: base, name: name)
            logs.add("Selected model: \(name)")
        } catch {
            await MainActor.run { errorText = "Failed to set model: \(error.localizedDescription)" }
            logs.add("POST /select_model ✗ \(error.localizedDescription)")
        }
    }

    private func requestInitialImmersiveSpace() {
        guard !hasScheduledImmersivePresentation else { return }
        hasScheduledImmersivePresentation = true
        Task {
            await ensureImmersiveSpacePresented()
        }
    }

    private func ensureImmersiveSpacePresented() async {
        guard !appModel.immersiveSpacePresented else { return }
        do {
            let result = try await openImmersiveSpace(id: "PoseSpace")
            if case .opened = result {
                await MainActor.run {
                    appModel.setImmersiveSpacePresented(true)
                }
                logs.add("Immersive space opened")
            } else {
                logs.add("Immersive space request was cancelled")
            }
        } catch {
            await MainActor.run { errorText = error.localizedDescription }
            logs.add("Open immersive failed: \(error.localizedDescription)")
        }
    }

    private func dismissImmersiveSpaceIfNeeded() async {
        guard appModel.immersiveSpacePresented else { return }
        await dismissImmersiveSpace()
        await MainActor.run {
            appModel.setImmersiveSpacePresented(false)
            hasScheduledImmersivePresentation = false
        }
        logs.add("Immersive space dismissed")
    }

    private func healthCheck() async {
        guard let base = appModel.baseURL else {
            await MainActor.run { errorText = "Invalid API URL" }
            return
        }
        let url = base.appendingPathComponent("health")
        logs.add("GET /health → \(url.absoluteString)")
        do {
            var req = URLRequest(url: url)
            req.httpMethod = "GET"
            let (_, resp) = try await URLSession.shared.data(for: req)
            if let http = resp as? HTTPURLResponse, http.statusCode == 200 {
                await MainActor.run { errorText = nil }
                logs.add("Health check OK")
            } else {
                await MainActor.run { errorText = "Health check failed" }
                logs.add("Health check failed")
            }
        } catch {
            await MainActor.run { errorText = error.localizedDescription }
            logs.add("GET /health ✗ \(error.localizedDescription)")
        }
    }

    private func sendCalibrationFrame(purpose: String) async {
        guard let base = appModel.baseURL else {
            await MainActor.run { errorText = "Invalid API URL" }
            logs.add("Calibration frame send failed: No API URL")
            return
        }

        // Build the URL with purpose query parameter
        var components = URLComponents(url: base.appendingPathComponent("capture_frame"), resolvingAgainstBaseURL: false)
        components?.queryItems = [URLQueryItem(name: "purpose", value: purpose)]

        guard let url = components?.url else {
            await MainActor.run { errorText = "Failed to build capture URL" }
            logs.add("Calibration frame send failed: Invalid URL")
            return
        }

        logs.add("POST /capture_frame?purpose=\(purpose) → \(url.absoluteString)")

        do {
            var req = URLRequest(url: url)
            req.httpMethod = "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")

            let (data, resp) = try await URLSession.shared.data(for: req)

            if let http = resp as? HTTPURLResponse {
                if http.statusCode == 200 {
                    await MainActor.run { errorText = nil }
                    logs.add("✓ Calibration frame sent (\(purpose))")
                } else {
                    let errorMsg = String(data: data, encoding: .utf8) ?? "Unknown error"
                    await MainActor.run { errorText = "Capture failed: HTTP \(http.statusCode)" }
                    logs.add("✗ Calibration frame failed: \(errorMsg)")
                }
            }
        } catch {
            await MainActor.run { errorText = "Capture error: \(error.localizedDescription)" }
            logs.add("✗ Calibration frame error: \(error.localizedDescription)")
        }
    }
}
