import SwiftUI

final class ArrowSettings: ObservableObject {
    @Published var color: Color = .red
    @Published var roiColor: Color = .cyan          // fully opaque default
    @Published var roiRadius: CGFloat = 120
}

import SwiftUI

final class LogStore: ObservableObject {
    @Published var lines: [String] = []

    func add(_ s: String) {
        let stamp = ISO8601DateFormatter().string(from: Date())
        DispatchQueue.main.async { self.lines.append("[\(stamp)] \(s)") }
    }

    func clear() {
        DispatchQueue.main.async { self.lines.removeAll() }
    }
}


@main
struct PoseOverlayApp: App {
    @StateObject private var settings = ArrowSettings()
    @StateObject private var logs = LogStore()
    @StateObject private var sensorModel = SensorDataModel()
    @StateObject private var appModel = AppModel()

    var body: some Scene {
        WindowGroup("Pose Overlay") { ContentView() }
            .environmentObject(appModel)
            .environmentObject(settings)
            .environmentObject(logs)
            .environmentObject(sensorModel)

        WindowGroup("ROI", id: "roi") {
            ROIWindowView()
        }
        .windowStyle(.plain)
        .environmentObject(appModel)
        .environmentObject(settings)
        .environmentObject(logs)

        WindowGroup("Logs", id: "logs") { LogsView() }
            .environmentObject(logs)

        WindowGroup("Sensor Monitor", id: "sensors") {
            SensorMonitorView()
        }
        .environmentObject(appModel)
        .environmentObject(sensorModel)

        WindowGroup("Debug Dashboard", id: "debug") {
            DebugDashboardView()
        }
        .environmentObject(appModel)
        .environmentObject(sensorModel)
        .environmentObject(logs)

        ImmersiveSpace(id: "PoseSpace") { ImmersiveSpaceView() }
            .environmentObject(appModel)
            .environmentObject(settings)
            .environmentObject(logs)
            .environmentObject(sensorModel)
    }
}
