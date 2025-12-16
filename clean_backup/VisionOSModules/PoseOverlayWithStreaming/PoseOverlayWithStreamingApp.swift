import SwiftUI

@main
struct PoseOverlayWithStreamingApp: App {
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

        ImmersiveSpace(id: "PoseSpace") { ImmersiveSpaceView() }
            .environmentObject(appModel)
            .environmentObject(settings)
            .environmentObject(logs)
            .environmentObject(sensorModel)
    }
}
