import SwiftUI

@main
struct PoseOverlayWithStreamingApp: App {
    @StateObject private var appModel = AppModel()
    @StateObject private var sensorModel = SensorDataModel()
    @StateObject private var calibrationManager: CalibrationManager
    @StateObject private var arucoStream: ArucoStreamModel
    @StateObject private var rsPoseModel = RealSensePoseModel()
    @StateObject private var logStore = LogStore()
    @StateObject private var arrowSettings = ArrowSettings()

    init() {
        let calibrationManager = CalibrationManager()
        _calibrationManager = StateObject(wrappedValue: calibrationManager)
        _arucoStream = StateObject(wrappedValue: ArucoStreamModel(calibrationManager: calibrationManager))
    }

    var body: some Scene {
        WindowGroup("Aruco Monitor") { ContentView() }
            .environmentObject(appModel)
            .environmentObject(sensorModel)
            .environmentObject(arucoStream)
            .environmentObject(calibrationManager)
            .environmentObject(rsPoseModel)
            .environmentObject(logStore)
            .environmentObject(arrowSettings)

        WindowGroup("Anchor Setup", id: "anchor") { AnchorSetupView() }
            .environmentObject(appModel)

        WindowGroup("ROI Window", id: "roi") { ROIWindowView() }
            .windowStyle(.plain)
            .environmentObject(arrowSettings)
            .environmentObject(logStore)

        WindowGroup("Debug Viewer", id: "debug") { DebugDashboardView() }
            .environmentObject(appModel)
            .environmentObject(sensorModel)
            .environmentObject(arucoStream)
            .environmentObject(logStore)

        ImmersiveSpace(id: "PoseSpace") { ImmersiveSpaceView() }
            .environmentObject(appModel)
            .environmentObject(sensorModel)
            .environmentObject(arucoStream)
            .environmentObject(calibrationManager)
            .environmentObject(rsPoseModel)
            .environmentObject(logStore)
    }
}
