import SwiftUI

@main
struct FinalAppApp: App {
    @StateObject private var appModel = AppModel()
    @StateObject private var sensorModel = SensorDataModel()
    @StateObject private var calibrationManager: CalibrationManager
    @StateObject private var arucoStream: ArucoStreamModel
    @StateObject private var rsPoseModel = RealSensePoseModel()
    @StateObject private var foundationPoseModel = FoundationPoseModel()
    @StateObject private var logStore = LogStore()
    @StateObject private var arrowSettings = ArrowSettings()
    @StateObject private var calibrationModel = CalibrationModel()

    init() {
        let calibrationManager = CalibrationManager()
        _calibrationManager = StateObject(wrappedValue: calibrationManager)
        _arucoStream = StateObject(wrappedValue: ArucoStreamModel(calibrationManager: calibrationManager))
        CameraTransformUtils.setCalibrationManager(calibrationManager)
    }

    var body: some Scene {
        WindowGroup("Aruco Monitor") { ContentView() }
            .environmentObject(appModel)
            .environmentObject(sensorModel)
            .environmentObject(arucoStream)
            .environmentObject(calibrationManager)
            .environmentObject(rsPoseModel)
            .environmentObject(foundationPoseModel)
            .environmentObject(logStore)
            .environmentObject(arrowSettings)
            .environmentObject(calibrationModel)

        WindowGroup("ROI Window", id: "roi") { ROIWindowView() }
            .windowStyle(.plain)
            .environmentObject(appModel)
            .environmentObject(arrowSettings)
            .environmentObject(logStore)

        WindowGroup("Debug Viewer", id: "debug") { DebugDashboardView() }
            .environmentObject(appModel)
            .environmentObject(sensorModel)
            .environmentObject(arucoStream)
            .environmentObject(foundationPoseModel)
            .environmentObject(logStore)

        WindowGroup("Help", id: "help") { HelpView() }
            .environmentObject(appModel)
            .environmentObject(logStore)

        WindowGroup("Anchor Setup", id: "anchor") { AnchorSetupView() }
            .environmentObject(calibrationManager)
            .environmentObject(logStore)
            .environmentObject(appModel)
            .environmentObject(calibrationModel)

        ImmersiveSpace(id: "PoseSpace") { ImmersiveSpaceView() }
            .environmentObject(appModel)
            .environmentObject(sensorModel)
            .environmentObject(arucoStream)
            .environmentObject(calibrationManager)
            .environmentObject(rsPoseModel)
            .environmentObject(foundationPoseModel)
            .environmentObject(logStore)
            .environmentObject(calibrationModel)
    }
}
