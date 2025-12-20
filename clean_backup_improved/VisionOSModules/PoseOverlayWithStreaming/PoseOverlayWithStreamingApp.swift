import SwiftUI

@main
struct PoseOverlayWithStreamingApp: App {
    @StateObject private var appModel = AppModel()
    @StateObject private var sensorModel = SensorDataModel()
    @StateObject private var calibrationManager = CalibrationManager()
    @StateObject private var arucoStream: ArucoStreamModel

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

        ImmersiveSpace(id: "PoseSpace") { ImmersiveSpaceView() }
            .environmentObject(appModel)
            .environmentObject(sensorModel)
            .environmentObject(arucoStream)
            .environmentObject(calibrationManager)
    }
}
