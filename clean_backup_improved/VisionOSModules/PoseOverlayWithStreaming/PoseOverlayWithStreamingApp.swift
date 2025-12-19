import SwiftUI

@main
struct PoseOverlayWithStreamingApp: App {
    @StateObject private var appModel = AppModel()
    @StateObject private var sensorModel = SensorDataModel()
    @StateObject private var arucoStream = ArucoStreamModel()

    var body: some Scene {
        WindowGroup("Aruco Monitor") { ContentView() }
            .environmentObject(appModel)
            .environmentObject(sensorModel)
            .environmentObject(arucoStream)

        ImmersiveSpace(id: "PoseSpace") { ImmersiveSpaceView() }
            .environmentObject(appModel)
            .environmentObject(sensorModel)
            .environmentObject(arucoStream)
    }
}
