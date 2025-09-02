import SwiftUI

struct ContentView: View {
    var body: some View {
        TabView {
            IntrinsicsView()
                .tabItem {
                    Label("Intrinsics + ROI", systemImage: "camera")
                }
            PoseView()
                .tabItem {
                    Label("Pose Estimation", systemImage: "cube")
                }
        }
    }
}

