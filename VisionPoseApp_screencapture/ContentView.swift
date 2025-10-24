import SwiftUI

struct ContentView: View {
    @StateObject var vm = PoseVM()
    @StateObject var cap = ScreenCapture.shared

    var body: some View {
        TabView {
            IntrinsicsAndROIView(vm: vm)
                .tabItem { Label("Intrinsics + ROI", systemImage: "camera.viewfinder") }
            PoseView(vm: vm)
                .tabItem { Label("Pose Estimation", systemImage: "cube") }
        }
        .onReceive(cap.$latestImage) { img in vm.bindCapture(img) }
        .onAppear {
            vm.startCapture()
            Task { await vm.fetchModels() }
        }
        .onDisappear {
            vm.stopCapture()
        }
    }
}

