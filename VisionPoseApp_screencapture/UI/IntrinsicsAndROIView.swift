import SwiftUI

struct IntrinsicsAndROIView: View {
    @ObservedObject var vm: PoseVM
    @State private var roiMode = false

    var body: some View {
        VStack(spacing: 12) {
            // Model picker + ROI toggle
            HStack(spacing: 16) {
                Text("Select Model:")
                Picker("Model", selection: $vm.selectedModel) {
                    ForEach(vm.modelNames, id: \.self) { Text($0) }
                }
                .onChange(of: vm.selectedModel) { _, _ in
                    Task { try? await vm.fetchMeshForSelected() }
                }

                Toggle(isOn: $roiMode) {
                    Text("Select ROI")
                }
                .toggleStyle(.switch)
            }

            // Capture buttons
            HStack(spacing: 16) {
                Button("Capture Left", action: vm.captureLeft)
                Button("Capture Right", action: vm.captureRight)
                Button("Send Intrinsics") { Task { await vm.sendIntrinsics() } }
            }

            // Left/Right thumbs
            HStack(alignment: .top, spacing: 20) {
                imageView(vm.leftSnap).frame(width: 320, height: 240).border(.gray)
                imageView(vm.rightSnap).frame(width: 320, height: 240).border(.gray)
            }

            // ROI Canvas on live preview
            ZStack {
                imageView(vm.preview)
                    .frame(width: 640, height: 480)
                    .border(.gray)
                if roiMode {
                    ROIOverlay(center: $vm.roiCenter, radius: $vm.roiRadius)
                        .frame(width: 640, height: 480)
                } else if let c = vm.roiCenter, let r = vm.roiRadius {
                    Circle()
                        .stroke(Color.green, lineWidth: 2)
                        .frame(width: r*2, height: r*2)
                        .position(c)
                }
            }

            // Logs
            ScrollView {
                Text(vm.log)
                    .font(.system(size: 12, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 140)
        }
        .padding()
    }

    @ViewBuilder
    func imageView(_ cg: CGImage?) -> some View {
        if let cg {
            Image(decorative: cg, scale: 1, orientation: .up)
                .resizable()
                .interpolation(.high)
                .scaledToFit()
        } else {
            ZStack { Color.black; Text("No Image").foregroundColor(.white) }
        }
    }
}

