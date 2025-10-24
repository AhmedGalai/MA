import SwiftUI

struct PoseView: View {
    @ObservedObject var vm: PoseVM

    var body: some View {
        VStack(spacing: 12) {
            HStack(spacing: 16) {
                Text("Rate (Hz):")
                TextField("Hz", text: Binding(
                    get: { String(format: "%.2f", vm.rateHz) },
                    set: { vm.rateHz = Double($0) ?? 1.0 }
                ))
                .frame(width: 80)
                .textFieldStyle(.roundedBorder)

                Button("Start Estimation", action: vm.startEstimation)
                Button("Stop Estimation", action: vm.stopEstimation)
            }

            HStack(alignment: .top, spacing: 16) {
                viewBox(title: "Live", image: vm.preview)
                viewBox(title: "Depth", image: vm.depthImage)
                viewBox(title: "Overlay", image: vm.overlayImage)
                viewBox(title: "Masked", image: vm.maskedImage)
            }

            ScrollView {
                Text(vm.log)
                    .font(.system(size: 12, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 160)
        }
        .padding()
    }

    @ViewBuilder
    func viewBox(title: String, image: CGImage?) -> some View {
        VStack {
            Text(title).font(.headline)
            if let image {
                Image(decorative: image, scale: 1, orientation: .up)
                    .resizable().scaledToFit()
                    .frame(width: 320, height: 240).border(.gray)
            } else {
                ZStack { Color.black; Text("No Image").foregroundColor(.white) }
                    .frame(width: 320, height: 240).border(.gray)
            }
        }
    }
}

