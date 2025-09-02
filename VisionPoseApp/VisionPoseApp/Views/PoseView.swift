import SwiftUI

struct PoseView: View {
    @State private var logs: [String] = []
    @State private var isRunning = false
    @State private var selectedModel: String?

    var body: some View {
        VStack {
            HStack {
                Button("Start") {
                    isRunning = true
                    logs.append("Started pose estimation")
                    APIClient.requestPose(modelFile: selectedModel ?? "default.ply") { resp in
                        DispatchQueue.main.async {
                            logs.append("Response: \(resp)")
                        }
                    }
                }
                Button("Stop") {
                    isRunning = false
                    logs.append("Stopped pose estimation")
                }
            }

            ScrollView {
                VStack(alignment: .leading) {
                    ForEach(logs, id: \.self) { log in
                        Text(log).foregroundColor(.green)
                            .font(.system(size: 12, design: .monospaced))
                    }
                }
            }
            .frame(maxHeight: 200)
            .background(Color.black)
        }
        .padding()
    }
}

