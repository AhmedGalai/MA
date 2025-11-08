import SwiftUI

struct IntrinsicsView: View {
    @State private var models: [String] = []
    @State private var selectedModel: String?
    @State private var intrinsics: [[Double]]?

    var body: some View {
        ScrollView {
            VStack(spacing: 16) {
                Text("Select Model")
                Picker("Model", selection: $selectedModel) {
                    ForEach(models, id: \.self) { model in
                        Text(model).tag(model as String?)
                    }
                }
                .pickerStyle(MenuPickerStyle())
                .onAppear {
                    APIClient.fetchModels { list in
                        DispatchQueue.main.async {
                            self.models = list
                            self.selectedModel = list.first
                        }
                    }
                }

                Button("Send Intrinsics") {
                    APIClient.sendIntrinsics(left: UIImage(systemName: "photo")!,
                                             right: UIImage(systemName: "photo")!) { K in
                        intrinsics = K
                    }
                }

                if let K = intrinsics {
                    Text("Intrinsics: \(K.description)")
                        .font(.system(size: 12, design: .monospaced))
                }
            }
            .padding()
        }
    }
}

