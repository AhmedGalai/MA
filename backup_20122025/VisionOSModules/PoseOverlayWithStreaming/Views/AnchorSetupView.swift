import SwiftUI

struct AnchorSetupView: View {
    @EnvironmentObject private var appModel: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Anchor Setup")
                    .font(.title2)
                    .bold()

                anchorPositionSection
                anchorRotationSection

                Button("Reset Anchor") {
                    appModel.anchorTranslation = .zero
                    appModel.anchorEulerDegrees = .zero
                }
                .buttonStyle(.bordered)
            }
            .padding()
        }
    }

    private var anchorPositionSection: some View {
        GroupBox("Position (m)") {
            VStack(alignment: .leading, spacing: 8) {
                axisSlider(title: "X", value: Binding(
                    get: { appModel.anchorTranslation.x },
                    set: { appModel.anchorTranslation.x = $0 }
                ), range: -2.0...2.0, step: 0.01)
                axisSlider(title: "Y", value: Binding(
                    get: { appModel.anchorTranslation.y },
                    set: { appModel.anchorTranslation.y = $0 }
                ), range: -2.0...2.0, step: 0.01)
                axisSlider(title: "Z", value: Binding(
                    get: { appModel.anchorTranslation.z },
                    set: { appModel.anchorTranslation.z = $0 }
                ), range: -2.0...2.0, step: 0.01)
            }
        }
    }

    private var anchorRotationSection: some View {
        GroupBox("Rotation (deg)") {
            VStack(alignment: .leading, spacing: 8) {
                axisSlider(title: "Pitch", value: Binding(
                    get: { appModel.anchorEulerDegrees.x },
                    set: { appModel.anchorEulerDegrees.x = $0 }
                ), range: -180.0...180.0, step: 1.0)
                axisSlider(title: "Yaw", value: Binding(
                    get: { appModel.anchorEulerDegrees.y },
                    set: { appModel.anchorEulerDegrees.y = $0 }
                ), range: -180.0...180.0, step: 1.0)
                axisSlider(title: "Roll", value: Binding(
                    get: { appModel.anchorEulerDegrees.z },
                    set: { appModel.anchorEulerDegrees.z = $0 }
                ), range: -180.0...180.0, step: 1.0)
            }
        }
    }

    private func axisSlider(title: String,
                            value: Binding<Double>,
                            range: ClosedRange<Double>,
                            step: Double) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("\(title): \(String(format: "%.2f", value.wrappedValue))")
                .font(.caption)
                .foregroundStyle(.secondary)
            Slider(value: value, in: range, step: step)
        }
    }
}

#Preview(windowStyle: .automatic) {
    AnchorSetupView()
        .environmentObject(AppModel())
}
