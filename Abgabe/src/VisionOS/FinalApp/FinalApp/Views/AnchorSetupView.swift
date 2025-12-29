import SwiftUI

struct AnchorSetupView: View {
    @EnvironmentObject private var calibrationModel: CalibrationModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Gizmo Calibration")
                .font(.largeTitle)
                .bold()

            Text("Use the sliders to align the virtual gizmo with a physical reference object (e.g., the ArUco board).")
                .foregroundStyle(.secondary)

            translationSliders
            rotationSliders
            trackingSliders

            Spacer()
        }
        .padding(32)
    }

    private var translationSliders: some View {
        GroupBox("Translation (meters)") {
            VStack {
                slider(label: "X (left/right)", value: $calibrationModel.xTranslation, range: -0.5...0.5)
                slider(label: "Y (up/down)", value: $calibrationModel.yTranslation, range: -0.5...0.5)
                slider(label: "Z (forward/backward)", value: $calibrationModel.zTranslation, range: -0.5...0.5)
            }
            .padding(.top, 8)
        }
    }

    private var rotationSliders: some View {
        GroupBox("Rotation (degrees)") {
            VStack {
                slider(label: "X-axis (Pitch)", value: $calibrationModel.xRotation, range: -180...180)
                slider(label: "Y-axis (Yaw)", value: $calibrationModel.yRotation, range: -180...180)
                slider(label: "Z-axis (Roll)", value: $calibrationModel.zRotation, range: -180...180)
            }
            .padding(.top, 8)
        }
    }

    private var trackingSliders: some View {
        GroupBox("Tracking") {
            VStack {
                slider(label: "ArUco smoothing", value: $calibrationModel.arucoSmoothingAlpha, range: 0.0...0.5)
            }
            .padding(.top, 8)
        }
    }

    private func slider(label: String, value: Binding<Float>, range: ClosedRange<Float>) -> some View {
        VStack(alignment: .leading) {
            Text("\(label): \(value.wrappedValue, specifier: "%.3f")")
            Slider(value: value, in: range, step: 0.001)
                .onChange(of: value.wrappedValue) { _, _ in
                    saveCalibration()
                }
        }
    }
    
    private func saveCalibration() {
        // The actual calculation will be done in the ImmersiveSpaceView
        // when a notification is received.
        NotificationCenter.default.post(name: .saveCalibration, object: nil)
    }
}

extension Notification.Name {
    static let saveCalibration = Notification.Name("saveCalibration")
}
