import SwiftUI

struct AnchorSetupView: View {
    @EnvironmentObject private var arucoStream: ArucoStreamModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Camera Offset Calibration")
                    .font(.title2)
                    .bold()

                Text("Fine-tune the estimated camera offset from the device anchor. Adjust these values until the 3D ArUco pose aligns with the physical marker.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                cameraOffsetSection

                Button("Reset to Default") {
                    CameraTransformUtils.estimatedCameraOffset = SIMD3<Float>(0.0, -0.01, 0.04)
                }
                .buttonStyle(.bordered)

                Divider()

                trackingStatusSection
            }
            .padding()
        }
    }

    private var cameraOffsetSection: some View {
        GroupBox("Camera Offset (meters)") {
            VStack(alignment: .leading, spacing: 8) {
                offsetSlider(title: "X (right)", value: Binding(
                    get: { CameraTransformUtils.estimatedCameraOffset.x },
                    set: { CameraTransformUtils.estimatedCameraOffset.x = $0 }
                ), range: -0.1...0.1, step: 0.001)

                offsetSlider(title: "Y (down)", value: Binding(
                    get: { CameraTransformUtils.estimatedCameraOffset.y },
                    set: { CameraTransformUtils.estimatedCameraOffset.y = $0 }
                ), range: -0.1...0.1, step: 0.001)

                offsetSlider(title: "Z (forward)", value: Binding(
                    get: { CameraTransformUtils.estimatedCameraOffset.z },
                    set: { CameraTransformUtils.estimatedCameraOffset.z = $0 }
                ), range: 0.0...0.15, step: 0.001)
            }
        }
    }

    private var trackingStatusSection: some View {
        GroupBox("Tracking Status") {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("ArUco Tracking:")
                    Spacer()
                    Text(arucoStream.isTracking ? "Active" : "Inactive")
                        .foregroundStyle(arucoStream.isTracking ? .green : .secondary)
                }

                if let deviceTransform = arucoStream.deviceToArucoTransform {
                    HStack {
                        Text("Continuous Tracking:")
                        Spacer()
                        Text("Enabled")
                            .foregroundStyle(.blue)
                    }
                    Text("Pose will persist even when marker is not visible")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    HStack {
                        Text("Continuous Tracking:")
                        Spacer()
                        Text("Waiting for detection")
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private func offsetSlider(title: String,
                             value: Binding<Float>,
                             range: ClosedRange<Float>,
                             step: Float) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("\(title): \(String(format: "%.3f", value.wrappedValue))m")
                .font(.caption)
                .foregroundStyle(.secondary)
            Slider(value: value, in: range, step: step)
        }
    }
}
