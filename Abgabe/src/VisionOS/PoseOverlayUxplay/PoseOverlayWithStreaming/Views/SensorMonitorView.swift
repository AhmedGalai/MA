import SwiftUI
import CoreMotion

struct SensorMonitorView: View {
    @EnvironmentObject private var sensorModel: SensorDataModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("Sensor Stream")
                    .font(.title)
                    .bold()

                headPoseSection
                motionSection
            }
            .padding(24)
        }
        .task {
            await MainActor.run {
                sensorModel.start()
            }
        }
    }

    private var headPoseSection: some View {
        DataSection(title: "Head Pose") {
            DataRow(title: "Position (m)", value: vectorString(sensorModel.headPosition))
            DataRow(title: "Quaternion", value: quaternionString(sensorModel.headOrientation))
            DataRow(title: "Euler (°)", value: vectorString(sensorModel.headEulerDegrees))
        }
    }

    private var motionSection: some View {
        DataSection(title: "Device Motion") {
            DataRow(title: "User Accel (g)", value: accelerationString(sensorModel.userAcceleration))
            DataRow(title: "Rotation Rate (rad/s)", value: rotationRateString(sensorModel.rotationRate))
            DataRow(title: "Gravity (g)", value: accelerationString(sensorModel.gravity))
            if let attitude = sensorModel.attitude {
                DataRow(
                    title: "Attitude (°)",
                    value: String(format: "%.2f  %.2f  %.2f",
                                  attitude.roll * 180 / .pi,
                                  attitude.pitch * 180 / .pi,
                                  attitude.yaw * 180 / .pi)
                )
            }
        }
    }

    private func vectorString(_ v: SIMD3<Double>) -> String {
        String(format: "%.3f  %.3f  %.3f", v.x, v.y, v.z)
    }

    private func quaternionString(_ q: simd_quatd) -> String {
        String(format: "%.3f  %.3f  %.3f  %.3f", q.imag.x, q.imag.y, q.imag.z, q.real)
    }

    private func accelerationString(_ a: CMAcceleration) -> String {
        String(format: "%.3f  %.3f  %.3f", a.x, a.y, a.z)
    }

    private func rotationRateString(_ r: CMRotationRate) -> String {
        String(format: "%.3f  %.3f  %.3f", r.x, r.y, r.z)
    }
}

private struct DataSection<Content: View>: View {
    let title: String
    private let content: () -> Content

    init(title: String, @ViewBuilder content: @escaping () -> Content) {
        self.title = title
        self.content = content
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.headline)
            VStack(alignment: .leading, spacing: 8) {
                content()
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }
}

private struct DataRow: View {
    let title: String
    let value: String

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            Text(value)
                .font(.system(.body, design: .monospaced))
        }
    }
}
