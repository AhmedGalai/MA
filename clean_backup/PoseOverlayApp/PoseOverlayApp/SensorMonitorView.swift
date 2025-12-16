import SwiftUI
import CoreMotion

struct SensorMonitorView: View {
    @EnvironmentObject private var sensorModel: SensorDataModel
    @EnvironmentObject private var appModel: AppModel
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    @State private var showOrientationGraph = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("Vision Pro Sensor Monitor")
                    .font(.title)
                    .bold()

                debugSection
                apiSyncSection
                immersiveButton
                statusSection
                headPoseSection
                motionSection

                Toggle("Show orientation visualization", isOn: $showOrientationGraph)

                if showOrientationGraph {
                    OrientationVisualizer(quaternion: sensorModel.headOrientation)
                        .frame(height: 180)
                        .transition(.opacity)
                }

                Spacer(minLength: 24)
            }
            .padding(24)
        }
        .task {
            await MainActor.run {
                sensorModel.start()
            }
        }
    }

    private var debugSection: some View {
        GroupBox("Debug Info") {
            VStack(alignment: .leading, spacing: 4) {
                Text("Start called: \(startStatusText)")
                    .foregroundColor(hasStarted ? .green : .orange)

                Text("Last motion update: \(lastMotionText)")
                    .foregroundColor(sensorModel.lastMotionUpdate == .distantPast ? .red : .green)

                Text("Frame count: \(sensorModel.frameCount)")
                    .font(.system(.body, design: .monospaced))
                    .bold()
            }
            .font(.caption)
        }
    }

    private var apiSyncSection: some View {
        GroupBox("API Sync") {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Base URL")
                    Spacer()
                    Text(appModel.baseURL?.absoluteString ?? "Not set")
                        .font(.footnote)
                        .foregroundStyle(appModel.baseURL == nil ? .red : .secondary)
                }
                HStack {
                    Text("Uploads sent")
                    Spacer()
                    Text("\(sensorModel.headPoseUploadCount)")
                        .font(.system(.body, design: .monospaced))
                }
                HStack {
                    Text("Last upload")
                    Spacer()
                    Text(lastUploadText)
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(sensorModel.lastHeadPoseUpload == .distantPast ? .secondary : .primary)
                }
                if let error = sensorModel.headPoseUploadError {
                    Text("Last error: \(error)")
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
            }
        }
    }

    private var immersiveButton: some View {
        Button(appModel.immersiveSpacePresented ? "Hide 3D View" : "Show 3D View (enables ARKit)") {
            Task { await toggleImmersiveSpace() }
        }
        .buttonStyle(.borderedProminent)
        .tint(appModel.immersiveSpacePresented ? .red : .green)
    }

    private var statusSection: some View {
        GroupBox("Status") {
            HStack(alignment: .top) {
                Circle()
                    .fill(sensorModel.statusMessage == "Tracking head pose" ? Color.green : Color.yellow)
                    .frame(width: 12, height: 12)
                VStack(alignment: .leading, spacing: 4) {
                    Text(sensorModel.statusMessage)
                    if sensorModel.lastPoseUpdate != .distantPast {
                        Text("Pose updated \(sensorModel.lastPoseUpdate.formatted(.dateTime.hour().minute().second()))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    if sensorModel.lastMotionUpdate != .distantPast {
                        Text("Motion updated \(sensorModel.lastMotionUpdate.formatted(.dateTime.hour().minute().second()))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private var headPoseSection: some View {
        GroupBox("Head Pose") {
            VStack(alignment: .leading, spacing: 8) {
                valueRow(title: "Position (m)", value: vectorString(sensorModel.headPosition))
                valueRow(title: "Orientation (quat)", value: quaternionString(sensorModel.headOrientation))
                valueRow(title: "Euler (°)", value: vectorString(sensorModel.headEulerDegrees))
            }
        }
    }

    private var motionSection: some View {
        GroupBox("Device Motion") {
            VStack(alignment: .leading, spacing: 8) {
                valueRow(title: "User Accel (g)", value: accelerationString(sensorModel.userAcceleration))
                valueRow(title: "Rotation Rate (rad/s)", value: rotationRateString(sensorModel.rotationRate))
                valueRow(title: "Gravity (g)", value: accelerationString(sensorModel.gravity))
                if let attitude = sensorModel.attitude {
                    valueRow(
                        title: "Attitude (°)",
                        value: String(format: "%.2f  %.2f  %.2f",
                                      attitude.roll * 180 / .pi,
                                      attitude.pitch * 180 / .pi,
                                      attitude.yaw * 180 / .pi)
                    )
                }
            }
        }
    }

    private func valueRow(title: String, value: String) -> some View {
        HStack {
            Text(title)
            Spacer()
            Text(value)
                .font(.system(.body, design: .monospaced))
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

    private func toggleImmersiveSpace() async {
        if appModel.immersiveSpacePresented {
            await dismissImmersiveSpace()
            await MainActor.run {
                appModel.setImmersiveSpacePresented(false)
            }
            return
        }
        do {
            let result = try await openImmersiveSpace(id: "PoseSpace")
            if case .opened = result {
                await MainActor.run {
                    appModel.setImmersiveSpacePresented(true)
                }
            }
        } catch {
            NSLog("📱 [SensorMonitor] Immersive error: %@", error.localizedDescription)
        }
    }

    private var hasStarted: Bool {
        sensorModel.statusMessage != "Starting…"
    }

    private var startStatusText: String {
        hasStarted ? "YES" : "NO"
    }

    private var lastMotionText: String {
        sensorModel.lastMotionUpdate == .distantPast
        ? "NEVER"
        : sensorModel.lastMotionUpdate.formatted(.dateTime.hour().minute().second())
    }

    private var lastUploadText: String {
        sensorModel.lastHeadPoseUpload == .distantPast
        ? "Not yet"
        : sensorModel.lastHeadPoseUpload.formatted(.dateTime.hour().minute().second())
    }
}

struct OrientationVisualizer: View {
    let quaternion: simd_quatd

    private var euler: SIMD3<Double> {
        SensorDataModel.eulerDegrees(from: quaternion)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Orientation Axes (°)")
                .font(.caption)
                .foregroundStyle(.secondary)
            let axes: [(label: String, value: Double, color: Color)] = [
                ("Roll", euler.x, .red),
                ("Pitch", euler.y, .green),
                ("Yaw", euler.z, .blue)
            ]
            ForEach(axes, id: \.label) { item in
                AxisBar(label: item.label, value: item.value, color: item.color)
            }
        }
    }
}

private struct AxisBar: View {
    let label: String
    let value: Double
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                Spacer()
                Text(String(format: "%.1f°", value))
                    .font(.system(.footnote, design: .monospaced))
            }
            GeometryReader { proxy in
                let width = proxy.size.width
                let normalized = min(max(value / 180.0, -1.0), 1.0)
                let barWidth = width * abs(normalized)
                ZStack(alignment: normalized >= 0 ? .leading : .trailing) {
                    Capsule()
                        .fill(Color.gray.opacity(0.2))
                    Capsule()
                        .fill(color.opacity(0.7))
                        .frame(width: barWidth)
                }
                .frame(height: 12)
            }
            .frame(height: 16)
        }
    }
}

#Preview(windowStyle: .automatic) {
    SensorMonitorView()
        .environmentObject(SensorDataModel())
        .environmentObject(AppModel())
}
