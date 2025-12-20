import SwiftUI

struct ROIWindowView: View {
    @EnvironmentObject private var settings: ArrowSettings
    @EnvironmentObject private var logs: LogStore

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Place the circular ROI over the object")
                .font(.footnote)
                .foregroundStyle(.secondary)

            ROIOverlayView(roiRadius: $settings.roiRadius, color: $settings.roiColor)
                .frame(minHeight: 240)
                .clipShape(RoundedRectangle(cornerRadius: 12))
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(.quaternary))
        }
        .padding()
    }
}

struct ROIOverlayView: View {
    @Binding var roiRadius: CGFloat
    @Binding var color: Color

    var body: some View {
        GeometryReader { geo in
            let center = CGPoint(x: geo.size.width * 0.5, y: geo.size.height * 0.5)

            ZStack {
                Color.clear // drawing surface

                disk(center: center, radius: roiRadius, color: color)
                    .allowsHitTesting(false)
            }
            .contentShape(Rectangle())
        }
    }

    // Filled disk + optional glow
    @ViewBuilder
    private func disk(center: CGPoint, radius: CGFloat, color: Color) -> some View {
        ZStack {
            // Fill
            Circle()
                .fill(color.opacity(0.30))

            // Crisp edge
            Circle()
                .stroke(color.opacity(0.95), lineWidth: 2)

            // Glow (optional; delete if you want flat)
            Circle()
                .stroke(color.opacity(0.45), lineWidth: 2)
                .blur(radius: 10)
            Circle()
                .stroke(color.opacity(0.25), lineWidth: 2)
                .blur(radius: 20)
        }
        .frame(width: radius * 2, height: radius * 2)
        .position(center)
    }
}
