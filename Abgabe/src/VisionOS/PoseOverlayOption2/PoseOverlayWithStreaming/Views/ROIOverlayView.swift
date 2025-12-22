import SwiftUI

//struct ROIWindowView: View {
//    @EnvironmentObject private var settings: ArrowSettings
//    @EnvironmentObject private var logs: LogStore
//
//    var body: some View {
//        VStack(alignment: .leading, spacing: 10) {
//            Text("Place the window ROI over the object")
//                .font(.footnote)
//                .foregroundStyle(.secondary)
//
//            ROIOverlayView(roiRadius: $settings.roiRadius, color: $settings.roiColor)
//                .frame(minHeight: 240)
//                .clipShape(RoundedRectangle(cornerRadius: 12))
//                .overlay(RoundedRectangle(cornerRadius: 12).stroke(.quaternary))
//        }
//        .padding()
//    }
//}

//struct ROIWindowView: View {
//    @EnvironmentObject private var settings: ArrowSettings
//    @EnvironmentObject private var logs: LogStore
//
//    var body: some View {
//        VStack(alignment: .leading, spacing: 10) {
//            Text("Draw a freeform ROI around the object")
//                .font(.footnote)
//                .foregroundStyle(.secondary)
//
//            ROIOverlayFreeformView(color: $settings.roiColor)
//                .frame(minHeight: 240)
//                .clipShape(RoundedRectangle(cornerRadius: 12))
//                .overlay(RoundedRectangle(cornerRadius: 12).stroke(.quaternary))
//
//        }
//        .padding()
//    }
//}


struct ROIWindowView: View {
    @EnvironmentObject private var settings: ArrowSettings

    var body: some View {
        ZStack {
            Color.clear
            ROIOverlayView(roiRadius: $settings.roiRadius, color: $settings.roiColor)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .ignoresSafeArea()
        .background(Color.clear)
    }
}


//
//struct ROIOverlayView: View {
//    @Binding var roiRadius: CGFloat
//    @Binding var color: Color
//
//    @State private var hoverEdge = false  // just to fatten the ring when hovered
//
//    var body: some View {
//        GeometryReader { geo in
//            let center = CGPoint(x: geo.size.width * 0.5, y: geo.size.height * 0.5)
//
//            ZStack {
//                // Transparent background for the drawing surface
//                Color.clear
//
//                // Glowing ring (no fill), centered
//                ring(center: center, radius: roiRadius, color: color, hot: hoverEdge)
//                    .allowsHitTesting(false)
//
//                // Edge handle (visual only)
//                let edgePos = CGPoint(x: center.x + roiRadius, y: center.y)
//                Circle()
//                    .fill(hoverEdge ? Color.accentColor : .white)
//                    .overlay(Circle().stroke(.black.opacity(0.25), lineWidth: 1))
//                    .frame(width: 18, height: 18)
//                    .shadow(color: hoverEdge ? .accentColor.opacity(0.6) : .clear, radius: 6)
//                    .position(edgePos)
//                    .onHover { hover in hoverEdge = hover }
//                    .allowsHitTesting(false) // no gestures; size is controlled by the slider
//            }
//            .contentShape(Rectangle()) // no drag/resize here
//        }
//    }
//
//    // Visuals
//    @ViewBuilder
//    private func ring(center: CGPoint, radius: CGFloat, color: Color, hot: Bool) -> some View {
//        let w: CGFloat = hot ? 3 : 2
//        ZStack {
//            Circle().stroke(color.opacity(0.95), lineWidth: w)
//            Circle().stroke(color.opacity(0.55), lineWidth: w).blur(radius: 8)
//            Circle().stroke(color.opacity(0.35), lineWidth: w).blur(radius: 16)
//        }
//        .frame(width: radius * 2, height: radius * 2)
//        .position(center)
//    }
//}
//
//
//private extension CGFloat {
//    func clamped(min: CGFloat, max: CGFloat) -> CGFloat { Swift.min(Swift.max(self, min), max) }
//}


struct ROIOverlayView: View {
    @Binding var roiRadius: CGFloat
    @Binding var color: Color

    @EnvironmentObject private var appModel: AppModel
    @State private var lastSent = AvpRoiConfigSnapshot.empty

    var body: some View {
        GeometryReader { geo in
            let center = CGPoint(x: geo.size.width * 0.5, y: geo.size.height * 0.5)
            let size = geo.size
            let globalFrame = geo.frame(in: .global)

            ZStack {
                Color.clear // drawing surface
                ring(center: center, radius: roiRadius, color: color)
                    .allowsHitTesting(false)
            }
            .contentShape(Rectangle()) // still no gestures here
            .task {
                await sendIfNeeded(size: size, globalFrame: globalFrame)
            }
            .onChange(of: roiRadius) { _, _ in
                Task { await sendIfNeeded(size: size, globalFrame: globalFrame) }
            }
            .onChange(of: size) { _, _ in
                Task { await sendIfNeeded(size: size, globalFrame: globalFrame) }
            }
        }
    }

    // Visuals
    @ViewBuilder
    private func ring(center: CGPoint, radius: CGFloat, color: Color) -> some View {
        let stroke: CGFloat = 2
        ZStack {
            Circle()
                .stroke(color.opacity(0.95), lineWidth: stroke)
            Circle()
                .stroke(color.opacity(0.55), lineWidth: stroke)
                .blur(radius: 8)
            Circle()
                .stroke(color.opacity(0.35), lineWidth: stroke)
                .blur(radius: 10)
        }
        .frame(width: radius * 2, height: radius * 2)
        .position(center)
    }

    private func sendIfNeeded(size: CGSize, globalFrame: CGRect) async {
        guard let baseURL = appModel.baseURL, size.width > 0, size.height > 0 else { return }
        let roi = RoiNormalizationStrategy.normalizedROI(
            size: size,
            globalFrame: globalFrame,
            radius: roiRadius
        )
        guard roi.isValid else { return }
        let snapshot = AvpRoiConfigSnapshot(roi)
        guard snapshot.shouldSend(comparedTo: lastSent) else { return }
        lastSent = snapshot
        do {
            try await AvpRoiService.update(baseURL: baseURL, roi: roi)
        } catch {
            // Silent failure to avoid UI spam; debug from server if needed.
        }
    }
}

private struct AvpRoiConfigSnapshot: Equatable {
    static let empty = AvpRoiConfigSnapshot(
        enabled: false,
        cx: -1,
        cy: -1,
        r: -1
    )

    let enabled: Bool
    let cx: Double
    let cy: Double
    let r: Double

    init(enabled: Bool, cx: Double, cy: Double, r: Double) {
        self.enabled = enabled
        self.cx = cx
        self.cy = cy
        self.r = r
    }

    init(_ roi: AvpRoiConfigPayload) {
        self.enabled = roi.enabled
        self.cx = roi.cx_n
        self.cy = roi.cy_n
        self.r = roi.r_n
    }

    func shouldSend(comparedTo other: AvpRoiConfigSnapshot) -> Bool {
        guard enabled == other.enabled else { return true }
        let eps = 0.002
        return abs(cx - other.cx) > eps || abs(cy - other.cy) > eps || abs(r - other.r) > eps
    }
}
