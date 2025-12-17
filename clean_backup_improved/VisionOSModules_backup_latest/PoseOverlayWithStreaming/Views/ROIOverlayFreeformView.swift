import SwiftUI

struct ROIOverlayFreeformView: View {
    @Binding var color: Color   // bind to settings.roiColor (defaults to .cyan)

    @State private var points: [CGPoint] = []
    @State private var isClosed = false
    @State private var isDrawing = false

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // IMPORTANT: a barely-visible background makes the view hit-testable across windows
                Rectangle().fill(Color.black.opacity(0.001))

                // Optional fill so you can see it when closed
                if points.count > 2, isClosed {
                    FreeformShape(points: points, close: true)
                        .fill(color.opacity(0.12))
                        .allowsHitTesting(false)
                }

                // Stroke (force 100% opacity)
                FreeformShape(points: points, close: isClosed)
                    .stroke(color.opacity(1.0), lineWidth: isDrawing ? 3 : 2)
                    .allowsHitTesting(false)
            }
            .contentShape(Rectangle())
            .gesture(drawGesture(in: geo.size))            // LOCAL coordinate space only
        }
    }

    private func drawGesture(in size: CGSize) -> some Gesture {
        DragGesture(minimumDistance: 0, coordinateSpace: .local)
            .onChanged { value in
                isDrawing = true
                if isClosed { points.removeAll(); isClosed = false }

                // throttle a bit to avoid huge paths
                if let last = points.last {
                    if hypot(value.location.x - last.x, value.location.y - last.y) > 1.5 {
                        points.append(clamp(value.location, in: size))
                    }
                } else {
                    points.append(clamp(value.location, in: size))
                }
            }
            .onEnded { _ in
                isDrawing = false
                guard points.count > 2 else { points.removeAll(); return }
                // auto-close if end near start
                if let first = points.first, let last = points.last,
                   hypot(last.x - first.x, last.y - first.y) < 16 {
                    isClosed = true
                }
            }
    }

    private func clamp(_ p: CGPoint, in size: CGSize) -> CGPoint {
        CGPoint(x: min(max(0, p.x), size.width),
                y: min(max(0, p.y), size.height))
    }
}

struct FreeformShape: Shape {
    let points: [CGPoint]
    let close: Bool

    func path(in rect: CGRect) -> Path {
        var path = Path()
        guard let first = points.first else { return path }
        path.move(to: first)
        for pt in points.dropFirst() { path.addLine(to: pt) }
        if close { path.closeSubpath() }
        return path
    }
}
