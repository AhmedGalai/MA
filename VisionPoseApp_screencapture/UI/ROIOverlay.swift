import SwiftUI

struct ROIOverlay: View {
    @Binding var center: CGPoint?
    @Binding var radius: CGFloat?
    @State private var tempRadius: CGFloat?

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Color.clear
                    .contentShape(Rectangle())
                    .gesture(TapGesture().onEnded {
                        // first tap sets center; second tap locks radius from temp
                        if center == nil {
                            // if gaze focus APIs available, you could seed with focus location instead
                            center = CGPoint(x: geo.size.width/2, y: geo.size.height/2)
                            radius = nil
                            tempRadius = nil
                        } else if let tr = tempRadius {
                            radius = tr
                        }
                    })
                    .gesture(DragGesture(minimumDistance: 0).onChanged { val in
                        guard let c = center else { return }
                        let dx = val.location.x - c.x
                        let dy = val.location.y - c.y
                        tempRadius = sqrt(dx*dx + dy*dy)
                    }.onEnded { _ in
                        if let tr = tempRadius { radius = tr }
                    })

                if let c = center, let r = tempRadius ?? radius {
                    Circle().stroke(Color.red, style: StrokeStyle(lineWidth: 2, dash: [4,4]))
                        .frame(width: r*2, height: r*2)
                        .position(c)
                }
            }
        }
    }
}

