import SwiftUI

struct ROIOverlay: View {
    @Binding var center: CGPoint?
    @Binding var radius: CGFloat?
    var background: UIImage?

    var body: some View {
        ZStack {
            if let bg = background {
                Image(uiImage: bg)
                    .resizable()
                    .scaledToFill()
            } else {
                Color.black
            }
            if let c = center, let r = radius {
                Circle()
                    .stroke(Color.green, lineWidth: 2)
                    .frame(width: r*2, height: r*2)
                    .position(c)
            }
        }
        .gesture(
            DragGesture(minimumDistance: 0)
                .onChanged { value in
                    if center == nil {
                        center = value.location
                    } else {
                        let dx = value.location.x - center!.x
                        let dy = value.location.y - center!.y
                        radius = sqrt(dx*dx + dy*dy)
                    }
                }
        )
    }
}

