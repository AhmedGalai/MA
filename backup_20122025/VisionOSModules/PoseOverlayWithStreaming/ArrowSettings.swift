import SwiftUI

@MainActor
final class ArrowSettings: ObservableObject {
    @Published var color: Color = .red
    @Published var roiColor: Color = .cyan          // fully opaque default
    @Published var roiRadius: CGFloat = 120
}
