import SwiftUI

@MainActor
final class LogStore: ObservableObject {
    @Published var lines: [String] = []

    func add(_ s: String) {
        let stamp = ISO8601DateFormatter().string(from: Date())
        DispatchQueue.main.async { self.lines.append("[\(stamp)] \(s)") }
    }

    func clear() {
        DispatchQueue.main.async { self.lines.removeAll() }
    }
}
