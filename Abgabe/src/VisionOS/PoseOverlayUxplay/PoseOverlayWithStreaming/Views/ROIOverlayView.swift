import SwiftUI

struct ROIWindowView: View {
    @EnvironmentObject private var appModel: AppModel
    @EnvironmentObject private var settings: ArrowSettings
    @StateObject private var feed = UxplayFeedModel()
    @State private var roiCx = 0.5
    @State private var roiCy = 0.5
    @State private var roiRadius = 0.2
    @State private var lastSent = AvpRoiConfigSnapshot.empty

    var body: some View {
        VStack(spacing: 12) {
            ZStack {
                Color.black
                if let image = feed.image {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "airplayvideo")
                            .font(.title2)
                        Text(feed.statusText)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }
                ROIOverlayView(cxN: roiCx, cyN: roiCy, rN: roiRadius, color: settings.roiColor)
                    .allowsHitTesting(false)
            }
            .clipShape(RoundedRectangle(cornerRadius: 12))

            roiControls
        }
        .padding()
        .background(Color.black.opacity(0.9))
        .task { await sendIfNeeded() }
        .onAppear { feed.start(baseURL: appModel.baseURL) }
        .onDisappear { feed.stop() }
        .onChange(of: appModel.baseURL) { _, newValue in
            feed.start(baseURL: newValue)
            Task { await sendIfNeeded() }
        }
        .onChange(of: roiCx) { _, _ in Task { await sendIfNeeded() } }
        .onChange(of: roiCy) { _, _ in Task { await sendIfNeeded() } }
        .onChange(of: roiRadius) { _, _ in Task { await sendIfNeeded() } }
    }

    private var roiControls: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("ROI Controls")
                .font(.headline)
                .foregroundStyle(.white)

            ColorPicker("ROI ring", selection: $settings.roiColor, supportsOpacity: false)
                .labelsHidden()

            roiSlider(title: "Center X", value: $roiCx)
            roiSlider(title: "Center Y", value: $roiCy)
            roiSlider(title: "Radius", value: $roiRadius, range: 0.05...0.9)
        }
    }

    private func roiSlider(title: String, value: Binding<Double>, range: ClosedRange<Double> = 0.0...1.0) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("\(title): \(String(format: "%.2f", value.wrappedValue))")
                .font(.caption)
                .foregroundStyle(.secondary)
            Slider(value: value, in: range)
        }
    }

    private func sendIfNeeded() async {
        guard let baseURL = appModel.baseURL else { return }
        let payload = AvpRoiConfigPayload(
            enabled: true,
            cx_n: clamp(roiCx),
            cy_n: clamp(roiCy),
            r_n: clamp(roiRadius)
        )
        guard payload.isValid else { return }
        let snapshot = AvpRoiConfigSnapshot(payload)
        guard snapshot.shouldSend(comparedTo: lastSent) else { return }
        lastSent = snapshot
        do {
            try await AvpRoiService.update(baseURL: baseURL, roi: payload)
        } catch {
            // Silent failure to avoid UI spam; debug from server if needed.
        }
    }

    private func clamp(_ value: Double) -> Double {
        max(0.0, min(1.0, value))
    }
}

struct ROIOverlayView: View {
    let cxN: Double
    let cyN: Double
    let rN: Double
    let color: Color

    var body: some View {
        GeometryReader { geo in
            let minSide = max(1, min(geo.size.width, geo.size.height))
            let radius = CGFloat(max(0.0, min(1.0, rN))) * minSide
            let center = CGPoint(
                x: CGFloat(max(0.0, min(1.0, cxN))) * geo.size.width,
                y: CGFloat(max(0.0, min(1.0, cyN))) * geo.size.height
            )
            ring(center: center, radius: radius, color: color)
        }
    }

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
}

@MainActor
private final class UxplayFeedModel: ObservableObject {
    @Published var image: UIImage?
    @Published var statusText = "Waiting for UXPlay frame…"

    private var task: Task<Void, Never>?

    func start(baseURL: URL?) {
        stop()
        guard let baseURL else {
            statusText = "Set API host/port first"
            image = nil
            return
        }
        task = Task { await run(baseURL: baseURL) }
    }

    func stop() {
        task?.cancel()
        task = nil
    }

    private func run(baseURL: URL) async {
        while !Task.isCancelled {
            do {
                if let frame = try await fetchFrame(baseURL: baseURL) {
                    image = frame
                    statusText = "UXPlay stream"
                }
            } catch {
                statusText = "Unable to load UXPlay frame"
            }
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
    }

    private func fetchFrame(baseURL: URL) async throws -> UIImage? {
        var comps = URLComponents(url: baseURL.appendingPathComponent("get_avp_latest_frame"),
                                  resolvingAgainstBaseURL: false)!
        comps.queryItems = [URLQueryItem(name: "purpose", value: "roi_selection")]
        let (data, response) = try await URLSession.shared.data(from: comps.url!)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw URLError(.badServerResponse)
        }
        let payload = try JSONDecoder().decode(AVPFrameResponse.self, from: data)
        return decodeImage(from: payload.rgb)
    }

    private func decodeImage(from dataURLString: String?) -> UIImage? {
        guard let dataURLString else { return nil }
        let base64Part: String
        if let commaIndex = dataURLString.firstIndex(of: ",") {
            base64Part = String(dataURLString[dataURLString.index(after: commaIndex)...])
        } else {
            base64Part = dataURLString
        }
        guard let data = Data(base64Encoded: base64Part, options: .ignoreUnknownCharacters) else { return nil }
        return UIImage(data: data)
    }
}

private struct AVPFrameResponse: Decodable {
    let rgb: String?
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
