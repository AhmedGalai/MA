import Foundation
import ReplayKit
import CoreMedia
import CoreImage
import CoreGraphics

final class ScreenCapture: NSObject, ObservableObject {
    static let shared = ScreenCapture()
    @Published var latestImage: CGImage?

    private let recorder = RPScreenRecorder.shared()

    func start() {
        guard recorder.isAvailable else { return }
        recorder.isMicrophoneEnabled = false
        recorder.startCapture(handler: { [weak self] sample, bufferType, error in
            if let error = error { print("Capture error:", error); return }
            if bufferType == .video, let img = sample.cgImage() {
                DispatchQueue.main.async { self?.latestImage = img }
            }
        }, completionHandler: { err in
            if let err = err { print("startCapture error:", err) }
        })
    }

    func stop() {
        recorder.stopCapture { err in
            if let err = err { print("stopCapture error:", err) }
        }
    }
}

private extension CMSampleBuffer {
    func cgImage() -> CGImage? {
        guard let pb = CMSampleBufferGetImageBuffer(self) else { return nil }
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext(options: nil)
        return ctx.createCGImage(ci, from: ci.extent)
    }
}

