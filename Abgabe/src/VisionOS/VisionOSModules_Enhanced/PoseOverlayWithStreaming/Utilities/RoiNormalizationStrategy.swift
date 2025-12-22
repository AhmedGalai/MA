import CoreGraphics
import Foundation
#if canImport(UIKit)
import UIKit
#endif

struct AvpRoiConfigPayload: Encodable {
    let enabled: Bool
    let cx_n: Double
    let cy_n: Double
    let r_n: Double

    var isValid: Bool {
        guard cx_n.isFinite, cy_n.isFinite, r_n.isFinite else { return false }
        return (0.0...1.0).contains(cx_n) && (0.0...1.0).contains(cy_n) && (0.0...1.0).contains(r_n)
    }
}

enum RoiNormalizationStrategy {
    static func normalizedROI(size: CGSize, globalFrame: CGRect, radius: CGFloat) -> AvpRoiConfigPayload {
        let clampedRadius = max(0, radius)
#if ROI_OPTION_PROJECTED
        return normalizedProjectedROI(size: size, globalFrame: globalFrame, radius: clampedRadius)
#elseif ROI_OPTION_OVERLAY
        return normalizedWindowROI(size: size, radius: clampedRadius)
#else
        return normalizedWindowROI(size: size, radius: clampedRadius)
#endif
    }

    private static func normalizedWindowROI(size: CGSize, radius: CGFloat) -> AvpRoiConfigPayload {
        let minSide = max(1.0, min(size.width, size.height))
        let r = Double(radius / minSide)
        return AvpRoiConfigPayload(
            enabled: true,
            cx_n: 0.5,
            cy_n: 0.5,
            r_n: clamp(r, 0.0, 1.0)
        )
    }

#if ROI_OPTION_PROJECTED
    private static func normalizedProjectedROI(size: CGSize, globalFrame: CGRect, radius: CGFloat) -> AvpRoiConfigPayload {
        let screenSize = screenBounds(fallback: size)
        let denom = max(1.0, min(screenSize.width, screenSize.height))
        let cx = globalFrame.midX / max(1.0, screenSize.width)
        let cy = globalFrame.midY / max(1.0, screenSize.height)
        let r = Double(radius / denom)
        return AvpRoiConfigPayload(
            enabled: true,
            cx_n: clamp(Double(cx), 0.0, 1.0),
            cy_n: clamp(Double(cy), 0.0, 1.0),
            r_n: clamp(r, 0.0, 1.0)
        )
    }

    private static func screenBounds(fallback: CGSize) -> CGSize {
#if canImport(UIKit) && !os(visionOS)
        let bounds = UIScreen.main.bounds.size
        if bounds.width > 0, bounds.height > 0 {
            return bounds
        }
#endif
        return fallback
    }
#endif

    private static func clamp(_ v: Double, _ lo: Double, _ hi: Double) -> Double {
        max(lo, min(hi, v))
    }
}
