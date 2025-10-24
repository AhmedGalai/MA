import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

enum ImageUtils {
    static func jpegData(from cgImage: CGImage, quality: CGFloat = 0.8) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else { return nil }
        let options = [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary
        CGImageDestinationAddImage(dest, cgImage, options)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }

    static func pngData(from cgImage: CGImage) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, cgImage, nil)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }

    static func cgImage(from data: Data) -> CGImage? {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(src, 0, nil)
    }

    static func makeMaskPNG(width: Int, height: Int, center: CGPoint, radius: CGFloat) -> Data? {
        let colorSpace = CGColorSpaceCreateDeviceGray()
        guard let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: width, space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return nil }
        ctx.setFillColor(CGColor(gray: 0, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.setFillColor(CGColor(gray: 1, alpha: 1))
        let rect = CGRect(x: center.x - radius, y: center.y - radius, width: radius*2, height: radius*2)
        ctx.fillEllipse(in: rect)
        guard let mask = ctx.makeImage() else { return nil }
        return pngData(from: mask)
    }

    static func applyMask(rgb: CGImage, maskPNG: Data?) -> CGImage? {
        guard let maskPNG, let maskImg = cgImage(from: maskPNG) else { return rgb }
        let w = rgb.width, h = rgb.height
        guard maskImg.width == w, maskImg.height == h else { return rgb }
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w*4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return rgb }
        ctx.draw(rgb, in: CGRect(x: 0, y: 0, width: w, height: h))
        // Use mask alpha as multiplier
        ctx.saveGState()
        ctx.clip(to: CGRect(x: 0, y: 0, width: w, height: h), mask: maskImg)
        ctx.setBlendMode(.destinationIn)
        ctx.setFillColor(CGColor(gray: 1, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
        ctx.restoreGState()
        return ctx.makeImage()
    }
}

