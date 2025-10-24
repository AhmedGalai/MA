import Foundation
import CoreGraphics
import SwiftUI
import simd

enum OverlayRenderer {
    static func drawOverlay(base: CGImage,
                            mesh: MeshData,
                            K: [[Double]],
                            T: [[Double]],
                            edgeColor: UIColor = .green,
                            thickness: CGFloat = 2.0) -> CGImage? {
        let w = base.width, h = base.height
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { return nil }
        ctx.draw(base, in: CGRect(x: 0, y: 0, width: w, height: h))

        let R = [
            [T[0][0], T[0][1], T[0][2]],
            [T[1][0], T[1][1], T[1][2]],
            [T[2][0], T[2][1], T[2][2]]
        ]
        let t = SIMD3(T[0][3], T[1][3], T[2][3])
        let Kmat = K

        func project(_ X: SIMD3<Double>) -> CGPoint? {
            let Xc = SIMD3<Double>(
                R[0][0]*X.x + R[0][1]*X.y + R[0][2]*X.z + t.x,
                R[1][0]*X.x + R[1][1]*X.y + R[1][2]*X.z + t.y,
                R[2][0]*X.x + R[2][1]*X.y + R[2][2]*X.z + t.z
            )
            if Xc.z <= 1e-9 || !Xc.x.isFinite || !Xc.y.isFinite || !Xc.z.isFinite { return nil }
            let u = Kmat[0][0]*Xc.x/Xc.z + Kmat[0][2]
            let v = Kmat[1][1]*Xc.y/Xc.z + Kmat[1][2]
            if u.isFinite && v.isFinite { return CGPoint(x: u, y: v) }
            return nil
        }

        // Draw edges
        ctx.setStrokeColor(edgeColor.cgColor)
        ctx.setLineWidth(thickness)
        for (i, j) in mesh.edges {
            if i < mesh.vertices.count, j < mesh.vertices.count,
               let p1 = project(mesh.vertices[i]), let p2 = project(mesh.vertices[j]) {
                ctx.beginPath()
                ctx.move(to: p1)
                ctx.addLine(to: p2)
                ctx.strokePath()
            }
        }

        // COM and axes
        let comW = SIMD3(
            R[0][0]*mesh.com.x + R[0][1]*mesh.com.y + R[0][2]*mesh.com.z + t.x,
            R[1][0]*mesh.com.x + R[1][1]*mesh.com.y + R[1][2]*mesh.com.z + t.y,
            R[2][0]*mesh.com.x + R[2][1]*mesh.com.y + R[2][2]*mesh.com.z + t.z
        )
        let axes = [
            (SIMD3(comW.x + mesh.axisLen*R[0][0], comW.y + mesh.axisLen*R[1][0], comW.z + mesh.axisLen*R[2][0]), UIColor.red),   // +X
            (SIMD3(comW.x + mesh.axisLen*R[0][1], comW.y + mesh.axisLen*R[1][1], comW.z + mesh.axisLen*R[2][1]), UIColor.green), // +Y
            (SIMD3(comW.x + mesh.axisLen*R[0][2], comW.y + mesh.axisLen*R[1][2], comW.z + mesh.axisLen*R[2][2]), UIColor.blue)   // +Z
        ]
        if let com2D = project(SIMD3(comW.x, comW.y, comW.z)) {
            for (endW, color) in axes {
                if let end2D = project(endW) {
                    ctx.setStrokeColor(color.cgColor)
                    ctx.setLineWidth(thickness)
                    ctx.beginPath()
                    ctx.move(to: com2D)
                    ctx.addLine(to: end2D)
                    ctx.strokePath()

                    ctx.setFillColor(color.cgColor)
                    ctx.fillEllipse(in: CGRect(x: com2D.x-2, y: com2D.y-2, width: 4, height: 4))
                }
            }
        }

        return ctx.makeImage()
    }
}

