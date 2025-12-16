//
//  MatrixUtils.swift
//  PoseOverlayApp
//
//  Created by match on 30.10.25.
//

import simd

enum MatrixUtils {
    /// Convert row-major JSON [[r11,r12,r13,tx],[r21,...,ty],[r31,...,tz],[0,0,0,1]]
    /// to column-major simd_float4x4 (RealityKit).
    static func simdMatrix(from dto: Matrix4x4DTO) -> simd_float4x4 {
        let m = dto.rows.map { $0.map { Float($0.value) } } // 4x4 row-major
        let c0 = SIMD4<Float>(m[0][0], m[1][0], m[2][0], m[3][0])
        let c1 = SIMD4<Float>(m[0][1], m[1][1], m[2][1], m[3][1])
        let c2 = SIMD4<Float>(m[0][2], m[1][2], m[2][2], m[3][2])
        let c3 = SIMD4<Float>(m[0][3], m[1][3], m[2][3], m[3][3])
        return simd_float4x4(columns: (c0, c1, c2, c3))
    }

    /// Convert a matrix expressed in OpenCV camera coordinates (x right, y down, z forward)
    /// into RealityKit's camera coordinate space (x right, y up, z backward).
    static func convertOpenCVToRealityKit(_ matrix: simd_float4x4) -> simd_float4x4 {
        let flip = simd_float4x4(
            SIMD4<Float>(1, 0, 0, 0),
            SIMD4<Float>(0,-1, 0, 0),
            SIMD4<Float>(0, 0,-1, 0),
            SIMD4<Float>(0, 0, 0, 1)
        )
        return flip * matrix * flip
    }

    /// Convenience for raw [[Double]] matrices coming from debug endpoints.
    static func simdMatrix(from rows: [[Double]]) -> simd_float4x4 {
        let m = rows.map { $0.map { Float($0) } }
        let c0 = SIMD4<Float>(m[0][0], m[1][0], m[2][0], m[3][0])
        let c1 = SIMD4<Float>(m[0][1], m[1][1], m[2][1], m[3][1])
        let c2 = SIMD4<Float>(m[0][2], m[1][2], m[2][2], m[3][2])
        let c3 = SIMD4<Float>(m[0][3], m[1][3], m[2][3], m[3][3])
        return simd_float4x4(columns: (c0, c1, c2, c3))
    }
}
