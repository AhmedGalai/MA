//
//  Extensions.swift
//  PoseApp
//
//  Created by match on 19.10.25.
//

import UIKit

extension UIImage {
    func toBase64() -> String? {
        guard let data = self.pngData() else { return nil }
        return data.base64EncodedString()
    }
}

