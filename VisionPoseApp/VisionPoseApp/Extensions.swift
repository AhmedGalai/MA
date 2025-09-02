import UIKit

extension UIImage {
    func toBase64() -> String? {
        guard let data = self.pngData() else { return nil }
        return data.base64EncodedString()
    }
}

