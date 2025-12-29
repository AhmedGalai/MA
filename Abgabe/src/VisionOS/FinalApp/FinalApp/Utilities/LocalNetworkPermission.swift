import Foundation
import Network

enum LocalNetworkPermission {
    private static var browser: NWBrowser?

    static func request() {
        guard browser == nil else { return }
        let params = NWParameters.tcp
        let browser = NWBrowser(for: .bonjour(type: "_http._tcp", domain: nil), using: params)
        self.browser = browser
        browser.browseResultsChangedHandler = { _, _ in }
        browser.stateUpdateHandler = { state in
            if case .ready = state {
                browser.cancel()
                self.browser = nil
            }
        }
        browser.start(queue: .main)

        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            browser.cancel()
            if self.browser === browser {
                self.browser = nil
            }
        }
    }
}
