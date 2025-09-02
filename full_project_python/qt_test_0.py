import sys, os, time, cv2, requests, base64, io, queue
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
import trimesh

API = "http://localhost:8000"
MODELS_DIR = "models"

# -------- Helpers --------
def list_ply_models(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(".ply")]

def pil_to_b64(img: Image.Image, fmt="JPEG"):
    buf = io.BytesIO(); img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")

def b64_to_pil(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64.encode())))

def get_camera_frame():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read(); cap.release()
    if not ret: return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def qimage_from_pil(pil_img, size=(320,240)):
    img = pil_img.resize(size)
    data = img.tobytes("raw", "RGB")
    qimg = QtGui.QImage(data, img.size[0], img.size[1], QtGui.QImage.Format_RGB888)
    return QtGui.QPixmap.fromImage(qimg)

# -------- Workers --------
class MaskWorker(QtCore.QThread):
    resultReady = QtCore.pyqtSignal(object, object)  # snap, mask_img
    def __init__(self, snap_queue, roi_info, left_img, parent=None):
        super().__init__(parent)
        self.snap_queue = snap_queue
        self.roi_info = roi_info
        self.left_img = left_img
        self.running = True
    def run(self):
        while self.running:
            try:
                snap = self.snap_queue.get(timeout=1)
                if not (self.left_img and self.roi_info):
                    continue
                cx, cy, r = self.roi_info
                payload={"left":pil_to_b64(self.left_img),
                         "roi_center":[cx,cy],
                         "roi_radius":r,
                         "snapshot":pil_to_b64(snap)}
                r2=requests.post(f"{API}/bounding_rect",json=payload).json()
                mask_img = b64_to_pil(r2["mask"])
                self.resultReady.emit(snap, mask_img)
            except queue.Empty:
                continue
            except Exception as e:
                print("MaskWorker error", e)
    def stop(self):
        self.running=False

class PoseWorker(QtCore.QThread):
    resultReady = QtCore.pyqtSignal(object, object, object, object) # snap, mask, depth, resp
    def __init__(self, mask_queue, K, parent=None):
        super().__init__(parent)
        self.mask_queue = mask_queue
        self.K = K
        self.running=True
    def run(self):
        while self.running:
            try:
                snap, mask_img = self.mask_queue.get(timeout=1)
                # Depth
                r=requests.post(f"{API}/depth",json={"rgb":pil_to_b64(snap)},timeout=20).json()
                depth=b64_to_pil(r["depth"])
                # Pose
                pose_req={"camera_matrix":self.K,
                          "images":[{"filename":"snap","rgb":pil_to_b64(snap),"depth":pil_to_b64(depth,fmt="PNG")}],
                          "mesh":"","mask":pil_to_b64(mask_img,fmt="PNG"),"depthscale":0.001}
                resp=requests.post(f"{API}/pose",json=pose_req,timeout=20).json()
                self.resultReady.emit(snap, mask_img, depth, resp)
            except queue.Empty:
                continue
            except Exception as e:
                print("PoseWorker error", e)
    def stop(self):
        self.running=False

# -------- Main Window --------
class PoseApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Pose Estimation Client")
        self.resize(1200, 800)

        self.left_img=None
        self.right_img=None
        self.K=[[700,0,320],[0,700,240],[0,0,1]]
        self.roi_info=None  # (cx,cy,r)
        self.snap_queue=queue.Queue(maxsize=5)
        self.mask_queue=queue.Queue(maxsize=5)

        # --- Top camera preview ---
        topWidget=QtWidgets.QWidget()
        topLayout=QtWidgets.QHBoxLayout(topWidget)
        self.previewLabel=QtWidgets.QLabel("Preview")
        self.previewLabel.setFixedSize(320,240)
        topLayout.addWidget(self.previewLabel)
        self.btnPreviewStart=QtWidgets.QPushButton("Start Preview")
        self.btnPreviewStop=QtWidgets.QPushButton("Stop Preview")
        topLayout.addWidget(self.btnPreviewStart)
        topLayout.addWidget(self.btnPreviewStop)

        self.btnPreviewStart.clicked.connect(self.startPreview)
        self.btnPreviewStop.clicked.connect(self.stopPreview)
        self.previewTimer=QtCore.QTimer()
        self.previewTimer.timeout.connect(self.updatePreview)

        # --- Tabs inside scroll area ---
        self.tabs=QtWidgets.QTabWidget()
        scroll=QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.tabs)

        # Tab1: Intrinsics + ROI
        tab1=QtWidgets.QWidget()
        layout1=QtWidgets.QVBoxLayout(tab1)
        self.modelBox=QtWidgets.QComboBox()
        self.modelBox.addItems(list_ply_models(MODELS_DIR))
        layout1.addWidget(QtWidgets.QLabel("Select Model:"))
        layout1.addWidget(self.modelBox)

        btns=QtWidgets.QHBoxLayout()
        self.btnLeft=QtWidgets.QPushButton("Capture Left")
        self.btnRight=QtWidgets.QPushButton("Capture Right")
        self.btnIntr=QtWidgets.QPushButton("Send Intrinsics")
        btns.addWidget(self.btnLeft); btns.addWidget(self.btnRight); btns.addWidget(self.btnIntr)
        layout1.addLayout(btns)

        imgs=QtWidgets.QHBoxLayout()
        self.leftLabel=QtWidgets.QLabel(); self.leftLabel.setFixedSize(320,240)
        self.rightLabel=QtWidgets.QLabel(); self.rightLabel.setFixedSize(320,240)
        imgs.addWidget(self.leftLabel); imgs.addWidget(self.rightLabel)
        layout1.addLayout(imgs)

        self.roiCanvas=QtWidgets.QLabel("ROI Canvas"); self.roiCanvas.setFixedSize(640,480)
        self.roiCanvas.setStyleSheet("background:black;")
        layout1.addWidget(self.roiCanvas)
        self.btnResetROI=QtWidgets.QPushButton("Reset ROI")
        layout1.addWidget(self.btnResetROI)

        self.tabs.addTab(tab1,"Intrinsics + ROI")

        # Tab2: Pose
        tab2=QtWidgets.QWidget()
        layout2=QtWidgets.QVBoxLayout(tab2)
        ctrl=QtWidgets.QHBoxLayout()
        self.rateSpin=QtWidgets.QDoubleSpinBox(); self.rateSpin.setValue(1.0); self.rateSpin.setRange(0.1,10.0)
        self.btnStart=QtWidgets.QPushButton("Start Estimation")
        self.btnStop=QtWidgets.QPushButton("Stop Estimation")
        ctrl.addWidget(QtWidgets.QLabel("Rate (Hz):")); ctrl.addWidget(self.rateSpin)
        ctrl.addWidget(self.btnStart); ctrl.addWidget(self.btnStop)
        layout2.addLayout(ctrl)

        imgs2=QtWidgets.QHBoxLayout()
        self.liveLabel=QtWidgets.QLabel(); self.liveLabel.setFixedSize(320,240)
        self.depthLabel=QtWidgets.QLabel(); self.depthLabel.setFixedSize(320,240)
        self.overlayLabel=QtWidgets.QLabel(); self.overlayLabel.setFixedSize(320,240)
        self.maskedLabel=QtWidgets.QLabel(); self.maskedLabel.setFixedSize(320,240)
        for w in [self.liveLabel,self.depthLabel,self.overlayLabel,self.maskedLabel]:
            imgs2.addWidget(w)
        layout2.addLayout(imgs2)

        self.logBox=QtWidgets.QPlainTextEdit(); self.logBox.setReadOnly(True)
        layout2.addWidget(self.logBox)

        self.tabs.addTab(tab2,"Pose Estimation")

        # Main layout
        central=QtWidgets.QWidget()
        vbox=QtWidgets.QVBoxLayout(central)
        vbox.addWidget(topWidget)
        vbox.addWidget(scroll)
        self.setCentralWidget(central)

        # Signals
        self.btnStart.clicked.connect(self.startEstimation)
        self.btnStop.clicked.connect(self.stopEstimation)

        # Workers
        self.maskWorker=None
        self.poseWorker=None

        # --- in PoseApp.__init__() ---
        self.btnLeft.clicked.connect(self.captureLeft)
        self.btnRight.clicked.connect(self.captureRight)
        self.btnIntr.clicked.connect(self.sendIntrinsics)
        self.btnResetROI.clicked.connect(self.resetROI)

        # Enable mouse events on ROI canvas
        self.roiCanvas.setMouseTracking(True)
        self.roiCanvas.mousePressEvent = self.roiClick
        self.roiCanvas.mouseMoveEvent = self.roiMove
        self.roi_center = None
        self.roi_radius = None
        self.roi_preview = None


    # --- Preview ---
    def startPreview(self):
        self.previewTimer.start(100)
    def stopPreview(self):
        self.previewTimer.stop()
    def updatePreview(self):
        frame=get_camera_frame()
        if frame:
            self.previewLabel.setPixmap(qimage_from_pil(frame))

    def captureLeft(self):
        frame = get_camera_frame()
        if frame:
            self.left_img = frame
            self.leftLabel.setPixmap(qimage_from_pil(frame))
            self.roiCanvas.setPixmap(qimage_from_pil(frame, size=(640,480)))  # show on canvas too

    def captureRight(self):
        frame = get_camera_frame()
        if frame:
            self.right_img = frame
            self.rightLabel.setPixmap(qimage_from_pil(frame))

    def roiClick(self, event):
        if self.left_img is None: return
        if self.roi_center is None:
            self.roi_center = (event.x(), event.y())
        else:
            dx = event.x() - self.roi_center[0]
            dy = event.y() - self.roi_center[1]
            self.roi_radius = int((dx**2 + dy**2)**0.5)
            self.updateRoiOverlay(final=True)

    def roiMove(self, event):
        if self.roi_center and self.roi_radius is None:
            dx = event.x() - self.roi_center[0]
            dy = event.y() - self.roi_center[1]
            r = int((dx**2 + dy**2)**0.5)
            self.roi_preview = r
            self.updateRoiOverlay(final=False)

    def updateRoiOverlay(self, final=False):
        # draw ROI circle on top of left image
        pixmap = qimage_from_pil(self.left_img, size=(640,480))
        painter = QtGui.QPainter(pixmap)
        pen = QtGui.QPen(QtCore.Qt.green if final else QtCore.Qt.red, 2, QtCore.Qt.SolidLine if final else QtCore.Qt.DashLine)
        painter.setPen(pen)
        r = self.roi_radius if final else self.roi_preview
        if r:
            cx, cy = self.roi_center
            painter.drawEllipse(QtCore.QPoint(cx, cy), r, r)
        painter.end()
        self.roiCanvas.setPixmap(pixmap)

    def resetROI(self):
        self.roi_center=None; self.roi_radius=None; self.roi_preview=None
        if self.left_img:
            self.roiCanvas.setPixmap(qimage_from_pil(self.left_img, size=(640,480)))


    # --- Estimation ---
    def startEstimation(self):
        self.logBox.appendPlainText("Estimation started")
        self.maskWorker=MaskWorker(self.snap_queue, self.roi_info, self.left_img)
        self.poseWorker=PoseWorker(self.mask_queue, self.K)
        self.maskWorker.resultReady.connect(lambda s,m:self.mask_queue.put((s,m)))
        self.poseWorker.resultReady.connect(self.updatePoseUI)
        self.maskWorker.start()
        self.poseWorker.start()
        self.timer=QtCore.QTimer(); self.timer.timeout.connect(self.loop)
        self.timer.start(int(1000/self.rateSpin.value()))
    def stopEstimation(self):
        if self.maskWorker: self.maskWorker.stop()
        if self.poseWorker: self.poseWorker.stop()
        self.timer.stop()
        self.logBox.appendPlainText("Estimation stopped")
    def loop(self):
        snap=get_camera_frame()
        if snap:
            try: self.snap_queue.put_nowait(snap)
            except queue.Full: pass
    def updatePoseUI(self,snap,mask_img,depth,resp):
        self.liveLabel.setPixmap(qimage_from_pil(snap))
        self.depthLabel.setPixmap(qimage_from_pil(depth))
        self.maskedLabel.setPixmap(qimage_from_pil(mask_img))
        self.overlayLabel.setPixmap(qimage_from_pil(snap)) # TODO overlay mesh
        self.logBox.appendPlainText("Pose update")

    def sendIntrinsics(self):
        if not (self.left_img and self.right_img):
            self.logBox.appendPlainText("❌ Capture left and right images first")
            return
        try:
            payload = {
                "left": pil_to_b64(self.left_img),
                "right": pil_to_b64(self.right_img),
            }
            r = requests.post(f"{API}/intrinsics", json=payload, timeout=10).json()
            self.K = r.get("camera_matrix", self.K)
            self.logBox.appendPlainText(f"✅ Intrinsics estimated: {self.K}")

            # Optional debug
            if "debug_image" in r:
                dbg = b64_to_pil(r["debug_image"])
                self.rightLabel.setPixmap(qimage_from_pil(dbg))

            # Rectify (if endpoint exists)
            rect_req = {"left": payload["left"], "right": payload["right"], "camera_matrix": self.K}
            r2 = requests.post(f"{API}/rectify", json=rect_req, timeout=10).json()
            if "rectified" in r2:
                rect = b64_to_pil(r2["rectified"])
                self.leftLabel.setPixmap(qimage_from_pil(rect))
                self.logBox.appendPlainText("Rectified left image shown")

        except Exception as e:
            self.logBox.appendPlainText(f"❌ Intrinsics API error: {e}")


# --- Main ---
if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    win=PoseApp()
    win.show()
    sys.exit(app.exec_())

