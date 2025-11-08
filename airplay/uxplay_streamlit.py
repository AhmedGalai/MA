
#You said:
#!/usr/bin/env python3
import os, sys, time, json, subprocess, threading, atexit
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2 as cv
import streamlit as st

# Prefer software H.264 to avoid v4l2 decoder churn
os.environ.setdefault(
    "GST_PLUGIN_FEATURE_RANK",
    "v4l2h264dec:0;v4l2codecs*dec:0;avdec_h264:999"
)

def ts(): return f"[{time.strftime('%H:%M:%S')}]"
def dprint(*a): print(ts(), *a, flush=True)

# ---------- ArUco helpers ----------
def get_aruco_handles():
    if not hasattr(cv, "aruco"):
        dprint("cv2.aruco not available; detection disabled")
        return None, None, None
    aruco = cv.aruco
    try:
        dct = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    except Exception:
        dct = aruco.Dictionary_get(aruco.DICT_4X4_50)
    try:
        params = aruco.DetectorParameters()
    except Exception:
        params = aruco.DetectorParameters_create()
    if hasattr(aruco, "ArucoDetector"):
        det = aruco.ArucoDetector(dct, params)
        api = "new"
    else:
        det = params
        api = "old"
    dprint("ArUco API:", api)
    return dct, det, api

ROWS, COLS = 3, 4
MARKER_SIZE_M = 0.030
SEPARATION_M  = 0.010

def make_gridboard(dct):
    if dct is None: return None
    aruco = cv.aruco
    try:
        return aruco.GridBoard((COLS, ROWS), MARKER_SIZE_M, SEPARATION_M, dct)
    except Exception:
        return aruco.GridBoard_create(COLS, ROWS, MARKER_SIZE_M, SEPARATION_M, dct)

def board_id_to_corners_m(marker_id: int):
    if marker_id < 0 or marker_id >= ROWS*COLS: return None
    row, col = divmod(marker_id, COLS)
    x0 = col * (MARKER_SIZE_M + SEPARATION_M)
    y0 = row * (MARKER_SIZE_M + SEPARATION_M)
    return np.array([[x0,               y0,               0],
                     [x0+MARKER_SIZE_M, y0,               0],
                     [x0+MARKER_SIZE_M, y0+MARKER_SIZE_M, 0],
                     [x0,               y0+MARKER_SIZE_M, 0]], dtype=np.float32)

def solve_board_pose_fallback(corners, ids, K, dist):
    if ids is None or len(ids)==0: return None
    obj_pts, img_pts = [], []
    for i, c in zip(ids.flatten().tolist(), corners):
        obj = board_id_to_corners_m(i)
        if obj is None: continue
        pts = np.asarray(c, dtype=np.float32).reshape(-1,2)
        obj_pts.append(obj); img_pts.append(pts)
    if not obj_pts: return None
    obj_pts = np.concatenate(obj_pts, axis=0)
    img_pts = np.concatenate(img_pts, axis=0)
    ok, rvec, tvec = cv.solvePnP(obj_pts, img_pts, K, dist, flags=cv.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv.solvePnP(obj_pts, img_pts, K, dist)
        if not ok: return None
    return rvec.reshape(3), tvec.reshape(3)

def draw_axis(bgr, K, dist, rvec, tvec, s=0.05):
    axis = np.float32([[0,0,0],[s,0,0],[0,s,0],[0,0,s]])
    proj, _ = cv.projectPoints(axis, rvec, tvec, K, dist)
    p = proj.reshape(-1,2).astype(int)
    cv.line(bgr, tuple(p[0]), tuple(p[1]), (0,0,255), 2)
    cv.line(bgr, tuple(p[0]), tuple(p[2]), (0,255,0), 2)
    cv.line(bgr, tuple(p[0]), tuple(p[3]), (255,0,0), 2)

def draw_detected(bgr, corners, ids):
    out = bgr.copy()
    if ids is not None and len(ids)>0:
        try:
            cv.aruco.drawDetectedMarkers(out, corners, ids)
        except Exception:
            for c in corners:
                pts = np.asarray(c, dtype=np.int32).reshape(-1,2)
                for i in range(4):
                    cv.line(out, tuple(pts[i]), tuple(pts[(i+1)%4]), (0,255,255), 2)
    return out

# ---------- Receiver + async reader ----------
@dataclass
class AirPlayReceiver:
    uxplay_path: str
    name: str
    width: int
    height: int
    proc: Optional[subprocess.Popen] = None

    def _supports_short_name(self) -> bool:
        try:
            p = subprocess.run([self.uxplay_path, "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            txt = (p.stdout + "\n" + p.stderr)
            return "\n  -n " in txt or " -n " in txt
        except Exception:
            return True

    def start(self):
        self.stop()
        sink = f"videoconvert ! videoscale ! video/x-raw,format=BGR,width={self.width},height={self.height} ! fdsink fd=1 sync=false"
        if self._supports_short_name():
            cmd = [self.uxplay_path, "-n", self.name, "-as", "0", "-vs", sink]
        else:
            cmd = [self.uxplay_path, "-as", "0", "-vs", sink]
        dprint("Starting UxPlay:", " ".join(cmd))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7)
        threading.Thread(target=self._pump_stderr, daemon=True).start()
        atexit.register(self.stop)

    def _pump_stderr(self):
        if not self.proc or not self.proc.stderr: return
        for line in iter(self.proc.stderr.readline, b''):
            dprint("UxPlay:", line.decode(errors="replace").rstrip())

    def stop(self):
        if self.proc and self.proc.poll() is None:
            dprint("Stopping UxPlay")
            try:
                self.proc.terminate()
                try: self.proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired: self.proc.kill()
            except Exception:
                pass
        self.proc = None

class FrameReader:
    def __init__(self, rx: AirPlayReceiver):
        self.rx = rx
        self.w = rx.width
        self.h = rx.height
        self._nbytes = self.w * self.h * 3
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._alive = False
        self._frames = 0
        self._last_ts = 0.0

    def start(self):
        if self._alive: return
        if not self.rx.proc or not self.rx.proc.stdout:
            raise RuntimeError("Receiver not started")
        self._alive = True
        threading.Thread(target=self._run, daemon=True).start()
        dprint("Capture source opened (async reader)")

    def _run(self):
        buf = bytearray(self._nbytes)
        view = memoryview(buf)
        while self._alive and self.rx.proc and self.rx.proc.stdout:
            read_total = 0
            while read_total < self._nbytes:
                chunk = self.rx.proc.stdout.read(self._nbytes - read_total)
                if not chunk:
                    time.sleep(0.01)
                    break
                view[read_total:read_total+len(chunk)] = chunk
                read_total += len(chunk)
            if read_total != self._nbytes:
                continue
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((self.h, self.w, 3)).copy()
            with self._lock:
                self._latest = frame
                self._frames += 1
                self._last_ts = time.time()

    def get_latest(self) -> Tuple[Optional[np.ndarray], int, float]:
        with self._lock:
            f = None if self._latest is None else self._latest.copy()
            return f, self._frames, self._last_ts

    def stop(self):
        self._alive = False

# ---------- Intrinsics cache ----------
CACHE_PATH = "intrinsics_cache.json"

def load_cached_intrinsics():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r") as f:
                d = json.load(f)
            K = np.array(d["K"], np.float32)
            dist = np.array(d["dist"], np.float32)
            size = tuple(d["img_size"])
            return K, dist, size
        except Exception:
            pass
    return None, None, None

def save_cached_intrinsics(K, dist, img_size):
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump({"K": K.tolist(), "dist": dist.tolist(), "img_size": list(img_size)}, f, indent=2)
    except Exception:
        pass

def default_K_for_size(w, h):
    f = 0.8 * max(w, h)
    cx, cy = w/2.0, h/2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((5,1), np.float32)
    return K, dist

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AirPlay Vision (async)", layout="wide")

st.sidebar.header("UxPlay")
uxplay_path = st.sidebar.text_input("UxPlay path", "/home/ag/Desktop/MA/airplay/UxPlay/build/uxplay")
svc_name    = st.sidebar.text_input("Service name", "AirPlay-Pipeline")
W           = st.sidebar.number_input("Width",  min_value=160, max_value=1920, value=960, step=2)
H           = st.sidebar.number_input("Height", min_value=120, max_value=1080, value=540, step=2)

st.sidebar.header("Timing")
proc_fps    = st.sidebar.number_input("Process FPS (Python)", min_value=1, max_value=30, value=2, step=1)
ui_hz       = st.sidebar.slider("Auto-refresh (Hz)", 1, 20, 6)  # << NEW
run         = st.sidebar.toggle("Start / Stop", value=False)

st.sidebar.header("Views")
show_detection = st.sidebar.toggle("Detection view", value=True)
show_intrinsics= st.sidebar.toggle("Intrinsics view", value=False)
show_disparity = st.sidebar.toggle("Disparity view (placeholder)", value=False)
show_roi       = st.sidebar.toggle("ROI view (HSV near picked color)", value=False)
show_contours  = st.sidebar.toggle("Active contours (needs ROI)", value=False)

st.sidebar.header("ROI color")
hex_color   = st.sidebar.color_picker("Pick color", "#22AAFF")
h_tol_deg   = st.sidebar.slider("Hue tol (deg)", 5, 30, 12)
s_tol_pct   = st.sidebar.slider("Sat tol (%)", 10, 60, 50)
v_tol_pct   = st.sidebar.slider("Val tol (%)", 10, 60, 50)

clear_cache = st.sidebar.button("Clear intrinsics cache")

col1, col2, col3 = st.columns(3)
ph_rgb  = col1.empty()
ph_det  = col2.empty()
ph_misc = col3.empty()
status  = st.empty()

# Session vars
if "rx" not in st.session_state: st.session_state.rx = None
if "reader" not in st.session_state: st.session_state.reader = None
if "last_proc_t" not in st.session_state: st.session_state.last_proc_t = 0.0
if "K_cache" not in st.session_state:
    Kc,Dc,Sz = load_cached_intrinsics()
    st.session_state.K_cache, st.session_state.D_cache, st.session_state.ImSz = Kc, Dc, Sz

# ArUco init
_ARUCO_DICT, _DET_OR_PARAMS, _API = get_aruco_handles()
_BOARD = make_gridboard(_ARUCO_DICT)

if clear_cache and os.path.exists(CACHE_PATH):
    os.remove(CACHE_PATH)
    st.session_state.K_cache = None
    st.session_state.D_cache = None
    st.session_state.ImSz = None
    st.sidebar.success("Cleared intrinsics_cache.json")

def ensure_receiver():
    rx = st.session_state.rx
    need = (rx is None or rx.uxplay_path != uxplay_path or rx.name != svc_name or rx.width != int(W) or rx.height != int(H))
    if need:
        if rx:
            rx.stop()
            dprint("Capture released")
        st.session_state.rx = AirPlayReceiver(uxplay_path, svc_name, int(W), int(H))
        dprint(f"Receiver reconfigured: {svc_name} {W}x{H}")
    return st.session_state.rx

def ensure_reader():
    rd = st.session_state.reader
    rx = st.session_state.rx
    if rd is None or rd.w != rx.width or rd.h != rx.height:
        st.session_state.reader = FrameReader(rx)
    return st.session_state.reader

def hex_to_hsv_bounds(hex_str, h_deg, s_pct, v_pct):
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16); g = int(hex_str[2:4], 16); b = int(hex_str[4:6], 16)
    bgr = np.uint8([[[b,g,r]]])
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)[0,0,:].astype(int)
    h0,s0,v0 = hsv.tolist()
    dh = int(round((h_deg/360.0)*179.0))
    ds = int(round((s_pct/100.0)*255.0))
    dv = int(round((v_pct/100.0)*255.0))
    lo = np.array([max(0, h0-dh), max(0, s0-ds), max(0, v0-dv)], dtype=np.uint8)
    hi = np.array([min(179, h0+dh), min(255, s0+ds), min(255, v0+dv)], dtype=np.uint8)
    return lo, hi

def make_info_panel(K, dist, size):
    panel = np.zeros((160, 380, 3), np.uint8)
    y=24
    def put(s):
        nonlocal y
        cv.putText(panel, s, (8,y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv.LINE_AA)
        y += 22
    if K is None:
        put("Intrinsics: default (no cache)")
    else:
        put("Intrinsics:")
        put(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f}")
        put(f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
    if size: put(f"img={size[0]}x{size[1]}")
    return panel

# ---------- Main (nonblocking) ----------
if run:
    rx = ensure_receiver()
    if rx.proc is None:
        rx.start()
    reader = ensure_reader()
    try:
        reader.start()
    except RuntimeError as e:
        status.error(str(e))

    frame, count, last_ts = reader.get_latest()
    if frame is None:
        status.warning("No frame yet. Start mirroring to the AirPlay name.")
        blank = np.zeros((int(H), int(W), 3), np.uint8)
        ph_rgb.image(blank, caption="RGB", width="stretch")
        ph_det.image(blank, caption="Detection", width="stretch")
        ph_misc.image(blank, caption="Misc", width="stretch")
    else:
        h, w = frame.shape[:2]
        if st.session_state.K_cache is None or st.session_state.ImSz != (w,h):
            K_use, D_use = default_K_for_size(w, h)
        else:
            K_use, D_use = st.session_state.K_cache, st.session_state.D_cache

        ph_rgb.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB), caption=f"RGB ({w}x{h})", width="stretch")

        min_dt = 1.0 / float(proc_fps)
        now = time.time()
        do_process = (now - st.session_state.last_proc_t) >= min_dt

        det_img = None
        if do_process and show_detection and _ARUCO_DICT is not None:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if _API == "new":
                corners, ids, _rej = _DET_OR_PARAMS.detectMarkers(gray)
            else:
                corners, ids, _rej = cv.aruco.detectMarkers(gray, _ARUCO_DICT, parameters=_DET_OR_PARAMS)
            det_img = draw_detected(frame, corners, ids)
            if ids is not None and len(ids)>0 and show_intrinsics:
                sol = solve_board_pose_fallback(corners, ids, K_use, D_use)
                if sol is not None:
                    rvec, tvec = sol
                    try: draw_axis(det_img, K_use, D_use, rvec, tvec)
                    except Exception: pass
            dprint(f"Detection: ids={0 if ids is None else len(ids)}")

        if det_img is not None:
            ph_det.image(cv.cvtColor(det_img, cv.COLOR_BGR2RGB), caption="Detection", width="stretch")
        else:
            ph_det.image(np.zeros((h, w, 3), np.uint8), caption="Detection (idle / gated)", width="stretch")

        misc_panels = []
        if show_intrinsics:
            misc_panels.append(make_info_panel(K_use, D_use, (w,h)))

        if show_disparity:
            disp = np.zeros((h, w, 3), np.uint8)
            cv.putText(disp, "Disparity placeholder (no depth)", (10,30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv.LINE_AA)
            misc_panels.append(disp)

        if do_process and show_roi:
            lo, hi = hex_to_hsv_bounds(hex_color, h_tol_deg, s_tol_pct, v_tol_pct)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            m = cv.inRange(hsv, lo, hi)
            roi = cv.bitwise_and(frame, frame, mask=m)
            misc_panels.append(roi)

        if do_process and show_contours and show_roi:
            lo, hi = hex_to_hsv_bounds(hex_color, h_tol_deg, s_tol_pct, v_tol_pct)
            hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
            m = cv.inRange(hsv, lo, hi)
            cnts, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            ac = frame.copy()
            if cnts:
                c = max(cnts, key=cv.contourArea)
                cv.drawContours(ac, [c], -1, (0,0,255), 2)
            misc_panels.append(ac)

        if misc_panels:
            target_w = w
            res = []
            for p in misc_panels:
                phh, pww = p.shape[:2]
                if pww != target_w:
                    scale = target_w / float(pww)
                    p = cv.resize(p, (target_w, int(phh*scale)), interpolation=cv.INTER_AREA)
                if p.ndim == 2: p = cv.cvtColor(p, cv.COLOR_GRAY2BGR)
                res.append(p)
            stack = np.vstack(res)
            ph_misc.image(cv.cvtColor(stack, cv.COLOR_BGR2RGB), caption="Misc", width="stretch")
        else:
            ph_misc.image(np.zeros((h, w, 3), np.uint8), caption="Misc (idle / gated)", width="stretch")

        if do_process:
            st.session_state.last_proc_t = now

        age = now - last_ts
        status.info(f"Frames read: {count} | Latest age: {age:.2f}s | Proc FPS={proc_fps} | UI={ui_hz}Hz | Name='{svc_name}'")

    # -------------- AUTO-REFRESH --------------
    # Re-run the script at a controlled UI refresh rate while running.
    # This keeps the views updating without touching widgets.
    time.sleep(1.0 / float(ui_hz))
    st.rerun()

else:
    # stop everything
    if st.session_state.reader:
        st.session_state.reader.stop()
        st.session_state.reader = None
    if st.session_state.rx:
        st.session_state.rx.stop()
    status.warning("Stopped. Toggle Start / Stop to run.")
    # No rerun here

