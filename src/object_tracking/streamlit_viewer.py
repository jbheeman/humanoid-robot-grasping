import time
from datetime import datetime

import cv2
import streamlit as st


DEFAULT_PIPELINE = (
    "udpsrc port=5600 ! "
    "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000 ! "
    "queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink sync=false drop=true max-buffers=1"
)


def opencv_gstreamer_enabled() -> bool:
    try:
        info = cv2.getBuildInformation()
        return "GStreamer:                   YES" in info or "GStreamer: YES" in info
    except Exception:
        return False


def open_capture(pipeline: str):
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap


st.set_page_config(page_title="Unitree G1 Camera", layout="wide")

st.title("Unitree G1 Camera Stream")

with st.sidebar:
    st.subheader("Camera Source")
    pipeline = st.text_area("GStreamer pipeline", DEFAULT_PIPELINE, height=180)

    target_fps = st.slider("Display FPS", 1, 30, 10)
    mirror = st.checkbox("Mirror image", value=False)
    save_snapshot = st.button("Save snapshot")

    st.subheader("OpenCV")
    st.write(f"OpenCV version: `{cv2.__version__}`")
    st.write(f"GStreamer enabled: `{opencv_gstreamer_enabled()}`")

placeholder = st.empty()
status = st.empty()

cap = open_capture(pipeline)

if not cap.isOpened():
    st.error("Could not open camera pipeline.")
    st.code(pipeline)
    st.stop()

status.success("Camera opened. Receiving frames...")

frame_delay = 1.0 / max(target_fps, 1)
last_snapshot_frame = None

while True:
    ok, frame = cap.read()

    if not ok or frame is None:
        status.warning("No frame received yet. Make sure the Unitree relay is running.")
        time.sleep(0.2)
        continue

    if mirror:
        frame = cv2.flip(frame, 1)

    last_snapshot_frame = frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    placeholder.image(rgb, channels="RGB", use_container_width=True)

    if save_snapshot and last_snapshot_frame is not None:
        path = f"/tmp/unitree_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(path, last_snapshot_frame)
        status.success(f"Saved snapshot: {path}")

    time.sleep(frame_delay)
