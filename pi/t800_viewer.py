#!/usr/bin/env python3
"""
T-800 Vision Viewer
====================
Standalone OpenCV window that reads the annotated MJPEG stream from the
brain's dashboard server and displays it on the local screen.

Runs cv2.imshow from the MAIN THREAD (required by Qt5/X11) so the window
always renders correctly, regardless of how the brain is threaded.

Usage:
    python3 t800_viewer.py                    # default localhost:5000
    python3 t800_viewer.py http://pi.local:5000
"""
import sys
import time
import cv2

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000") + "/video_feed"

print(f"[VIEWER] Connecting to {URL}")

# Wait for the brain dashboard to come up
for attempt in range(30):
    cap = cv2.VideoCapture(URL)
    if cap.isOpened():
        break
    cap.release()
    time.sleep(1)
    print(f"[VIEWER] Waiting for stream... ({attempt+1}/30)")
else:
    print("[VIEWER] Could not connect — exiting")
    sys.exit(1)

cv2.namedWindow("T-800 VISION", cv2.WINDOW_NORMAL)
cv2.resizeWindow("T-800 VISION", 854, 480)

print("[VIEWER] Stream connected — displaying feed")
consecutive_failures = 0

while True:
    ret, frame = cap.read()
    if ret:
        consecutive_failures = 0
        cv2.imshow("T-800 VISION", frame)
    else:
        consecutive_failures += 1
        if consecutive_failures > 60:
            print("[VIEWER] Stream lost — reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(URL)
            consecutive_failures = 0
        time.sleep(0.05)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q or ESC
        break

cap.release()
cv2.destroyAllWindows()
print("[VIEWER] Closed")
