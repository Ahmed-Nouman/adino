#!/usr/bin/env python3
"""
Live stream from a Luxonis OAK-D Pro (DepthAI v3 API).

Windows:
  - "RGB"   : color camera stream
  - "Depth" : colorized stereo depth (click anywhere to read distance)

Keys:
  q / ESC : quit
  i       : toggle IR dot projector (helps depth on textureless surfaces)
  f       : toggle IR flood LED (helps in low light)
"""
import time

import cv2
import numpy as np
import depthai as dai

# ---------------- Settings ----------------
RGB_SIZE = (1280, 720)      # color stream resolution
MONO_SIZE = (640, 400)      # stereo input resolution (OV9282 native is 1280x800)
FPS = 30
DEPTH_MIN_MM = 200          # visualization range
DEPTH_MAX_MM = 8000
IR_DOT_INTENSITY = 0.5      # 0.0 - 1.0
IR_FLOOD_INTENSITY = 0.5    # 0.0 - 1.0
# ------------------------------------------

query_point = None  # (x, y) in depth image, set by mouse click


def on_mouse(event, x, y, flags, param):
    global query_point
    if event == cv2.EVENT_LBUTTONDOWN:
        query_point = (x, y)


def colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    """Map uint16 depth (mm) to a color image; invalid (0) pixels are black."""
    d = np.clip(depth_mm, DEPTH_MIN_MM, DEPTH_MAX_MM).astype(np.float32)
    norm = ((d - DEPTH_MIN_MM) / (DEPTH_MAX_MM - DEPTH_MIN_MM) * 255).astype(np.uint8)
    color = cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)  # near = red, far = blue
    color[depth_mm == 0] = 0
    return color


def depth_at(depth_mm: np.ndarray, x: int, y: int, k: int = 3) -> float:
    """Median depth (mm) in a small window around (x, y), ignoring invalid pixels."""
    h, w = depth_mm.shape
    patch = depth_mm[max(0, y - k):min(h, y + k + 1), max(0, x - k):min(w, x + k + 1)]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else 0.0


def main():
    global query_point

    with dai.Pipeline() as pipeline:
        device = pipeline.getDefaultDevice()
        print("Connected:", device.getDeviceName())
        try:
            print("USB speed:", device.getUsbSpeed().name)  # should be SUPER / SUPER_PLUS
        except Exception:
            pass

        # --- Cameras ---
        cam_rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A, sensorFps=FPS)
        cam_left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B, sensorFps=FPS)
        cam_right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C, sensorFps=FPS)

        # --- Stereo depth ---
        stereo = pipeline.create(dai.node.StereoDepth).build(
            cam_left.requestOutput(MONO_SIZE),
            cam_right.requestOutput(MONO_SIZE),
        )
        stereo.setRectification(True)
        stereo.setLeftRightCheck(True)      # better occlusion handling
        stereo.setSubpixel(True)            # better accuracy at longer range
        stereo.setExtendedDisparity(False)  # set True for very close objects

        # --- Output queues (non-blocking, keep only latest frames) ---
        q_rgb = cam_rgb.requestOutput(RGB_SIZE).createOutputQueue(maxSize=4, blocking=False)
        q_depth = stereo.depth.createOutputQueue(maxSize=4, blocking=False)

        pipeline.start()

        # OAK-D Pro IR illumination
        ir_dot_on, ir_flood_on = True, False
        try:
            device.setIrLaserDotProjectorIntensity(IR_DOT_INTENSITY)
            device.setIrFloodLightIntensity(0.0)
        except Exception as e:
            print("IR control not available:", e)

        cv2.namedWindow("Depth")
        cv2.setMouseCallback("Depth", on_mouse)

        rgb_frame, depth_frame = None, None
        frames, t0, fps = 0, time.time(), 0.0

        while pipeline.isRunning():
            in_rgb = q_rgb.tryGet()
            in_depth = q_depth.tryGet()

            if in_rgb is not None:
                rgb_frame = in_rgb.getCvFrame()
                frames += 1
                if time.time() - t0 >= 1.0:
                    fps = frames / (time.time() - t0)
                    frames, t0 = 0, time.time()

            if in_depth is not None:
                depth_frame = in_depth.getFrame()  # uint16, millimeters

            if rgb_frame is not None:
                show = rgb_frame.copy()
                cv2.putText(show, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("RGB", show)

            if depth_frame is not None:
                vis = colorize_depth(depth_frame)
                h, w = depth_frame.shape
                px, py = query_point if query_point else (w // 2, h // 2)
                dist = depth_at(depth_frame, px, py)
                cv2.drawMarker(vis, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
                label = f"{dist / 1000:.2f} m" if dist > 0 else "no depth"
                cv2.putText(vis, label, (px + 10, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ir_txt = f"IR dot: {'on' if ir_dot_on else 'off'} | flood: {'on' if ir_flood_on else 'off'}"
                cv2.putText(vis, ir_txt, (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("Depth", vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("i"):
                ir_dot_on = not ir_dot_on
                device.setIrLaserDotProjectorIntensity(IR_DOT_INTENSITY if ir_dot_on else 0.0)
            elif key == ord("f"):
                ir_flood_on = not ir_flood_on
                device.setIrFloodLightIntensity(IR_FLOOD_INTENSITY if ir_flood_on else 0.0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
