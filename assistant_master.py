import cv2
import numpy as np
import socket
import sys
from ultralytics import YOLO
import mediapipe as mp

# ==================================================
# 🌐 GLOBAL NETWORKING HARDWARE BINDINGS
# ==================================================
PC_IP = "192.168.29.112"              
PORT = 5005                           
phone_ip_address = "192.168.29.172"
CAMERA_URL = "http://" + phone_ip_address + ":8080/video"

server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ==================================================
# 🎮 CHOOSE SYSTEM MODE
# ==================================================
print("\n==================================================")
print("  MULTI-MODAL ASSISTIVE ROBOT INTEGRATION CORE   ")
print("==================================================")
print(" SELECT OPERATIONAL MODE FOR THE DISTRIBUTED SYSTEM:")
print(" [1] -> Caregiver Following Mode (YOLOv8 Neural Net)")
print(" [2] -> Patient Neuro-Gaze Mode (MediaPipe Mesh)")
print("==================================================")
user_choice = input(" Enter Mode Number (1 or 2): ").strip()

# Initialize selected frameworks
if user_choice == "1":
    print("\n[INIT] Initializing YOLOv8 Object Tracking Arrays...")
    model = YOLO("yolov8n.pt")
elif user_choice == "2":
    print("\n[INIT] Initializing MediaPipe Face Mesh Medical Models...")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)
    # Eye vector key indices
    LEFT_PUPIL, LEFT_INNER, LEFT_OUTER = 468, 362, 263
else:
    print("[ERROR] Invalid selection. Shutting down system matrix.")
    sys.exit()

cap = cv2.VideoCapture(CAMERA_URL)
if not cap.isOpened():
    print("[FATAL ERROR] Could not connect to the network video stream.")
    sys.exit()

print("\n==================================================")
print("        ECOSYSTEM RUNNING - STATUS: ACTIVE        ")
print(" Press 'q' on your keyboard to exit safely.       ")
print("==================================================")

last_sent_command = "STOP"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    screen_center_x = int(width / 2)
    current_command = "STOP"

    # ==================================================
    # 👕 MODE 1: YOLOv8 CAREGIVER TRACKER RUNTIME
    # ==================================================
    if user_choice == "1":
        horizon_cutoff = int(height * 0.35)
        roi_frame = frame[horizon_cutoff:height, 0:width]
        results = model(roi_frame, stream=True, verbose=False)
        target_detected = False

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if model.names[int(box.cls[0])] == "person" and not target_detected:
                    target_detected = True
                    obj_x = int((x1 + x2) / 2)
                    box_width = x2 - x1

                    cv2.rectangle(frame, (x1, y1 + horizon_cutoff), (x2, y2 + horizon_cutoff), (0, 255, 0), 2)
                    cv2.putText(frame, "TARGET: CAREGIVER", (x1, y1 + horizon_cutoff - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if obj_x < (screen_center_x - 90): current_command = "LEFT"
                    elif obj_x > (screen_center_x + 90): current_command = "RIGHT"
                    else: current_command = "FORWARD" if box_width < 180 else "STOP"

        cv2.line(frame, (0, horizon_cutoff), (width, horizon_cutoff), (0, 0, 255), 2)

    # ==================================================
    # 🧠 MODE 2: MEDIAPIPE PATIENT EYE CONTROLLER RUNTIME
    # ==================================================
    elif user_choice == "2":
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                p_x = int(landmarks.landmark[LEFT_PUPIL].x * width)
                p_y = int(landmarks.landmark[LEFT_PUPIL].y * height)
                i_x = int(landmarks.landmark[LEFT_INNER].x * width)
                o_x = int(landmarks.landmark[LEFT_OUTER].x * width)

                eye_width = max(1, o_x - i_x)
                gaze_ratio = (p_x - i_x) / eye_width

                cv2.circle(frame, (p_x, p_y), 3, (0, 255, 0), -1)
                cv2.circle(frame, (i_x, p_y), 2, (0, 0, 255), -1)
                cv2.circle(frame, (o_x, p_y), 2, (0, 0, 255), -1)

                if gaze_ratio < 0.42: current_command = "RIGHT"
                elif gaze_ratio > 0.58: current_command = "LEFT"
                else: current_command = "FORWARD"

    # ==================================================
    # 📡 UNIFIED TRANSMISSION LAYER
    # ==================================================
    if current_command != last_sent_command:
        try:
            server_socket.sendto(current_command.encode(), (PC_IP, PORT))
            print(f"[NET BURST] Mode {user_choice} Vector -> {current_command}")
            last_sent_command = current_command
        except Exception:
            pass

    cv2.line(frame, (screen_center_x, 0), (screen_center_x, height), (255, 255, 255), 1)
    cv2.putText(frame, f"AI COMMAND: {current_command}", (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("Unified Assistive Ecosystem Diagnostic Window", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
server_socket.close()
cv2.destroyAllWindows()
