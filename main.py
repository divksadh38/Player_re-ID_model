import cv2
from ultralytics import YOLO
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch

# Load YOLOv8 model
model = YOLO("best.pt")
model.overrides["imgsz"]=640

# Initialize DeepSORT tracker
import torch

tracker = DeepSort(
    max_age=60,
    n_init=2,
    nn_budget=100,
    embedder='torchreid',
    embedder_model_name='osnet_x1_0',          
    embedder_wts='osnet_x1_0_market1501.pth',            
    embedder_gpu=torch.cuda.is_available(),              
    nms_max_overlap=0.7,
    bgr=True,
    half=True
)

CONF_THRESHOLD = 0.7
PLAYER_CLASS_ID = 2

cap = cv2.VideoCapture("15sec_input_720p.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    boxes_xywh = []
    confs = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD or cls != PLAYER_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w = x2 - x1
        h = y2 - y1
        boxes_xywh.append([x1, y1, w, h])
        confs.append(conf)



    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv11 + DeepSORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
