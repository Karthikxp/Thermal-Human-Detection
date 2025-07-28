from huggingface_hub import hf_hub_download
from ultralytics import YOLO

model_path = hf_hub_download(
    repo_id="pitangent-ds/YOLOv8-human-detection-thermal",
    filename="model.pt"
)

model = YOLO(model_path)

import cv2
from supervision import Detections, BoxAnnotator, LabelAnnotator

# Use 0 for default webcam, or replace with your RTSP stream URL
cap = cv2.VideoCapture(0)
box_annotator = BoxAnnotator()
label_annotator = LabelAnnotator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.6)[0]
    detections = Detections.from_ultralytics(results)

    labels = [
        f"{cls} {conf:.2f}"
        for cls, conf
        in zip(detections.data['class_name'], detections.confidence)
    ]

    annotated_frame = box_annotator.annotate(
        scene=frame.copy(),
        detections=detections,
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )

    cv2.imshow("Thermal Humans", annotated_frame)
    if cv2.waitKey(1) == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

