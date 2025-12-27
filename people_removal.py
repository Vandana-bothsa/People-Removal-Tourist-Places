import cv2
import numpy as np
from ultralytics import YOLO
import os

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")

# Folders
input_folder = "images/input"
output_folder = "images/output"
mask_folder = "masks"

os.makedirs(output_folder, exist_ok=True)
os.makedirs(mask_folder, exist_ok=True)

for file_name in os.listdir(input_folder):
    if file_name.lower().endswith((".jpg", ".png", ".jpeg")):
        input_path = os.path.join(input_folder, file_name)
        image = cv2.imread(input_path)

        if image is None:
            continue

        print(f"Processing {file_name}...")

        # Run YOLO segmentation
        results = model(image)

        # Create mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for r in results:
            if r.masks is not None:
                for seg, cls in zip(r.masks.xy, r.boxes.cls):
                    if int(cls) == 0:  # person class
                        points = np.array(seg, dtype=np.int32)
                        cv2.fillPoly(mask, [points], 255)

        # Save mask
        cv2.imwrite(os.path.join(mask_folder, file_name), mask)

        # Inpainting (remove people)
        output = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        cv2.imwrite(os.path.join(output_folder, file_name), output)

print("People removal completed for all images.")
