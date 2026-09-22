import numpy as np
import cv2
from ultralytics import YOLO
import os
from pathlib import Path

# 1. Setup Model
model = YOLO('../model/best.pt')

def get_orientation_angle(roi):
    """Calculates the rotation angle of a cube within a cropped image."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    angle = rect[-1]
    
    width, height = rect[1]
    if width < height:
        angle = 90 + angle
        
    return round(angle, 2)

# 2. Process images from a folder
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(script_dir, '..', 'cube_dataset')
output_folder = os.path.join(script_dir, '..', 'results')

os.makedirs(output_folder, exist_ok=True)

# Get all image files
image_files = list(Path(image_folder).glob('*.jpg')) + \
              list(Path(image_folder).glob('*.png')) + \
              list(Path(image_folder).glob('*.jpeg'))

print(f"Found {len(image_files)} images to process...")

for i, image_path in enumerate(image_files, 1):
    color_image = cv2.imread(str(image_path))
    if color_image is None:
        continue
    
    color_image = cv2.resize(color_image, (640, 480))
    results = model(color_image, stream=True, conf=0.8, verbose=False)
    
    cube_count = 0
    for r in results:
        for box in r.boxes:
            cube_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            pad = 5
            roi = color_image[max(0, y1-pad):min(480, y2+pad), max(0, x1-pad):min(640, x2+pad)]
            
            if roi.size > 0:
                cube_angle = get_orientation_angle(roi)
            else:
                cube_angle = 0.0
            
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{model.names[int(box.cls[0])]}: Angle:{cube_angle}°"
            #label = f"{model.names[int(box.cls[0])]}"
            cv2.putText(color_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Save result
    output_path = os.path.join(output_folder, f"{image_path.name}")
    cv2.imwrite(output_path, color_image)

print(f"\nDone! Results saved to {output_folder}")
