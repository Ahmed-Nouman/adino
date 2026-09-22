best.pt: The trained YOLOv8s (Small) model weights. Optimized for high precision on 640x480 input.

ko_yolo_centres.py: The primary deployment script. Handles real-time camera streaming, object detection, 3D deprojection (XYZ), and orientation (Yaw) calculation.



Technical Specifications
Model Performance
Architecture: YOLOv8s (Fine-tuned)

Input Resolution: 640x480 pixels

mAP@50: ~0.99 (Excellent detection reliability)

mAP@50-95: ~0.95 (High bounding-box precision for center calculation)

Coordinate Outputs
The script outputs a 4-DOF (Degrees of Freedom) coordinate set for each detected cube:

X, Y, Z: Coordinates in meters relative to the camera's optical center.

Yaw (Angle): The rotation of the cube in degrees relative to the horizontal axis of the image (0° to 90°).