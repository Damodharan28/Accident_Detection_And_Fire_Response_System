import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import os

# Load the YOLO model
model_path = r'.\runs\detect\train\weights\best.pt'  # Path to your trained model
model = YOLO(model_path)

# Define function to save frames with detections
def save_frame_with_detections(frame, boxes, folder_path, frame_number):
    # Draw bounding boxes on the frame
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Get bounding box coordinates
        class_id = int(box.cls.item())  # Get class ID
        confidence = box.conf.item()  # Get confidence score
        label = f"{model.names[class_id]}: {confidence:.2f}"  # Create label

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle in green
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the frame with detections
    frame_filename = os.path.join(folder_path, f"frame_{frame_number}.jpg")
    cv2.imwrite(frame_filename, frame)

# Set the output folder to `testing/images`
output_folder = r'.\testing\images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the video file
video_path = r".\data\testing.mp4"  # Path to your video
cap = cv2.VideoCapture(video_path)

frame_count = 0
accident_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break  # Stop if the video ends

    frame_count += 1

    # Resize the frame to 640x640
    frame_resized = cv2.resize(frame, (640, 640))

    # Convert OpenCV frame (BGR) to PIL Image (RGB)
    img_pil = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))

    # Make predictions on the resized frame using YOLO
    results = model(img_pil)

    # Check if any accident is detected in the frame
    accident_detected = False
    boxes_to_draw = []  # Store boxes to draw later
    for result in results:
        for box in result.boxes:
            class_name = model.names[box.cls.item()]
            if class_name == "Accident":
                accident_detected = True
                boxes_to_draw.append(box)  # Add box to draw list
                break

    # If accident is detected, save the frame with detections in the `testing/images` folder
    if accident_detected:
        accident_count += 1
        save_frame_with_detections(frame_resized, boxes_to_draw, output_folder, accident_count)
        print(f"Accident detected in frame {frame_count}. Saved as frame_{accident_count}.jpg")

# Release video capture object
cap.release()
cv2.destroyAllWindows()

print(f"Total frames processed: {frame_count}")
print(f"Accident frames saved: {accident_count}")
