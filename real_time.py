import cv2
import torch

# Load your YOLOv8 model
model = torch.hub.load('ultralytics/yolov8', 'custom', path='.\runs\detect\train\weights\best.pt')

# Initialize video capture (0 for webcam, or use an IP camera URL)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Render results on the frame
    results.render()

    # Display the output
    cv2.imshow('Accident Detection', frame)

    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
