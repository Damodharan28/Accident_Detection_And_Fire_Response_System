import torch
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO

# Use raw strings or forward slashes to avoid escape sequence issues
model_path = r'.\runs\detect\train\weights\best.pt'  # Path to model
image_path = r'.\testing\images\frame_55.jpg'  # Path to input image

def load_model(model_path):
    return YOLO(model_path)

def load_image(image_path):
    return Image.open(image_path)

def resize_image(image, size=(640, 640)):
    return image.resize(size, Image.LANCZOS)  # Updated to use LANCZOS

def make_prediction(model, image):
    return model(image)

def display_results(results, original_image, output_path):
    # Use the original image to draw results on
    img = original_image.copy()
    
    # Draw bounding boxes on the image
    if results[0].boxes is not None:
        draw = ImageDraw.Draw(img)
        for box in results[0].boxes:
            # Coordinates of the bounding box
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = box.cls.item()  # Class index
            conf = box.conf.item()  # Confidence score
            
            # Add class name and confidence score
            label = f"{model.names[cls]}: {conf:.2f}"
            
            # Draw rectangle and label
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), label, fill="red")
    
    # Save the image with bounding boxes and labels
    img.save(output_path)
    img.show()  # Show the image

if __name__ == "__main__":
    output_path = "ndw_accident.jpg"  # Path to save output image

    model = load_model(model_path)
    original_img = load_image(image_path)  # Load original image
    img_resized = resize_image(original_img)  # Resize image to 640x640
    results = make_prediction(model, img_resized)  # Use the resized image for prediction
    display_results(results, original_img, output_path)  # Pass original image to display_results
    print(f"Results saved to {output_path}")
