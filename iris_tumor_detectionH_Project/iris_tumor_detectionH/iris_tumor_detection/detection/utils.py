import cv2
import numpy as np
import joblib
from django.core.files.uploadedfile import InMemoryUploadedFile

# Load the trained model
model_path = 'detection/cnn_model.pkl'  # Update this path as necessary
cnn_model = joblib.load(model_path)  # Ensure you load the correct model (CNN model in this case)

def preprocess_image(image: InMemoryUploadedFile):
    # Convert to OpenCV format
    file_bytes = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Check if the image is valid
    if img is None:
        raise ValueError("Invalid image provided")
    
    # Resize to match the original model input
    img_resized = cv2.resize(img, (64, 64))  # Assuming the model expects 64x64 images
    
    # Flatten the array to match the input shape expected by the CNN model
    img_flattened = img_resized.flatten().reshape(1, -1)  # Should be (1, 12288) for 64x64x3 images
    
    return img_flattened

def detect_tumor(image: InMemoryUploadedFile):
    features = preprocess_image(image)
    
    # Predict using the CNN model
    prediction = cnn_model.predict(features)  # Use cnn_model (which is correctly loaded)
    
    return prediction[0]  # Return the prediction (0 or 1)

# Example usage in Django view
def handle_uploaded_image(image: InMemoryUploadedFile):
    result = detect_tumor(image)
    if result == 1:
        return "Tumor detected"
    else:
        return "No tumor detected"