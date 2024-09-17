import streamlit as st
import cv2
import pytesseract
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
import faiss

# Set Tesseract executable path if it's not in your PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path as needed

# Initialize object detection model (YOLOv5)
model_name = "yolov5s"  # You can choose a pre-trained YOLOv5 model for object detection
model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
model.to('cpu')  # Assuming CPU inference for simplicity

# Initialize CLIP model and processor for image embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = AutoFeatureExtractor.from_pretrained("openai/clip-vit-base-patch16")

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Document Genie: Analyze and Understand Your Images

This AI tool empowers you to upload images, extract information and objects within them, and gain insights using powerful models.
""")

# Convert image bytes to a format suitable for OpenCV
def convert_image_bytes(image_bytes):
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image

# Extract text from images using Tesseract OCR
def extract_text_from_image(image):
    try:
        # Convert the image to grayscale (optional, but can improve OCR accuracy)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Use Tesseract to extract text
        extracted_text = pytesseract.image_to_string(gray_image)
        return extracted_text
    except Exception as e:
        st.warning(f"Failed to extract text from image. Error: {str(e)}")
        return ""

# Use object detection model to identify objects in the image
def detect_objects(image):
    results = model(image)  # Run object detection
    objects = results.pandas().xyxy[0]  # Extract object information
    return objects["name"].tolist()  # Return a list of detected object names

# Generate comprehensive response based on extracted information and user query
def create_response(image_bytes, user_query):
    image = convert_image_bytes(image_bytes)

    # Extract text using OCR
    extracted_text = extract_text_from_image(image)

    # Detect objects using the object detection model
    detected_objects = detect_objects(image)

    # Start building the response based on detected objects and text
    response = f"The image contains the following objects: {', '.join(detected_objects)}.\n"
    if extracted_text:
        response += f"Additionally, the following text was detected: {extracted_text}\n"

    # Answering specific questions from the user
    if user_query:
        # Check for specific keywords in the user query
        if "object" in user_query.lower() or "contain" in user_query.lower():
            response += f"The image contains these objects: {', '.join(detected_objects)}.\n"
        if "text" in user_query.lower():
            response += f"The text in the image says: {extracted_text}\n"
        if "color" in user_query.lower():
            colors = detect_colors_in_image(image)
            response += f"The prominent colors in the image are: {', '.join(colors)}.\n"
        if "model" in user_query.lower():
            # For example, if you're analyzing car models, add specific logic here
            possible_models = find_car_models(detected_objects, extracted_text)
            if possible_models:
                response += f"Possible car models detected: {', '.join(possible_models)}.\n"

    return response

# Function to detect colors in the image (you can refine this logic)
def detect_colors_in_image(image):
    # Simple color detection based on dominant colors (you can improve this)
    avg_color = np.mean(image, axis=(0, 1))  # Average color in BGR
    return [f"R: {avg_color[2]:.0f}, G: {avg_color[1]:.0f}, B: {avg_color[0]:.0f}"]

# Sample logic to find car models based on text and detected objects
def find_car_models(detected_objects, extracted_text):
    car_models = []
    for model in CAR_MODEL_KEYWORDS:
        if model in extracted_text.lower() or model in detected_objects:
            car_models.append(model)
    return car_models

# Keywords for car models (expand as needed)
CAR_MODEL_KEYWORDS = ["toyota", "honda", "ford", "bmw", "mercedes"]

def main():
    st.header("AI Image Analyzer ")

    # User image upload
    image_file = st.file_uploader("Upload Image (JPG, JPEG)", type="jpg", accept_multiple_files=False, key="image_uploader")

    if image_file:
        with st.spinner("Analyzing image..."):
            image_bytes = image_file.read()

            # Initial analysis without query
            response = create_response(image_bytes, "")  # Initial response without query

            # Display the initial response
            st.success(response)

            # Allow user to ask questions
            user_query = st.text_input("Ask a question about the image:")
            if user_query:
                # Generate updated response with query
                response = create_response(image_bytes, user_query)
                st.success(response)

if __name__ == "__main__":
    main()
