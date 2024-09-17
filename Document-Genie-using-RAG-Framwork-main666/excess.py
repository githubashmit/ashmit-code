import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import pytesseract
import google.generativeai as palm
from langchain.embeddings.google_palm import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# Set Tesseract executable path if it's not in your PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update as needed

# Initialize YOLOv5 model for object detection
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to('cpu')  # Running on CPU for simplicity

# Function to initialize Google Palm API embeddings
def create_google_palm_embeddings(api_key):
    try:
        # Configure Google PaLM with your API key
        palm.configure(api_key="AIzaSyDden6jxcPipiPTr5045WQ65a74-3hA_44")
        
        # List available models if needed
        available_models = palm.list_models()
        st.write(f"Available models: {available_models}")

        # Initialize the embedding model (you may need to change the model if it's incorrect)
        embeddings = GooglePalmEmbeddings(google_api_key=api_key)
        return embeddings
    except Exception as e:
        st.error(f"Error initializing GooglePalmEmbeddings: {str(e)}")
        return None

# Convert uploaded image to OpenCV format
def convert_image_bytes(image_bytes):
    np_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return image

# Extract text from images using Tesseract OCR
def extract_text_from_image(image):
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        extracted_text = pytesseract.image_to_string(gray_image)
        return extracted_text
    except Exception as e:
        st.warning(f"Failed to extract text from image. Error: {str(e)}")
        return ""

# Object detection using YOLOv5
def detect_objects(image):
    results = model(image)  # Run object detection
    objects = results.pandas().xyxy[0]  # Extract object information
    return objects["name"].tolist()  # Return a list of detected object names

# Function to create a conversational chain with a retriever
def get_conversational_chain(embeddings):
    try:
        # Load the FAISS index using Google Palm embeddings
        faiss_retriever = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Load a QA chain for combining the documents
        qa_chain = load_qa_chain(OpenAI(), chain_type="stuff")  # OpenAI can be replaced with any other LLM

        # Define a prompt template for question generation
        question_template = """
        Given the following conversation history and a follow-up question, generate a concise question:

        Conversation History:
        {conversation_history}
        
        Follow-up question: {user_question}
        """
        question_prompt = PromptTemplate(template=question_template, input_variables=["conversation_history", "user_question"])
        question_generator_chain = LLMChain(OpenAI(), question_prompt)

        # Initialize the conversational retrieval chain
        chain = ConversationalRetrievalChain(
            retriever=faiss_retriever.as_retriever(),
            combine_docs_chain=qa_chain,
            question_generator=question_generator_chain
        )
        
        return chain
    except Exception as e:
        st.error(f"Error initializing the conversational chain: {str(e)}")
        return None

# Function to handle user query and return response using FAISS
def user_input(user_question, embeddings):
    try:
        chain = get_conversational_chain(embeddings)
        if chain:
            # Perform similarity search and generate conversational response
            response = chain({"question": user_question}, return_only_outputs=True)
            st.write("Reply: ", response["output_text"])
        else:
            st.error("Conversational chain could not be initialized.")
    except Exception as e:
        st.error(f"Error processing user input: {str(e)}")

# Generate comprehensive response based on extracted information
def create_response(image_bytes, user_query, embeddings):
    image = convert_image_bytes(image_bytes)

    # Extract text using OCR
    extracted_text = extract_text_from_image(image)

    # Detect objects using YOLOv5
    detected_objects = detect_objects(image)

    # Build the initial response
    response = f"The image contains the following objects: {', '.join(detected_objects)}.\n"
    if extracted_text:
        response += f"Additionally, the following text was detected: {extracted_text}\n"

    # Use FAISS to answer specific questions
    if user_query:
        user_input(user_query, embeddings)
    else:
        st.write(response)

# Streamlit app layout and logic
def main():
    st.set_page_config(page_title="Document Genie", layout="wide")

    st.header("AI Image Analyzer: Upload and Analyze Images")

    # User inputs API key for Google Palm
    api_key = st.text_input("Enter your Google API Key:", type="password")

    # Initialize Google Palm Embeddings
    embeddings = None
    if api_key:
        embeddings = create_google_palm_embeddings(api_key)

    # Add a check for embeddings before proceeding
    if embeddings is None:
        st.error("Embeddings could not be initialized. Please check your API key and configuration.")
        return  # Stop execution if embeddings are not initialized

    # Upload image
    image_file = st.file_uploader("Upload Image (JPG, JPEG)", type=["jpg", "jpeg"])

    if image_file:
        with st.spinner("Analyzing image..."):
            image_bytes = image_file.read()

            # Ask the user for a query
            user_query = st.text_input("Ask a question about the image:")

            # Generate the response
            create_response(image_bytes, user_query, embeddings)

if __name__ == "__main__":
    main()
