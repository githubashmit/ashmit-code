import streamlit as st
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import pytesseract
import torch
import cv2
import numpy as np
from PIL import Image

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path as needed

# Initialize YOLOv5 model for object detection
model_name = "yolov5s"  # You can choose a pre-trained YOLOv5 model for object detection
model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
model.to('cpu')  # Assuming CPU inference for simplicity

# Directly provide the API key here
api_key = "AIzaSyDden6jxcPipiPTr5045WQ65a74-3hA_44"

def create_google_palm_embeddings():
    try:
        # Configure Google PaLM with your API key
        palm.configure(api_key=api_key)
        embeddings = GooglePalmEmbeddings(google_api_key=api_key)
        return embeddings
    except Exception as e:
        st.error(f"Error initializing GooglePalmEmbeddings: {str(e)}")
        return None

# Function to detect objects in the image using YOLOv5
def detect_objects(image):
    results = model(image)  # Run object detection
    objects = results.pandas().xyxy[0]  # Extract object information
    return objects["name"].tolist()  # Return a list of detected object names

# Function to get a conversational chain
def get_conversational_chain(embeddings):
    # Create a FAISS vector store from documents (you need to provide your own documents)
    # Placeholder text file content (replace with your own document or content)
    placeholder_text = "This is a placeholder document for testing. Replace this with your actual content."

    # Create a sample text file with placeholder content
    with open("sample_texts/sample.txt", "w") as file:
        file.write(placeholder_text)

    try:
        # Load the sample text document
        text_loader = TextLoader("sample_texts/sample.txt")  # Ensure this file exists or adjust the path
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_loader.load_and_split(text_splitter=text_splitter)

        # Create the FAISS index
        faiss_index = FAISS.from_documents(docs, embeddings)
        
        # Initialize memory to store conversation history
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Set up a template for conversation
        prompt_template = PromptTemplate(input_variables=["context", "question"],
                                         template="""
                                         Context: {context}
                                         Question: {question}
                                         """)
        
        # Initialize the conversational retrieval chain
        conversational_chain = ConversationalRetrievalChain(
            retriever=faiss_index.as_retriever(), 
            combine_docs_chain=None,  # Optional: You can add a custom combination chain here
            question_generator=None,  # Optional: You can add a custom question generator
            memory=memory,
            return_source_documents=True
        )

        return conversational_chain

    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None

def main():
    st.title("Document Genie: Analyze and Understand Your Images")

    # User image upload
    image_file = st.file_uploader("Upload an Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

    if image_file:
        try:
            # Load the image
            image = np.array(Image.open(image_file))
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Run object detection
            detected_objects = detect_objects(image)
            st.write(f"Detected objects: {', '.join(detected_objects)}")

            # Create embeddings for the conversational chain
            embeddings = create_google_palm_embeddings()

            if embeddings:
                # Initialize the conversational chain
                chain = get_conversational_chain(embeddings)

                # User input for conversation
                user_query = st.text_input("Ask a question about the image:")
                
                if user_query and chain:
                    # Generate a conversational response
                    response = chain({"question": user_query})
                    st.write("Response:", response.get("answer", "No response found."))

        except Exception as e:
            st.error(f"Error processing the image or generating a response: {str(e)}")

if __name__ == "__main__":
    main()
