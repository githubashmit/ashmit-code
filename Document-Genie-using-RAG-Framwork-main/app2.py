import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import easyocr  # Added EasyOCR for image text extraction
import os
from PIL import Image  # Added PIL for image handling
import numpy as np  # Added numpy to convert PIL images

st.set_page_config(page_title="Document Genie", layout="wide")

st.markdown("""
## Document Genie: Get instant insights from your Documents

This chatbot is built using the Retrieval-Augmented Generation (RAG) framework, leveraging Google's Generative AI model Gemini-PRO. It processes uploaded PDF and image files by breaking them down into manageable chunks, creates a searchable vector store, and generates accurate answers to user queries. This advanced approach ensures high-quality, contextually relevant responses for an efficient and effective user experience.

### How It Works

Follow these simple steps to interact with the chatbot:

1. **Enter Your API Key**: You'll need a Google API key for the chatbot to access Google's Generative AI models. Obtain your API key [here](https://makersuite.google.com/app/apikey).

2. **Upload Your Documents or Images**: The system accepts multiple PDF, JPG, and JPEG files at once, analyzing the content to provide comprehensive insights.

3. **Ask a Question**: After processing the documents, ask any question related to the content of your uploaded files for a precise answer.
""")

# API Key Input
api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

# Initialize OCR reader
reader = easyocr.Reader(['en'])

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Extract text from images using EasyOCR
def get_image_text(image_files):
    text = ""
    for img in image_files:
        # Convert the uploaded image file to a format easyocr can process
        image = Image.open(img)  # Open the image using PIL
        image_np = np.array(image)  # Convert the image to a numpy array
        result = reader.readtext(image_np, detail=0)  # Extract text without detailed info
        text += " ".join(result)  # Combine extracted text
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    the provided context, just say, "The answer is not available in the context." Do not provide incorrect answers.

    Context:\n {context}?\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.header("AI clone chatbot💁")

    user_question = st.text_input("Ask a Question from the Uploaded Files", key="user_question")

    if user_question and api_key:  # Ensure API key and user question are provided
        user_input(user_question, api_key)

    with st.sidebar:
        st.title("Menu:")
        
        # Upload PDF and image files
        uploaded_files = st.file_uploader("Upload PDF or Image Files (JPG, JPEG)", accept_multiple_files=True, key="file_uploader")
        
        if st.button("Submit & Process", key="process_button") and api_key:  # Check if API key is provided before processing
            if uploaded_files:
                with st.spinner("Processing..."):
                    # Separate PDF and image files
                    pdf_docs = [file for file in uploaded_files if file.name.endswith('.pdf')]
                    image_files = [file for file in uploaded_files if file.name.endswith(('.jpg', '.jpeg'))]
                    
                    # Extract text from PDFs
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        st.write("Extracted text from PDFs:")
                        st.write(raw_text)
                    else:
                        st.write("No text extracted from PDFs.")

                    # Extract text from images
                    image_text = get_image_text(image_files)
                    if image_text:
                        st.write("Extracted text from images:")
                        st.write(image_text)
                    else:
                        st.write("No text extracted from images.")
                    
                    # Combine PDF and image text
                    full_text = raw_text + " " + image_text
                    
                    if full_text.strip():
                        # Split text into chunks
                        text_chunks = get_text_chunks(full_text)
                        st.write("Text successfully split into chunks.")
                        
                        # Create vector store
                        get_vector_store(text_chunks, api_key)
                        st.success("Vector store created and saved successfully!")
                    else:
                        st.warning("No text found in the uploaded files.")
            else:
                st.warning("Please upload some files to process.")

if __name__ == "__main__":
    main()
