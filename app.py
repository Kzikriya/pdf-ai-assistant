# app.py
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import re
from docx import Document
from textblob import Word
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def correct_spelling(text):
    words = text.split()
    corrected = [Word(word).correct() for word in words]
    return " ".join(corrected)


def read_docx(file):
    doc = Document(file)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

# Load the local embedding model


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


embedder = load_embedding_model()

# Set up the page
st.set_page_config(page_title="PDF AI Assistant", page_icon="🤖")
st.header("PDF AI Assistant 🤖")

# Initialize document storage in session state
if "document_embeddings" not in st.session_state:
    st.session_state.document_embeddings = None
if "document_chunks" not in st.session_state:
    st.session_state.document_chunks = None

# Function to get text from uploaded PDF


def get_pdf_text(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Function to split text into chunks


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to process document and create embeddings


def process_document(text_chunks):
    # Create embeddings for all chunks
    chunk_embeddings = embedder.encode(text_chunks)
    return text_chunks, chunk_embeddings

# Function to get the most relevant chunks for a query


def get_relevant_chunks(query, k=3):
    # Generate embedding for the query
    query_embedding = embedder.encode([query])

    # Calculate similarity scores
    similarities = cosine_similarity(
        query_embedding, st.session_state.document_embeddings)[0]

    # Get indices of top k most similar chunks
    top_indices = similarities.argsort()[-k:][::-1]

    # Return the most relevant chunks
    return [st.session_state.document_chunks[i] for i in top_indices]

# Function to generate AI-like response from context


def generate_response(question, context_chunks):
    if not context_chunks:
        return "I couldn't find information about this specific topic in the document. Could you try asking about something else mentioned in the text?"

    # Combine the most relevant chunks
    context = " ".join(context_chunks[:2])

    # Clean up the text
    context = re.sub(r'\s+', ' ', context).strip()

    # Create a proper AI-style response
    response = f"According to the document, {context}"

    # Ensure proper sentence structure
    if response[-1] not in ['.', '!', '?']:
        response += '.'

    # Make it more helpful
    response += " Would you like me to elaborate on any specific part of this?"

    return response

# Main application logic


def main():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize PDF processed state
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    # Show file uploader only if PDF hasn't been processed yet
    if not st.session_state.pdf_processed:
        pdf_file = st.file_uploader(
            "Upload your document", type=["pdf", "txt", "docx"])

        if pdf_file is not None:
            with st.spinner("Reading and learning from your document..."):
                # Handle different file types
                if pdf_file.type == "application/pdf":
                    raw_text = get_pdf_text(pdf_file)
                elif pdf_file.type == "text/plain":
                    raw_text = str(pdf_file.read(), "utf-8")
                elif pdf_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    raw_text = read_docx(pdf_file)
                else:
                    st.error("Unsupported file type")
                    return

                text_chunks = get_text_chunks(raw_text)

                # Process document and store embeddings
                document_chunks, document_embeddings = process_document(
                    text_chunks)
                st.session_state.document_chunks = document_chunks
                st.session_state.document_embeddings = document_embeddings
                st.session_state.pdf_processed = True

            st.success(
                "Document processed successfully! You can now ask me questions about it.")
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input (only show if PDF is processed)
    if st.session_state.pdf_processed:
        if prompt := st.chat_input("Ask me anything about your document..."):
            # Add user message to chat history
            st.session_state.messages.append(
                {"role": "user", "content": prompt})

            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get AI response (WITH SPELL CHECK)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # SPELL CHECK - Correct the user's question
                    corrected_question = correct_spelling(prompt)
                    relevant_chunks = get_relevant_chunks(
                        corrected_question, k=3)
                    response = generate_response(prompt, relevant_chunks)

                st.markdown(response)

            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response})

    # Add a reset button
    if st.session_state.pdf_processed:
        if st.button("Upload a different document"):
            st.session_state.pdf_processed = False
            st.session_state.messages = []
            st.session_state.document_chunks = None
            st.session_state.document_embeddings = None
            st.rerun()


if __name__ == "__main__":
    main()
