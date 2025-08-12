# Force Python to use the newer pysqlite3-binary package
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import nltk
nltk.data.path.append("nltk_data")  # Optional: custom path if you store it locally
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
import streamlit as st
import os
import uuid
import logging
import base64
import io
import re
import time
from PIL import Image
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image as UnstructuredImage, Text as UnstructuredText
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import FastEmbedEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.schema import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from transformers import pipeline
import google.generativeai as genai

# --- Setup and Configuration ---

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CHROMA_DB_DIR = "./chroma_db"
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# --- Streamlit App Layout ---

st.set_page_config(page_title="Multimodal RAG Chatbot", layout="wide")
st.title("📄 PDF Chatbot with Multimodal RAG")
st.write("Upload a PDF to extract text and images, then ask questions about its content.")

# Initialize session state for storing processed data and state
if 'processed_flag' not in st.session_state:
    st.session_state.processed_flag = False
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

# --- Helper Functions ---

@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama3-70b-8192",
        temperature=0
    )

# The vision model is now Gemini, so this function is modified
@st.cache_resource
def get_embedding_model():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Removed get_vision_model() as it's no longer needed

def image_summarize_with_gemini(img_base64):
    """
    Summarizes an image using the Gemini 1.5 Flash API.
    """
    try:
        # Get the Gemini model for multimodal input
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Convert base64 string to a PIL Image object
        image_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(image_bytes))

        # Create a prompt for summarization
        prompt = "Please provide a detailed and concise summary of this image, capturing all key visual elements and concepts. This summary will be used for a retrieval-augmented generation (RAG) system, so be descriptive."

        # Generate the summary with the model
        response = model.generate_content([prompt, image], stream=False)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating image summary with Gemini: {e}")
        return None

# --- Core Processing Logic ---

def process_pdf_and_create_retriever(uploaded_file):
    temp_path = "temp_uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # llm is still needed for text summarization
    llm = get_llm()
    
    # Extract text and images
    st.info("Extracting text and images...")
    elements = partition_pdf(
        filename=temp_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True
    )
    
    text_elements = [el for el in elements if isinstance(el, UnstructuredText)]
    image_elements = [el for el in elements if isinstance(el, UnstructuredImage)]

    # --- Text Processing (unchanged) ---
    text_content = " ".join([el.text for el in text_elements if el.text and el.text.strip()])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.create_documents([text_content]) if text_content.strip() else []

    st.info("Summarizing extracted text...")
    # NOTE: The t5-small summarization pipeline is still used for text
    # You could also use a separate LLM call for text summarization if needed.
    summarizer_pipeline = pipeline("summarization", model="t5-small")
    text_summaries = []
    for chunk in chunks:
        try:
            if chunk.page_content.strip():
                summary = summarizer_pipeline(
                    chunk.page_content,
                    max_length=200,
                    min_length=50,
                    do_sample=False
                )[0]['summary_text'].strip()
                if summary:
                    text_summaries.append(summary)
                else:
                    logger.warning("Skipped empty summary for text chunk.")
        except Exception as e:
            logger.warning(f"Skipping text chunk due to summarization error: {e}")

    st.success("Text extraction and summarization complete. ✅")

    # --- Image Processing (Modified) ---
    img_base64_list, image_summaries = [], []
    if image_elements:
        st.info(f"Found {len(image_elements)} images. Summarizing images with Gemini...")
        for idx, el in enumerate(image_elements):
            if hasattr(el, 'contents') and el.contents:
                base64_image = base64.b64encode(el.contents).decode("utf-8")
                try:
                    # Calling the new Gemini summarization function
                    summary = image_summarize_with_gemini(base64_image)
                    if summary and summary.strip():
                        img_base64_list.append(base64_image)
                        image_summaries.append({
                            "page": getattr(el.metadata, "page_number", None),
                            "image_index": idx,
                            "summary": summary.strip()
                        })
                    else:
                        logger.warning(f"Skipped empty summary for image {idx}.")
                except Exception as e:
                    logger.warning(f"Skipping image {idx} due to summarization error: {e}")
        st.success("Image extraction and summarization complete. ✅")
    else:
        st.warning("No images found in the PDF.")

    # --- Retriever Creation (unchanged) ---
    st.info("Building vector database and retriever...")
    vectorstore = Chroma(
        collection_name="mm_rag",
        embedding_function=get_embedding_model(),
        persist_directory=CHROMA_DB_DIR
    )
    
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    def filter_non_empty_docs(docs):
        valid_docs = [doc for doc in docs if doc.page_content and doc.page_content.strip()]
        skipped = len(docs) - len(valid_docs)
        if skipped > 0:
            logger.warning(f"Skipped {skipped} empty documents.")
        return valid_docs

    if text_summaries:
        text_doc_ids = [str(uuid.uuid4()) for _ in text_summaries]
        text_summary_docs = filter_non_empty_docs([
            Document(page_content=s, metadata={id_key: text_doc_ids[i]})
            for i, s in enumerate(text_summaries)
        ])
        if text_summary_docs:
            retriever.vectorstore.add_documents(text_summary_docs)
            retriever.docstore.mset(list(zip(text_doc_ids, chunks)))
    else:
        logger.warning("No valid text summaries to add.")

    if image_summaries:
        image_doc_ids = [str(uuid.uuid4()) for _ in image_summaries]
        image_summary_docs = filter_non_empty_docs([
            Document(page_content=s["summary"], metadata={id_key: image_doc_ids[i], "page": s["page"]})
            for i, s in enumerate(image_summaries)
        ])
        if image_summary_docs:
            retriever.vectorstore.add_documents(image_summary_docs)
            retriever.docstore.mset(list(zip(image_doc_ids, img_base64_list)))
    else:
        logger.warning("No valid image summaries to add.")
    
    st.success("PDF processing complete! You can now ask questions.")
    st.session_state.processed_flag = True
    
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    return retriever

def multi_modal_rag_chain(retriever):
    llm = get_llm()
    def resize_base64_image(base64_string, size=(1300, 600)):
        try:
            img_data = base64.b64decode(base64_string)
            img = Image.open(io.BytesIO(img_data))
            resized_img = img.resize(size, Image.LANCZOS)
            buffered = io.BytesIO()
            resized_img.save(buffered, format=img.format)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return base64_string

    def looks_like_base64(sb):
        return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

    def is_image_data(b64data):
        image_signatures = {
            b"\xFF\xD8\xFF": "jpg",
            b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A": "png",
            b"\x47\x49\x46\x38": "gif",
            b"\x52\x49\x46\x46": "webp",
        }
        try:
            header = base64.b64decode(b64data)[:8]
            for sig, format in image_signatures.items():
                if header.startswith(sig):
                    return True
            return False
        except Exception:
            return False

    def split_image_text_types(docs):
        b64_images = []
        texts = []
        for doc in docs or []:
            content = doc.page_content if isinstance(doc, Document) else doc
            if looks_like_base64(content) and is_image_data(content):
                b64_images.append(resize_base64_image(content))
            else:
                texts.append(content)
        return {"texts": texts, "images": b64_images}
    
    def img_prompt_func(data_dict):
        formatted_texts = "\n\n".join(data_dict["context"]["texts"])
        if data_dict["context"]["images"]:
            # Note: This is still for a text-only LLM (Groq)
            formatted_texts = "[This context includes images that cannot be displayed in this text-only model.]\n\n" + formatted_texts
        
        prompt = (
            "You are a helpful assistant.\n"
            "You will be given mixed information.\n"
            "Use this information to answer the user's question.\n\n"
            f"User question: {data_dict['question']}\n\n"
            "Context:\n"
            f"{formatted_texts}"
        )
        return [HumanMessage(content=prompt)]

    chain = (
        {
            "context": retriever | RunnableLambda(split_image_text_types),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | llm
        | StrOutputParser()
    )
    return chain

# --- Streamlit App Logic ---

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and not st.session_state.processed_flag:
    with st.spinner("Processing PDF, this may take a moment..."):
        try:
            st.session_state.retriever = process_pdf_and_create_retriever(uploaded_file)
            st.session_state.processed_flag = True
        except Exception as e:
            st.error(f"An error occurred during PDF processing: {e}")
            logger.error(f"Processing error: {e}")

if st.session_state.processed_flag:
    st.subheader("Ask a Question")
    user_query = st.text_input("Enter your question:", key="query_input")
    
    if user_query:
        if st.session_state.retriever:
            with st.spinner("Generating response..."):
                chain = multi_modal_rag_chain(st.session_state.retriever)
                response = chain.invoke(user_query)
                st.write("**Response:**")
                st.markdown(response)
        else:
            st.warning("Retriever not initialized. Please re-upload the PDF.")
