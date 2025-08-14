# Force Python to use the newer pysqlite3-binary package 
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import nltk
nltk.data.path.append("nltk_data")
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
from PIL import Image
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image as UnstructuredImage, Text as UnstructuredText
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
import pandas as pd
import pdfplumber
import fitz # PyMuPDF
from statistics import mean
from os.path import commonprefix

# --- Setup ---
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

genai.configure(api_key=GEMINI_API_KEY)

# --- Streamlit UI ---
st.set_page_config(page_title="Multimodal RAG Chatbot", layout="wide")
st.title("📄 PDF Chatbot with Multimodal RAG")

if 'processed_flag' not in st.session_state:
    st.session_state.processed_flag = False
if 'retriever' not in st.session_state:
    st.session_state.retriever = None

@st.cache_resource
def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama3-70b-8192",
        temperature=0
    )

@st.cache_resource
def get_embedding_model():
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def image_summarize_with_gemini(img_base64):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        image_bytes = base64.b64decode(img_base64)
        image = Image.open(io.BytesIO(image_bytes))
        prompt = "Please provide a detailed and concise summary of this image for a RAG system."
        response = model.generate_content([prompt, image], stream=False)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating image summary: {e}")
        return None

# --- Table Extraction Functions ---
def page_has_table(pdf_path, page_num):
    """Check if a page has a table using pdfplumber (primary) or PyMuPDF (fallback)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            tables = pdf.pages[page_num - 1].extract_tables()
            if tables:
                return True
    except Exception as e:
        print(f"pdfplumber failed on page {page_num}: {e}")

    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_num - 1]
            blocks = page.get_text("blocks")
            rows = {}
            for b in blocks:
                x0, y0, x1, y1, text, *_ = b
                y0_rounded = round(y0, 1)
                rows.setdefault(y0_rounded, []).append(round(x0, 1))

            col_counts = [len(set(x_positions)) for x_positions in rows.values()]
            if len(col_counts) > 3 and len(set(col_counts)) == 1 and col_counts[0] > 2:
                avg_row_height = mean(
                    [abs(y2 - y1) for (_, y1, _, y2, _, *_) in blocks]
                )
                if 5 < avg_row_height < 50:
                    return True
    except Exception as e:
        print(f"PyMuPDF failed on page {page_num}: {e}")

    return False

def extract_tables_from_pdf(pdf_path, page_number):
    """Extract all tables from a page."""
    extracted = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number - 1]
            tables = page.extract_tables()
            if not tables:
                return []

            text = page.extract_text()
            lines = text.split('\n') if text else []
            footnote = None
            for line in lines:
                if line.strip().startswith('*'):
                    footnote = line.strip()
                    break

            for idx, raw_table in enumerate(tables, start=1):
                df = process_table(raw_table)
                extracted.append((df, footnote, page_number, idx))
    except Exception as e:
        print(f"Error extracting table from page {page_number}: {e}")
    return extracted

def process_table(raw_table):
    """Clean and format raw extracted table data."""
    data_start = 0
    for i, row in enumerate(raw_table):
        cell = row[0] if len(row) > 0 else None
        if cell and (cell.endswith('kg') or cell.endswith('t')):
            data_start = i
            break

    header_rows = raw_table[:data_start]
    data_rows = raw_table[data_start:]

    filled_headers = []
    for row in header_rows:
        filled_row = []
        last = None
        for cell in row:
            if cell is None:
                cell = ''
            if cell.strip():
                last = cell
            filled_row.append(last)
        filled_headers.append(filled_row)

    num_cols = len(raw_table[0])
    column_headers = []
    for col in range(num_cols):
        parts = []
        for h_row in filled_headers:
            cell = h_row[col]
            if cell and cell.strip():
                parts.append(cell.strip())
        combined = ' - '.join(parts)
        column_headers.append(combined)

    model_columns = column_headers[2:]
    if model_columns:
        common_pref = commonprefix(model_columns)
        if common_pref.endswith(' - '):
            prefix_len = len(common_pref)
            for i in range(2, num_cols):
                column_headers[i] = column_headers[i][prefix_len:]

    df = pd.DataFrame(data_rows, columns=column_headers)

    for i in range(1, len(df.columns)):
        df.iloc[:, i] = df.iloc[:, i].ffill()

    df = df.replace('', '—').fillna('—')

    return df

def print_table(df, footnote=None, page_number=None, table_index=None):
    """Print table in markdown format."""
    if page_number:
        print(f"\n## Table {table_index} from Page {page_number}\n")

    columns = [str(col) if col is not None else "Unknown" for col in df.columns]
    print('| ' + ' | '.join(columns) + ' |')
    print('| ' + ' | '.join(['---'] * len(columns)) + ' |')

    for _, row in df.iterrows():
        row_values = [str(val) if val and val != 'None' else '—' for val in row]
        print('| ' + ' | '.join(row_values) + ' |')

    if footnote:
        print('\n' + footnote)


# --- Core Processing ---
def process_pdf_and_create_retriever(uploaded_file):
    progress = st.progress(0, text="Starting PDF processing...")
    temp_path = "temp_uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    llm = get_llm()

    # Step 1: Extract text & images
    progress.progress(10, text="Extracting text and images...")
    elements = partition_pdf(
        filename=temp_path,
        strategy="hi_res",
        extract_images_in_pdf=True,
        infer_table_structure=True
    )
    text_elements = [el for el in elements if isinstance(el, UnstructuredText)]
    image_elements = [el for el in elements if isinstance(el, UnstructuredImage)]

    # Step 2: Summarize text
    progress.progress(40, text="Summarizing extracted text...")
    text_content = " ".join([el.text for el in text_elements if el.text and el.text.strip()])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.create_documents([text_content]) if text_content.strip() else []
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
        except Exception as e:
            logger.warning(f"Text summarization error: {e}")

    # Step 2.5: Extract tables from PDF
    progress.progress(60, text="Extracting tables...")
    all_table_docs = []
    with pdfplumber.open(temp_path) as pdf:
        total_pages = len(pdf.pages)
    
    pages_with_tables = [p for p in range(1, total_pages + 1) if page_has_table(temp_path, p)]
    
    for page_num in pages_with_tables:
        tables_on_page = extract_tables_from_pdf(temp_path, page_num)
        for df, footnote, page, idx in tables_on_page:
            table_text = df.to_markdown()
            metadata = {"page": page, "table_index": idx, "type": "table"}
            if footnote:
                table_text += f"\n\nFootnote: {footnote}"
            
            all_table_docs.append(Document(page_content=table_text, metadata=metadata))
            
            # Print table to terminal for debugging
            print_table(df, footnote, page, idx)

    # Step 3: Summarize images
    progress.progress(70, text="Summarizing extracted images...")
    img_base64_list, image_summaries = [], []
    for idx, el in enumerate(image_elements):
        if hasattr(el, 'contents') and el.contents:
            base64_image = base64.b64encode(el.contents).decode("utf-8")
            summary = image_summarize_with_gemini(base64_image)
            if summary:
                img_base64_list.append(base64_image)
                image_summaries.append({
                    "page": getattr(el.metadata, "page_number", None),
                    "image_index": idx,
                    "summary": summary.strip()
                })

    # Step 4: Build vector DB
    progress.progress(90, text="Building vector database...")
    vectorstore = Chroma(
        collection_name="mm_rag",
        embedding_function=get_embedding_model(),
        persist_directory=CHROMA_DB_DIR
    )
    store = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")

    if text_summaries:
        text_doc_ids = [str(uuid.uuid4()) for _ in text_summaries]
        retriever.vectorstore.add_documents([
            Document(page_content=s, metadata={"doc_id": text_doc_ids[i], "type": "text"})
            for i, s in enumerate(text_summaries)
        ])
        retriever.docstore.mset(list(zip(text_doc_ids, chunks)))

    # Store all table markdown as searchable content
    if all_table_docs:
        table_doc_ids = [str(uuid.uuid4()) for _ in all_table_docs]
        retriever.vectorstore.add_documents([
            Document(
                page_content=doc.page_content,  # Full table markdown
                metadata={"doc_id": table_doc_ids[i], "type": "table", "page": doc.metadata["page"]}
            )
            for i, doc in enumerate(all_table_docs)
        ])
        retriever.docstore.mset(list(zip(table_doc_ids, [doc.page_content for doc in all_table_docs])))

    # Store image summaries for retrieval but keep base64 for full context
    if image_summaries:
        image_doc_ids = [str(uuid.uuid4()) for _ in image_summaries]
        retriever.vectorstore.add_documents([
            Document(
                page_content=s["summary"],
                metadata={"doc_id": image_doc_ids[i], "type": "image", "page": s["page"]}
            )
            for i, s in enumerate(image_summaries)
        ])
        retriever.docstore.mset(list(zip(image_doc_ids, img_base64_list)))

    # ✅ Always mark processing as complete
    progress.progress(100, text="Processing complete!")
    st.session_state.processed_flag = True

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return retriever

# --- RAG Chain ---
def multi_modal_rag_chain(retriever):
    llm = get_llm()

    def split_image_text_types(docs):
        b64_images, texts = [], []
        for doc in docs or []:
            if isinstance(doc, str):  # Raw table or text content
                texts.append(doc)
                continue
            # If it's image base64
            if re.match("^[A-Za-z0-9+/]+[=]{0,2}$", doc.page_content):
                try:
                    header = base64.b64decode(doc.page_content)[:8]
                    if header.startswith(b"\xFF\xD8\xFF") or header.startswith(b"\x89PNG"):
                        b64_images.append(doc.page_content)
                        continue
                except:
                    pass
            texts.append(doc.page_content)
        return {"texts": texts, "images": b64_images}


    def img_prompt_func(data_dict):
        text_context = "\n\n".join(data_dict["context"]["texts"])
        if data_dict["context"]["images"]:
            text_context += "\n\n[PDF also contains images related to this context]"
        
        prompt = (
            "You must ONLY use the provided PDF content (including text, tables, and images) to answer.\n"
            "If the answer is not in the PDF, reply: 'The document does not contain this information.'\n\n"
            f"Question: {data_dict['question']}\n\n"
            f"PDF Content:\n{text_context}"
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

# --- Main App ---
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and not st.session_state.processed_flag:
    st.session_state.retriever = process_pdf_and_create_retriever(uploaded_file)

if st.session_state.processed_flag:
    st.subheader("Ask a Question")
    user_query = st.text_input("Enter your question:")
    if user_query and st.session_state.retriever:
        with st.spinner("Generating response..."):
            chain = multi_modal_rag_chain(st.session_state.retriever)
            response = chain.invoke(user_query)
            st.markdown(f"**Response:** {response}")