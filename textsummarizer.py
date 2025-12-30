# app_with_progress.py
import os 
import warnings
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import streamlit as st
import PyPDF2
from transformers import AutoTokenizer
import docx
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
# Updated imports for new LangChain versions
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import Runnable
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time  # used to simulate progress

load_dotenv()

# ---------------------------
# Configuration & Setup
# ---------------------------
# Suppress backend warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

st.set_page_config(page_title="AI Text Summarizer", layout="wide")

if "GOOGLE_API_KEY" not in st.session_state:
    # Try fetching from environment; default to "" if not found
    st.session_state["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")
if "doc_style" not in st.session_state:
    st.session_state["doc_style"] = "You are a precise analyst."

# ---------------------------
# 1. Resource Caching
# ---------------------------

@st.cache_resource
def load_llm(api_key):
    if not api_key: return None
    return ChatGoogleGenerativeAI(
        model="gemini-pro-latest",
        google_api_key=api_key,
        temperature=0.0,  # 0.0 is critical for STRICT mode (no creativity)
        convert_system_message_to_human=True
    )

@st.cache_resource
def load_embeddings():
    # Local CPU embeddings (Fast & Free)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data
def process_file(file_content, file_type):
    text = ""
    if file_type == "application/pdf":
        reader = PyPDF2.PdfReader(file_content)
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        doc = docx.Document(file_content)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text

@st.cache_resource
def create_vectorstore(_docs, _embeddings):
    return FAISS.from_documents(_docs, _embeddings)

# ---------------------------
# 2. Document Analysis (Context Detection)
# ---------------------------
def detect_document_style(llm, text_snippet):
    """
    Scans the document start to understand what it is (Contract, Textbook, Code),
    but maintains strict boundaries for the Q&A phase.
    """
    prompt = f"""
    Analyze the following text snippet. 
    1. Identify the Document Type (e.g., Legal Contract, Python Documentation, History Textbook).
    2. Write a one-sentence "Persona" for an AI that answers questions about this text.
    
    TEXT SNIPPET: "{text_snippet[:2000]}"
    
    OUTPUT FORMAT (String only):
    "This is a [Type]. Act as a [Role]. Answer strictly based on facts provided."
    """
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception:
        return "This is a document. Answer strictly based on the context."

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🛡️ AI Text Summarizer")

# Sidebar
#with st.sidebar:
#    st.header("Settings")
#    
#    # Check if key is already loaded from Env
#    is_key_loaded = bool(st.session_state["GOOGLE_API_KEY"])
#    
    # Show input only if needed, or allow override
#    api_key_input = st.text_input(
#        "Enter Google API Key", 
#        type="password",
#        value=st.session_state["GOOGLE_API_KEY"] if is_key_loaded else "",
#        placeholder="Key loaded from environment" if is_key_loaded else "Paste key here"
#    )
#    
#    if api_key_input:
#        st.session_state["GOOGLE_API_KEY"] = api_key_input
#    
#    st.info("ℹ️ **Strict Mode Active:** The AI will refuse to answer if the information is not in the document.")

#if not st.session_state["GOOGLE_API_KEY"]:
#    st.warning("Please enter your Google API Key in the sidebar.")
#    st.stop()

llm = load_llm(st.session_state["GOOGLE_API_KEY"])
embeddings = load_embeddings()

uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])

if uploaded_file and llm:
    # 1. Process File
    text = process_file(uploaded_file, uploaded_file.type)
    
    # 2. Detect Style (Run once per file)
    file_hash = hash(text[:100])
    if "current_file_hash" not in st.session_state or st.session_state["current_file_hash"] != file_hash:
        with st.spinner("Analyzing document structure..."):
            st.session_state["doc_style"] = detect_document_style(llm, text)
            st.session_state["current_file_hash"] = file_hash
            
    st.success(f"**Context Locked:** {st.session_state['doc_style']}")

    # 3. Split & Embed
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=2000,      
        chunk_overlap=200     
    )
    docs = text_splitter.create_documents([text])
    vectorstore = create_vectorstore(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # ---------------------------
    # STRICT Prompt Construction
    # ---------------------------
    
    # We enforce strictness here
    qa_template = """
    SYSTEM INSTRUCTION:
    {style_guide}
    
    STRICT RULES:
    1. Use ONLY the provided context to answer the question.
    2. Do NOT use outside knowledge (e.g., do not use general knowledge about world events).
    3. If the answer is not explicitly contained in the Context, reply: "I cannot find this information in the document."

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    # Inject style guide
    final_qa_prompt = qa_template.format(
        style_guide=st.session_state["doc_style"], 
        context="{context}", 
        question="{question}"
    )

    QA_PROMPT = PromptTemplate(template=final_qa_prompt, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    # ---------------------------
    # Summary Generation
    # ---------------------------
    if st.button("Generate Summary"):
        with st.spinner("Generating Summary..."):
            # Map Prompt
            map_prompt = """
            {style_guide}
            Summarize the following text strictly based on the content provided:
            "{text}"
            SUMMARY:"""
            
            # Format map prompt string first
            final_map_prompt = map_prompt.format(style_guide=st.session_state["doc_style"], text="{text}")
            map_prompt_template = PromptTemplate(template=final_map_prompt, input_variables=["text"])

            # Reduce Prompt
            combine_prompt = """
            {style_guide}
            Combine these summaries into a factual executive summary. Do not add outside opinion.
            ```{text}```
            SUMMARY:"""
            
            # Format combine prompt string first
            final_combine_prompt = combine_prompt.format(style_guide=st.session_state["doc_style"], text="{text}")
            combine_prompt_template = PromptTemplate(template=final_combine_prompt, input_variables=["text"])

            summary_chain = load_summarize_chain(
                llm=llm,
                chain_type="map_reduce",
                map_prompt=map_prompt_template,
                combine_prompt=combine_prompt_template,
                verbose=False
            )
            output = summary_chain.invoke({"input_documents": docs})
            st.markdown(output["output_text"])

    # ---------------------------
    # Q&A Interface
    # ---------------------------
    st.divider()
    user_query = st.text_input("Ask a specific question about the document:")
    
    if user_query:
        with st.spinner("Searching document..."):
            start_time = time.time()
            response = qa_chain.invoke({"query": user_query})
            end_time = time.time()
            st.write(response["result"])
            st.caption(f"Search time: {end_time - start_time:.2f}s")