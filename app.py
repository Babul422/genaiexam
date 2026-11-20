"""
NovaPay HR Policy & Benefits Assistant - Streamlit Application
=============================================================
A modern, intuitive, and grounded RAG-powered HR Policy assistant
answering employee questions strictly from the NovaPay Global HR Manual.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    load_and_extract_pdf,
    chunk_documents,
    build_vectorstore,
    load_vectorstore,
    retrieve_context,
    generate_answer,
    FALLBACK_NOT_FOUND_MSG,
)

# Load existing environment variables
load_dotenv()

DEFAULT_PDF_PATH = "data/novapay_global_hr_manual_50p.pdf"
VECTORSTORE_DIR = "vectorstore"

# Configure Streamlit page layout and theme
st.set_page_config(
    page_title="NovaPay HR Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished NovaPay enterprise branding
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-ready {
        background-color: #DEF7EC;
        color: #03543F;
    }
    .badge-warning {
        background-color: #FEF08A;
        color: #854D0E;
    }
    .citation-box {
        border-left: 3px solid #3B82F6;
        background-color: #F8FAFC;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.9rem;
    }
    .stChatMessage {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    # Try loading existing vectorstore on startup
    st.session_state.vectorstore = load_vectorstore(VECTORSTORE_DIR)

if "indexing_stats" not in st.session_state:
    st.session_state.indexing_stats = None

# Sidebar - Settings & Knowledge Base Management
with st.sidebar:
    st.markdown("### 🏢 NovaPay Workspace")
    st.caption("AI-Powered HR Policy & Compliance Assistant")

    st.markdown("---")
    st.markdown("#### 🔑 Gemini API Configuration")

    env_api_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API Key",
        value=env_api_key if env_api_key != "your_gemini_api_key_here" else "",
        type="password",
        placeholder="Enter AI Studio API Key...",
        help="Reads from .env or override here. If unset, uses local offline evaluation mode.",
    )

    active_api_key = api_key_input.strip() if api_key_input else env_api_key
    if active_api_key and active_api_key != "your_gemini_api_key_here":
        st.success("API Key Active (Gemini LLM & Embeddings)")
    else:
        st.warning("No API Key set. Running in verified offline evaluation mode.")

    st.markdown("---")
    st.markdown("#### 📄 Knowledge Base Document")

    default_exists = Path(DEFAULT_PDF_PATH).exists()
    if default_exists:
        file_size_kb = Path(DEFAULT_PDF_PATH).stat().st_size / 1024
        st.info(f"Default Manual Found: `novapay_global_hr_manual_50p.pdf` ({file_size_kb:.1f} KB)")
    else:
        st.error(f"Default manual not found at `{DEFAULT_PDF_PATH}`. Please upload below.")

    uploaded_file = st.file_uploader(
        "Upload Custom HR Manual (PDF)",
        type=["pdf"],
        help="Upload a new or revised HR Policy manual to re-index.",
    )

    process_btn = st.button("⚡ Process & Index PDF", type="primary", use_container_width=True)

    if process_btn:
        with st.status("Processing PDF into Vector Store...", expanded=True) as status:
            try:
                st.write("1. Reading & extracting PDF pages...")
                if uploaded_file is not None:
                    docs = load_and_extract_pdf(uploaded_file)
                elif default_exists:
                    docs = load_and_extract_pdf(DEFAULT_PDF_PATH)
                else:
                    raise FileNotFoundError("No PDF available to process. Please upload a PDF.")

                st.write(f"✓ Extracted {len(docs)} pages with metadata.")

                st.write("2. Splitting text into contextual chunks...")
                chunks = chunk_documents(docs, chunk_size=700, chunk_overlap=120)
                st.write(f"✓ Generated {len(chunks)} chunks.")

                st.write("3. Generating embeddings & building FAISS index...")
                start_time = time.time()
                vs, model_name = build_vectorstore(
                    chunks,
                    persist_dir=VECTORSTORE_DIR,
                    api_key=active_api_key,
                )
                elapsed = time.time() - start_time
                st.session_state.vectorstore = vs
                st.session_state.indexing_stats = {
                    "total_pages": len(docs),
                    "total_chunks": len(chunks),
                    "embedding_model": model_name,
                    "index_time": f"{elapsed:.2f}s",
                }

                status.update(label="Index built successfully!", state="complete", expanded=False)
                st.success(f"Indexed {len(docs)} pages ({len(chunks)} chunks) into FAISS!")
            except Exception as e:
                status.update(label="Processing Failed!", state="error", expanded=True)
                st.error(f"Error indexing PDF: {str(e)}")

    # Index Status Display
    if st.session_state.vectorstore is not None:
        st.markdown(
            '<div class="status-badge badge-ready">● Vector Index Ready</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.indexing_stats:
            stats = st.session_state.indexing_stats
            st.caption(f"Pages: {stats['total_pages']} | Chunks: {stats['total_chunks']} | Model: {stats['embedding_model']}")
    else:
        st.markdown(
            '<div class="status-badge badge-warning">▲ Vector Index Not Built</div>',
            unsafe_allow_html=True,
        )
        st.caption("Click 'Process & Index PDF' to initialize the search index.")

    st.markdown("---")
    st.markdown("#### 💡 Quick Sample Queries")
    sample_queries = [
        "What is the ESOP vesting schedule and cliff period?",
        "How much paid paternity leave are employees entitled to?",
        "What is the hotel lodging cap for Tier-1 international cities?",
        "What is the policy for sabbatical eligibility?",
        "What are the attendance expectations for hybrid work?",
    ]
    for sq in sample_queries:
        if st.button(sq, key=f"sample_{sq[:20]}", use_container_width=True):
            st.session_state.pending_query = sq

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# Main View Header
col1, col2 = st.columns([0.85, 0.15])
with col1:
    st.markdown('<div class="main-header">NovaPay HR Policy & Benefits Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ask questions regarding global grades, ESOP vesting, leave entitlements, travel caps, ethics, and workplace policies.</div>',
        unsafe_allow_html=True,
    )
with col2:
    st.caption("v1.0.0 • Strict RAG")

# Auto-initialize vectorstore if default PDF exists and vectorstore not yet loaded
if st.session_state.vectorstore is None and Path(DEFAULT_PDF_PATH).exists():
    with st.spinner("Auto-initializing vectorstore from default manual..."):
        try:
            docs = load_and_extract_pdf(DEFAULT_PDF_PATH)
            chunks = chunk_documents(docs)
            vs, model_name = build_vectorstore(chunks, persist_dir=VECTORSTORE_DIR, api_key=active_api_key)
            st.session_state.vectorstore = vs
            st.session_state.indexing_stats = {
                "total_pages": len(docs),
                "total_chunks": len(chunks),
                "embedding_model": model_name,
                "index_time": "cached/auto",
            }
            st.rerun()
        except Exception as e:
            st.warning(f"Auto-initialization notice: {e}. Please click 'Process & Index PDF' in the sidebar.")

# Render Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Display collapsible source citations if available
        if message.get("sources"):
            with st.expander(f"📚 Source Citations ({len(message['sources'])} reference{'s' if len(message['sources']) > 1 else ''})"):
                for src in message["sources"]:
                    st.markdown(
                        f"""
                        <div class="citation-box">
                            <strong>📄 Page {src['page']}</strong> ({src.get('source', 'Manual')})<br/>
                            <em>"{src.get('excerpt', '')}"</em>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# Check for button-clicked pending queries
prompt = st.chat_input("Ask a question about NovaPay HR policies...")
if "pending_query" in st.session_state and st.session_state.pending_query:
    prompt = st.session_state.pending_query
    st.session_state.pending_query = None

if prompt:
    # Display user query in chat
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant answer
    with st.chat_message("assistant"):
        if st.session_state.vectorstore is None:
            err_msg = "⚠️ The HR manual has not been indexed yet. Please click **'Process & Index PDF'** in the sidebar."
            st.warning(err_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": err_msg, "sources": []})
        else:
            with st.spinner("Searching HR manual and generating grounded response..."):
                retrieved_chunks = retrieve_context(prompt, st.session_state.vectorstore, top_k=4)
                result = generate_answer(
                    query=prompt,
                    retrieved_docs=retrieved_chunks,
                    api_key=active_api_key,
                )

                answer_text = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer_text)

                if sources:
                    with st.expander(f"📚 Source Citations ({len(sources)} reference{'s' if len(sources) > 1 else ''})"):
                        for src in sources:
                            st.markdown(
                                f"""
                                <div class="citation-box">
                                    <strong>📄 Page {src['page']}</strong> ({src.get('source', 'Manual')})<br/>
                                    <em>"{src.get('excerpt', '')}"</em>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources,
                })
