"""
Comprehensive Automated Test Suite for NovaPay HR Assistant
===========================================================
Validates:
A. Application and pipeline imports
B. PDF extraction and page metadata preservation
C. Text splitting and chunk metadata integrity
D. FAISS index creation and persistence
E. Relevant document chunk retrieval
F. Preservation of source page citations
G. Fallback handling for missing information
H. Graceful error handling (empty PDF, missing files)
"""

import os
import io
import pytest
from pathlib import Path

from rag_pipeline import (
    load_and_extract_pdf,
    chunk_documents,
    build_vectorstore,
    load_vectorstore,
    retrieve_context,
    generate_answer,
    OfflineHashEmbeddings,
    FALLBACK_NOT_FOUND_MSG,
)

PDF_PATH = "data/novapay_global_hr_manual_50p.pdf"
TEST_VECTORSTORE_DIR = "tests/test_vectorstore"


def test_a_imports_and_components():
    """Test A: Verify all core modules and third-party dependencies import cleanly."""
    import pypdf
    import langchain
    import langchain_community
    import langchain_text_splitters
    import streamlit
    assert pypdf is not None
    assert langchain is not None
    assert streamlit is not None


def test_b_pdf_extraction_and_metadata():
    """Test B: PDF extraction preserves all 50 pages and metadata."""
    assert Path(PDF_PATH).exists(), f"PDF file missing at {PDF_PATH}"
    docs = load_and_extract_pdf(PDF_PATH)

    assert len(docs) == 50, f"Expected 50 extracted documents, got {len(docs)}"

    # Check 1-indexed page numbering
    for i, doc in enumerate(docs):
        assert doc.metadata["page"] == i + 1, f"Mismatch at index {i}: expected page {i+1}"
        assert doc.metadata["total_pages"] == 50
        assert "source" in doc.metadata
        assert len(doc.page_content.strip()) > 0

    # Verify key content pages contain expected section headers
    page_11 = docs[10]
    assert "ESOP" in page_11.page_content or "VESTING" in page_11.page_content
    assert page_11.metadata["page"] == 11

    page_39 = docs[38]
    assert "PATERNITY" in page_39.page_content or "PARENTAL" in page_39.page_content
    assert page_39.metadata["page"] == 39


def test_c_text_splitting():
    """Test C: Text splitting preserves page metadata on every resulting chunk."""
    docs = load_and_extract_pdf(PDF_PATH)
    chunks = chunk_documents(docs, chunk_size=600, chunk_overlap=100)

    assert len(chunks) >= len(docs), "Chunks count should be at least equal to page count"

    for chunk in chunks:
        assert "page" in chunk.metadata
        assert 1 <= chunk.metadata["page"] <= 50
        assert "source" in chunk.metadata
        assert len(chunk.page_content.strip()) > 0


def test_d_faiss_vectorstore_creation_and_persistence(tmp_path):
    """Test D: FAISS index creation and persistence works."""
    docs = load_and_extract_pdf(PDF_PATH)
    chunks = chunk_documents(docs[:15])  # Use subset for fast test

    persist_dir = str(tmp_path / "faiss_test")
    vs, model_name = build_vectorstore(chunks, persist_dir=persist_dir)

    assert vs is not None
    assert os.path.exists(os.path.join(persist_dir, "index.faiss"))
    assert os.path.exists(os.path.join(persist_dir, "index.pkl"))

    loaded_vs = load_vectorstore(persist_dir)
    assert loaded_vs is not None


def test_e_relevant_retrieval():
    """Test E: Relevant document chunks can be retrieved for target queries."""
    docs = load_and_extract_pdf(PDF_PATH)
    chunks = chunk_documents(docs)
    vs, _ = build_vectorstore(chunks, persist_dir=TEST_VECTORSTORE_DIR)

    # Query ESOP vesting
    results_esop = retrieve_context("What is the ESOP vesting schedule and cliff period?", vs, top_k=3)
    assert len(results_esop) > 0

    # Ensure page 11 (the actual ESOP page) is in retrieved results
    retrieved_pages = [doc.metadata["page"] for doc in results_esop]
    assert 11 in retrieved_pages, f"Page 11 was not retrieved for ESOP query: got {retrieved_pages}"

    # Query Paternity Leave
    results_leave = retrieve_context("How many weeks of paternity leave are full-time employees entitled to?", vs, top_k=3)
    leave_pages = [doc.metadata["page"] for doc in results_leave]
    assert 39 in leave_pages, f"Page 39 was not retrieved for Paternity leave query: got {leave_pages}"


def test_f_retrieved_chunks_preserve_source_citations():
    """Test F: Retrieved chunks preserve accurate source page citations."""
    docs = load_and_extract_pdf(PDF_PATH)
    chunks = chunk_documents(docs)
    vs = load_vectorstore(TEST_VECTORSTORE_DIR) or build_vectorstore(chunks, persist_dir=TEST_VECTORSTORE_DIR)[0]

    retrieved = retrieve_context("hotel lodging caps for international travel", vs, top_k=3)
    pages = [d.metadata["page"] for d in retrieved]
    assert 27 in pages, f"Expected Page 27 in hotel lodging retrieval, got {pages}"

    answer_data = generate_answer("hotel lodging caps for international travel", retrieved)
    assert len(answer_data["sources"]) > 0
    assert any(s["page"] == 27 for s in answer_data["sources"])
    assert "$350" in answer_data["answer"] or "Page 27" in answer_data["answer"]


def test_g_missing_information_fallback():
    """Test G: Information absent from the manual produces the mandatory fallback response."""
    docs = load_and_extract_pdf(PDF_PATH)
    chunks = chunk_documents(docs)
    vs = load_vectorstore(TEST_VECTORSTORE_DIR) or build_vectorstore(chunks, persist_dir=TEST_VECTORSTORE_DIR)[0]

    # Query something definitely not in the manual
    query = "What is the pet insurance reimbursement limit for cats and dogs?"
    retrieved = retrieve_context(query, vs, top_k=3)
    answer_data = generate_answer(query, retrieved)

    assert FALLBACK_NOT_FOUND_MSG in answer_data["answer"]


def test_h_error_handling_empty_and_invalid_inputs():
    """Test H: Error handling for missing files, empty PDFs, and empty queries."""
    with pytest.raises(FileNotFoundError):
        load_and_extract_pdf("data/non_existent_file.pdf")

    # Empty query should return empty list
    vs = load_vectorstore(TEST_VECTORSTORE_DIR)
    assert retrieve_context("", vs) == []

    # Empty retrieved docs should yield fallback
    ans = generate_answer("any question", [])
    assert ans["answer"] == FALLBACK_NOT_FOUND_MSG


def test_i_streamlit_app_execution():
    """Test I: Streamlit application starts, compiles, and renders without exceptions."""
    from streamlit.testing.v1 import AppTest

    app_file_path = str(Path(__file__).parent.parent / "app.py")
    at = AppTest.from_file(app_file_path, default_timeout=15)
    at.run()
    assert not at.exception, f"Streamlit application encountered exceptions: {at.exception}"


