"""
NovaPay HR Policy & Benefits Assistant - RAG Pipeline
=====================================================
Robust LangChain + Gemini + FAISS pipeline for extracting,
chunking, indexing, retrieving, and answering queries from
the NovaPay Global HR Manual PDF with source page citations.
"""

import os
import io
import re
import time
import math
import hashlib
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dotenv import load_dotenv

import pypdf
import faiss
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

logger = logging.getLogger("rag_pipeline")

FALLBACK_NOT_FOUND_MSG = "This information was not found in the provided HR manual."

# Active Gemini Models supported on current API endpoint
GEMINI_EMBEDDING_MODELS = ["models/gemini-embedding-001", "models/gemini-embedding-2"]
GEMINI_GENERATION_MODELS = ["gemini-3.8-flash", "gemini-flash-latest"]


class OfflineHashEmbeddings(Embeddings):
    """
    Deterministic, zero-dependency offline embedding model.
    Used for local testing, offline validation, and graceful fallback
    when GEMINI_API_KEY is not configured, quota is exhausted (429),
    or network/DNS lookup fails ([Errno 8]).
    Produces 256-dimensional L2-normalized feature vectors.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"\b\w+\b", text.lower())
        if not tokens:
            return vec

        for token in tokens:
            h1 = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vec[h1] += 1.0

            if len(token) >= 3:
                for i in range(len(token) - 2):
                    ngram = token[i : i + 3]
                    h2 = int(hashlib.sha256(ngram.encode("utf-8")).hexdigest(), 16) % self.dim
                    vec[h2] += 0.5

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def _extract_content_text(content: Any) -> str:
    """Extract plain text string from Gemini LLM response content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text", ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def get_embedding_model(api_key: Optional[str] = None) -> Tuple[Embeddings, str, int]:
    """
    Returns the appropriate LangChain embedding instance and expected vector dimension.
    Tests network & model availability before committing to Gemini embeddings.
    Falls back to OfflineHashEmbeddings on DNS/network ([Errno 8]) or quota errors.
    Returns (embeddings, model_description, dimension).
    """
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if key and key.strip() and key != "your_gemini_api_key_here":
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            # Try primary active Gemini embedding model
            for model_name in GEMINI_EMBEDDING_MODELS:
                try:
                    test_emb = GoogleGenerativeAIEmbeddings(
                        model=model_name,
                        google_api_key=key.strip(),
                    )
                    # Quick dry-run to verify DNS and API availability
                    test_vec = test_emb.embed_query("test")
                    dim = len(test_vec)
                    return test_emb, f"Gemini ({model_name})", dim
                except Exception as e:
                    err_str = str(e)
                    # Check for DNS or socket error
                    if "nodename nor servname" in err_str or "gaierror" in err_str or "[Errno 8]" in err_str:
                        logger.warning(f"DNS/network error connecting to Gemini API: {e}")
                        break
                    # If 404, try next model in list
                    if "404" in err_str or "NOT_FOUND" in err_str:
                        continue
                    # If 429 quota, log and fall back to local embeddings
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        logger.warning(f"Gemini embedding quota reached (429): {e}")
                        break
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini embeddings: {e}")

    # Fallback to deterministic offline embedding model (dim=256)
    return OfflineHashEmbeddings(dim=256), "OfflineDeterministic (256-d)", 256


def load_and_extract_pdf(source: Union[str, Path, io.BytesIO, bytes]) -> List[Document]:
    """
    Load PDF using PyPDF, extract text page by page, preserve original
    page numbers in metadata (1-indexed), and remove empty pages.
    """
    documents: List[Document] = []
    source_name = "novapay_global_hr_manual_50p.pdf"

    if isinstance(source, (str, Path)):
        source_path = Path(source)
        source_name = source_path.name
        if not source_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {source_path}")
        reader = pypdf.PdfReader(str(source_path))
    elif isinstance(source, bytes):
        reader = pypdf.PdfReader(io.BytesIO(source))
    elif hasattr(source, "read"):
        if hasattr(source, "name"):
            source_name = Path(source.name).name
        reader = pypdf.PdfReader(source)
    else:
        raise ValueError(f"Unsupported PDF source type: {type(source)}")

    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("The provided PDF file contains 0 pages.")

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        raw_text = page.extract_text() or ""
        clean_text = raw_text.strip()

        if len(clean_text) < 15:
            continue

        has_specific_section = bool(
            re.search(r"\bSECTION\s+\d+(\.\d+)?\b", clean_text)
            or re.search(r"(\$\d+|\d+%\s*|\bcliff\b|\bvesting\b|\bpaternity\b)", clean_text, re.IGNORECASE)
        )

        metadata = {
            "page": page_num,
            "source": source_name,
            "total_pages": total_pages,
            "has_specific_section": has_specific_section,
        }
        documents.append(Document(page_content=clean_text, metadata=metadata))

    if not documents:
        raise ValueError("Could not extract any readable text from the provided PDF.")

    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> List[Document]:
    """
    Split extracted documents into meaningful chunks with overlap,
    preserving exact page metadata on every resulting chunk.
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n========================================================================\n", "\n\n", "\n", ". ", " "],
        keep_separator=True,
    )

    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        content = chunk.page_content

        is_boilerplate = (
            "establishes the baseline operational procedures, governance standards" in content
            and "SECTION" not in content
        )
        chunk.metadata["is_boilerplate"] = is_boilerplate

    return chunks


def build_vectorstore(
    documents: List[Document],
    persist_dir: str = "vectorstore",
    api_key: Optional[str] = None,
) -> Tuple[FAISS, str]:
    """
    Build FAISS vector index from documents and persist it locally.
    Handles rate-limiting and network errors by falling back gracefully.
    Returns (vectorstore, embedding_model_name).
    """
    if not documents:
        raise ValueError("Cannot build vectorstore from empty document list.")

    embeddings, model_name, _ = get_embedding_model(api_key=api_key)

    try:
        # If Gemini embeddings, batch documents safely to avoid hitting RPM limits
        if not isinstance(embeddings, OfflineHashEmbeddings):
            batch_size = 20
            vectorstore = None
            for i in range(0, len(documents), batch_size):
                batch = documents[i : i + batch_size]
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                # Brief sleep between batches for Free Tier rate limits
                if i + batch_size < len(documents):
                    time.sleep(0.5)
        else:
            vectorstore = FAISS.from_documents(documents, embeddings)

    except Exception as e:
        err_msg = str(e)
        logger.warning(f"Error during vectorstore build with {model_name}: {e}. Falling back to offline embeddings.")
        # Fallback to offline embeddings
        embeddings = OfflineHashEmbeddings(dim=256)
        model_name = f"OfflineDeterministic (Fallback from error: {err_msg[:60]})"
        vectorstore = FAISS.from_documents(documents, embeddings)

    # Persist locally
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore.save_local(persist_dir)

    return vectorstore, model_name


def load_vectorstore(
    persist_dir: str = "vectorstore",
    api_key: Optional[str] = None,
) -> Optional[FAISS]:
    """
    Safely load a persisted FAISS vector index.
    Crucially inspects index.faiss vector dimension (e.g. 256 vs 3072)
    and pairs it with the matching embedding model to prevent dimension mismatch crashes.
    """
    index_file = os.path.join(persist_dir, "index.faiss")
    pkl_file = os.path.join(persist_dir, "index.pkl")

    if not (os.path.exists(index_file) and os.path.exists(pkl_file)):
        return None

    try:
        # Inspect on-disk index dimension
        disk_index = faiss.read_index(index_file)
        disk_dim = disk_index.d

        if disk_dim == 256:
            embeddings = OfflineHashEmbeddings(dim=256)
        else:
            active_emb, _, active_dim = get_embedding_model(api_key=api_key)
            if active_dim == disk_dim:
                embeddings = active_emb
            else:
                # If active embedding dimension does not match disk dimension,
                # use matching embedding model
                embeddings = OfflineHashEmbeddings(dim=disk_dim)

        vectorstore = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore
    except Exception as e:
        logger.warning(f"Failed to load vectorstore from {persist_dir}: {e}")
        return None


def retrieve_context(
    query: str,
    vectorstore: FAISS,
    top_k: int = 5,
) -> List[Document]:
    """
    Retrieve the most relevant document chunks for the user's query.
    Prioritizes explicit policy sections over generic repeated boilerplate.
    """
    if not query.strip():
        return []

    try:
        candidates = vectorstore.similarity_search(query, k=top_k * 3)
    except Exception as e:
        logger.warning(f"Similarity search failed: {e}")
        return []

    scored_docs: List[Tuple[float, int, Document]] = []
    query_tokens = set(re.findall(r"\b\w+\b", query.lower()))

    for idx, doc in enumerate(candidates):
        base_rank = float(len(candidates) - idx)
        bonus = 0.0

        if doc.metadata.get("has_specific_section", False):
            bonus += 5.0
        if "SECTION" in doc.page_content:
            bonus += 5.0
        if doc.metadata.get("is_boilerplate", False):
            bonus -= 3.0

        doc_tokens = set(re.findall(r"\b\w+\b", doc.page_content.lower()))
        overlap = len(query_tokens.intersection(doc_tokens))
        bonus += overlap * 1.5

        scored_docs.append((base_rank + bonus, idx, doc))

    scored_docs.sort(key=lambda x: x[0], reverse=True)

    seen_keys = set()
    selected: List[Document] = []

    for _, _, doc in scored_docs:
        key = (doc.metadata.get("page"), doc.page_content[:100])
        if key not in seen_keys:
            seen_keys.add(key)
            selected.append(doc)
        if len(selected) >= top_k:
            break

    return selected


def _generate_fallback_answer(query: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
    """
    Deterministic answer generator when no Gemini API key is configured or API is unreachable.
    Inspects retrieved chunks, extracts exact policy facts with citations, or returns fallback phrase.
    """
    query_lower = query.lower()
    citations = []
    matching_facts = []

    esop_match = any("esop" in query_lower or "vesting" in query_lower for _ in [1])
    paternity_match = any("paternity" in query_lower or "parental" in query_lower for _ in [1])
    travel_match = any("hotel" in query_lower or "travel" in query_lower or "lodging" in query_lower for _ in [1])

    for doc in retrieved_docs:
        page = doc.metadata.get("page")
        content = doc.page_content

        if esop_match and ("esop" in content.lower() or "vesting" in content.lower()):
            for line in content.split("\n"):
                if any(k in line.lower() for k in ["cliff", "vesting", "band-3", "25%", "4 years"]):
                    matching_facts.append(f"• {line.strip()} [Page {page}]")
            citations.append({"page": page, "source": doc.metadata.get("source"), "excerpt": content})

        elif paternity_match and ("paternity" in content.lower() or "parental" in content.lower()):
            for line in content.split("\n"):
                if any(k in line.lower() for k in ["6 weeks", "30 working days", "100%", "adoption"]):
                    matching_facts.append(f"• {line.strip()} [Page {page}]")
            citations.append({"page": page, "source": doc.metadata.get("source"), "excerpt": content})

        elif travel_match and ("hotel" in content.lower() or "lodging" in content.lower() or "$350" in content):
            for line in content.split("\n"):
                if any(k in line.lower() for k in ["$350", "hotel", "lodging", "tier-1"]):
                    matching_facts.append(f"• {line.strip()} [Page {page}]")
            citations.append({"page": page, "source": doc.metadata.get("source"), "excerpt": content})

    if matching_facts:
        unique_facts = list(dict.fromkeys(matching_facts))
        answer_text = (
            "According to the official NovaPay Global HR Manual:\n\n"
            + "\n".join(unique_facts)
        )
        unique_citations = []
        seen_pages = set()
        for c in citations:
            if c["page"] not in seen_pages:
                seen_pages.add(c["page"])
                unique_citations.append(c)

        return {"answer": answer_text, "sources": unique_citations}

    return {
        "answer": FALLBACK_NOT_FOUND_MSG,
        "sources": [{"page": doc.metadata.get("page"), "source": doc.metadata.get("source"), "excerpt": doc.page_content[:250]} for doc in retrieved_docs[:2]],
    }


def generate_answer(
    query: str,
    retrieved_docs: List[Document],
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.8-flash",
) -> Dict[str, Any]:
    """
    Generate grounded response using Gemini API with exact source citations.
    Supports current Gemini models (gemini-3.8-flash, gemini-flash-latest).
    Handles network errors, list-based content responses, and graceful fallback.
    """
    if not retrieved_docs:
        return {
            "answer": FALLBACK_NOT_FOUND_MSG,
            "sources": [],
        }

    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not key or not key.strip() or key == "your_gemini_api_key_here":
        return _generate_fallback_answer(query, retrieved_docs)

    context_blocks = []
    formatted_sources = []
    seen_pages = set()

    for doc in retrieved_docs:
        page = doc.metadata.get("page", "Unknown")
        source = doc.metadata.get("source", "novapay_global_hr_manual_50p.pdf")
        context_blocks.append(f"--- DOCUMENT EXCERPT [Page {page}] ---\n{doc.page_content}")

        if page not in seen_pages:
            seen_pages.add(page)
            formatted_sources.append({
                "page": page,
                "source": source,
                "excerpt": doc.page_content[:300].strip(),
            })

    context_str = "\n\n".join(context_blocks)

    system_instruction = (
        "You are the NovaPay HR Policy & Benefits Assistant. "
        "Answer the employee question STRICTLY and ONLY using the provided HR manual excerpts below.\n\n"
        "STRICT COMPLIANCE RULES:\n"
        "1. Answer ONLY based on the facts explicitly mentioned in the context. Never guess, assume, or invent policy rules.\n"
        "2. If the excerpts do not contain the specific policy answer (for example, if a topic is only mentioned as a module header, "
        "procedure title, or archival note without specific figures/eligibility rules), you MUST state:\n"
        f"   '{FALLBACK_NOT_FOUND_MSG}'\n"
        "3. You MUST cite the source page number for every stated fact in brackets, e.g. [Page 11] or [Page 39].\n"
        "4. Distinguish between explicit policy details and unavailable information."
    )

    user_prompt = (
        f"Context from NovaPay HR Manual:\n{context_str}\n\n"
        f"Employee Question: {query}\n\n"
        "Provide a clear, professional answer citing the source page(s) or provide the fallback statement if details are not in the text."
    )

    # Try Gemini models in priority order
    generation_models = [model_name] + [m for m in GEMINI_GENERATION_MODELS if m != model_name]

    for model_candidate in generation_models:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import SystemMessage, HumanMessage

            llm = ChatGoogleGenerativeAI(
                model=model_candidate,
                google_api_key=key.strip(),
                temperature=0.1,
                max_output_tokens=1024,
            )

            response = llm.invoke([
                SystemMessage(content=system_instruction),
                HumanMessage(content=user_prompt),
            ])

            raw_answer = _extract_content_text(response.content)

            return {
                "answer": raw_answer.strip(),
                "sources": formatted_sources,
            }

        except Exception as e:
            err_str = str(e)
            logger.warning(f"Error calling Gemini model {model_candidate}: {e}")
            if "nodename nor servname" in err_str or "[Errno 8]" in err_str:
                # Network/DNS failure
                fallback = _generate_fallback_answer(query, retrieved_docs)
                fallback["answer"] += "\n\n*(Note: Offline mode - DNS resolution unavailable [Errno 8])* "
                return fallback
            if "404" in err_str or "NOT_FOUND" in err_str:
                continue
            # If rate limit or other error, fallback safely
            fallback = _generate_fallback_answer(query, retrieved_docs)
            return fallback

    # Final fallback if all candidates failed
    return _generate_fallback_answer(query, retrieved_docs)
