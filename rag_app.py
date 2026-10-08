import hashlib
import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


st.set_page_config(
    page_title="Multi-PDF RAG",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Multi-PDF RAG App")
st.write(
    "Upload several PDFs, build one FAISS vector store, and ask questions "
    "across all of them with file and page sources."
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY is not set. Set it in your environment before running "
        "the app."
    )
    st.stop()

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 6

# Keep the vector store in Streamlit session state as required.
if "store" not in st.session_state:
    st.session_state.store = None

if "index_signature" not in st.session_state:
    st.session_state.index_signature = None

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_upload_signature(files):
    """Create a stable signature so the index is rebuilt only when uploads change."""
    hasher = hashlib.sha256()

    for uploaded_file in files:
        data = uploaded_file.getvalue()
        hasher.update(uploaded_file.name.encode("utf-8"))
        hasher.update(data)

    return hasher.hexdigest()


def build_vector_store(files):
    """Load, split and embed every uploaded PDF into one FAISS store."""
    all_chunks = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for uploaded_file in files:
        # PyPDFLoader works with a file path, so temporarily save the upload.
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            pages = PyPDFLoader(temp_path).load()

            # Replace the temporary path with the real uploaded filename.
            # PyPDFLoader gives page as a zero-based integer.
            for page in pages:
                page.metadata["source"] = uploaded_file.name

            chunks = splitter.split_documents(pages)

            # Give each chunk a simple source label for the LLM prompt.
            # Page is kept in metadata and remains available after splitting.
            for chunk in chunks:
                chunk.metadata["source"] = uploaded_file.name
                chunk.metadata["page"] = int(chunk.metadata.get("page", 0))

            all_chunks.extend(chunks)

        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    if not all_chunks:
        raise ValueError("No readable text was found in the uploaded PDFs.")

    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(all_chunks, embeddings)

    return store, len(all_chunks)


def format_context(docs):
    """Format retrieved chunks with source IDs for grounded answering."""
    context_parts = []

    for i, doc in enumerate(docs, start=1):
        filename = doc.metadata.get("source", "Unknown file")
        page = int(doc.metadata.get("page", 0)) + 1

        context_parts.append(
            f"[S{i}] {filename} - page {page}\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(context_parts)


def answer_question(question, docs):
    """Answer strictly from retrieved evidence and explicitly handle conflicts."""
    context = format_context(docs)

    prompt = f"""
You are a careful document question-answering assistant.

Answer the user's question ONLY from the supplied document excerpts.

Important rules:
1. Do not invent facts that are not supported by the excerpts.
2. If two or more sources disagree, DO NOT silently choose one.
   Explicitly say that the documents disagree and summarize the different
   positions, with the relevant source labels.
3. If the evidence is insufficient, say that the uploaded documents do not
   contain enough information to answer confidently.
4. Cite supporting evidence in the answer using source labels such as [S1]
   or [S2].
5. Keep the answer concise but complete.

User question:
{question}

Retrieved document excerpts:
{context}
"""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    response = llm.invoke(prompt)
    return response.content


def unique_sources(docs):
    """Return unique (filename, page) pairs in retrieval order."""
    seen = set()
    sources = []

    for doc in docs:
        filename = doc.metadata.get("source", "Unknown file")
        page = int(doc.metadata.get("page", 0)) + 1
        key = (filename, page)

        if key not in seen:
            seen.add(key)
            sources.append(key)

    return sources


# ---------------------------------------------------------------------------
# Upload and indexing
# ---------------------------------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload several PDFs. They will all be indexed into one FAISS store.",
)

if uploaded_files:
    current_signature = make_upload_signature(uploaded_files)

    if current_signature != st.session_state.index_signature:
        with st.spinner("Loading, chunking and embedding PDFs..."):
            try:
                store, chunk_count = build_vector_store(uploaded_files)

                st.session_state.store = store
                st.session_state.chunk_count = chunk_count
                st.session_state.index_signature = current_signature
                st.session_state.indexed_files = [
                    f.name for f in uploaded_files
                ]

            except Exception as exc:
                st.session_state.store = None
                st.session_state.index_signature = None
                st.session_state.chunk_count = 0
                st.error(f"Indexing failed: {exc}")
                st.stop()

        st.success(
            f"Indexed {st.session_state.chunk_count} chunks from "
            f"{len(uploaded_files)} PDF(s)."
        )

    else:
        st.success(
            f"Indexed {st.session_state.chunk_count} chunks from "
            f"{len(st.session_state.indexed_files)} PDF(s)."
        )

    with st.expander("Indexed files", expanded=False):
        for filename in st.session_state.indexed_files:
            st.write(f"• {filename}")


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------

if st.session_state.store is not None:
    st.divider()
    st.subheader("Ask a question")

    question = st.text_input(
        "Question",
        placeholder="Example: How many leave days do new joiners get?",
    )

    if question:
        with st.spinner("Searching the documents and generating an answer..."):
            try:
                retriever = st.session_state.store.as_retriever(
                    search_kwargs={"k": TOP_K}
                )
                docs = retriever.invoke(question)

                if not docs:
                    st.warning("No relevant document chunks were found.")
                else:
                    answer = answer_question(question, docs)

                    st.markdown("### Answer")
                    st.write(answer)

                    st.markdown("### Sources")
                    for filename, page in unique_sources(docs):
                        st.caption(f"{filename} — page {page}")

            except Exception as exc:
                st.error(f"Question answering failed: {exc}")

else:
    st.info("Upload one or more PDFs above to build the shared vector store.")
