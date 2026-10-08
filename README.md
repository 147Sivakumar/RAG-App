## What the app does
- Accepts multiple PDFs in one upload.
- Loads every PDF with PyPDFLoader.
- Preserves the PDF's page metadata.
- Chunks all documents with RecursiveCharacterTextSplitter.
- Embeds everything into one FAISS vector database.
- Stores the FAISS database in st.session_state.
- Displays the number of indexed chunks.
- Retrieves relevant chunks across all uploaded PDFs.
- Uses ChatOpenAI to answer only from retrieved content.
- Displays:
  - answer
  - source filename
  - source page number
- Handles conflicting documents explicitly:
  - e.g. "The documents disagree. handbook.pdf states X, while policy.pdf states Y."
- Uses OPENAI_API_KEY from the environment.

## One important detail about page numbers

PyPDFLoader uses zero-based page numbers internally, so the code converts them when displaying:
page = int(doc.metadata.get("page", 0)) + 1

## Why the conflict handling matters

The answer prompt contains an explicit rule:
So if:

policy.pdf 
handbook.pdf
faq.pdf

the model should report the disagreement rather than pretending there is one definitive answer.

## About the chunk-size question in the assignment

I used:
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 6

This is a reasonable starting point, but the assignment's question about seeing five different pages is important. If answers routinely require many pages, you can experiment with smaller chunks, for example:
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100

The goal isn't simply to minimize the number of sources. You want chunks that contain enough context to answer a question while remaining specific enough that retrieval brings back the most relevant pages.
