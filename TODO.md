# PDF Question Answering System — TODO and Project Requirements

## Project description

Build a PDF question-answering system that lets a user upload one or more PDFs, parse and chunk their contents, embed the chunks into a vector index, retrieve the most relevant chunks for a user question, and generate an answer grounded only in the retrieved PDF text. Every answer should include citations back to the source file and page, for example:

> BERT stands for Bidirectional Encoder Representations from Transformers.  
> [Source: bert-paper.pdf, p. 1]

The project should support:
- Text-native PDFs first, using `pypdf`.
- A path for more complex PDFs later, using `marker` or `docling`.
- Multi-document retrieval.
- Page-preserving citations.
- Manual evaluation on a small set of gold questions.

---

## Current state of the code

### Existing files

Current implementation is split across three main scripts:

```text
app.py
src/create_embedding.py
src/cosine_similarity.py
src/prompt.py       # expected by app.py, not included in the uploaded files
data/               # expected folder for uploaded PDFs
src/pdf_chunks.index
src/pdf_metadata.npy
```

The uploaded files currently appear as:
- `app.py`
- `create_embedding(1).py`
- `cosine_similarity.py`

For the app imports to work unchanged, the project should place the embedding and search scripts under `src/` as:

```text
src/create_embedding.py
src/cosine_similarity.py
```

### Main Streamlit app: `app.py`

Implemented:
- Loads environment variables using `dotenv`.
- Checks for `OPENAI_API_KEY`.
- Creates a Streamlit page titled **PDF Document Semantic Search (LangChain)**.
- Provides a sidebar PDF upload widget.
- Saves uploaded PDFs into the `data/` folder.
- Provides a **Rebuild Vector Database** button that calls `run_embeddings()`.
- Provides a text input for user questions.
- Provides a slider for number of retrieved chunks, from 1 to 5.
- Calls `search(user_query, k=num_results)` to retrieve top chunks.
- Displays retrieved chunks in an expandable section with:
  - PDF name
  - rank
  - page
  - line range
  - chunk text
- Builds a context block from retrieved chunks.
- Calls a LangChain chat model with `RAG_prompt`.
- Displays the generated answer.

Current limitations:
- The app expects `src.prompt.RAG_prompt`, but that file was not included.
- The LLM model name is hardcoded in `app.py`; this should move to environment/config.
- After rebuilding the vector database, the already-imported FAISS search module may still hold the old in-memory index unless the app reruns or search reloads the index.
- The UI does not yet show a clickable PDF citation link.
- The UI does not yet embed `pdf.js` or show the cited PDF page inline.
- The answer quality depends entirely on the missing `RAG_prompt`.
- No chat history is implemented yet; it is a single-turn QA flow.
- No uploaded file deletion/reset workflow exists.

### Embedding/indexing script: `create_embedding.py`

Implemented:
- Uses `pypdf.PdfReader` to parse PDFs from `data/`.
- Loops through all `.pdf` files in the input folder.
- Extracts text page by page.
- Preserves:
  - `pdf_name`
  - `page`
  - `lines`
  - raw chunk `text`
- Generates embeddings using OpenAI `text-embedding-3-large`.
- Creates a FAISS `IndexFlatL2` index.
- Saves:
  - `src/pdf_chunks.index`
  - `src/pdf_metadata.npy`
- Removes old index/metadata before rebuilding.

Current limitations:
- `TARGET_CHUNK_SIZE = 200` currently behaves as an approximate character-length limit, not a token limit.
- The project goal asks for 300–500 token chunks with overlap; overlap is not implemented yet.
- Metadata does not include `chunk_id`.
- Metadata uses `pdf_name`, but the project description asks for `{source_file, page_number, chunk_id}`. This should be standardized.
- Embeddings are generated in a single call for all documents; this may fail for larger PDFs. Add batching.
- No retry/backoff handling for API failures.
- No parser fallback for scanned/complex PDFs.
- No OCR path.
- No table/layout preservation.
- No document hash or incremental indexing; rebuilding always deletes and recreates the full index.
- No validation that `text-embedding-3-large` dimensions match the existing index before search.

### Search script: `cosine_similarity.py`

Implemented:
- Loads `src/pdf_chunks.index`.
- Loads `src/pdf_metadata.npy`.
- Embeds a user query using OpenAI `text-embedding-3-large`.
- Searches FAISS for top-k matches.
- Returns result dictionaries with:
  - rank
  - PDF name
  - page
  - line range
  - text
  - distance score

Current limitations:
- File is named `cosine_similarity.py`, but FAISS uses `IndexFlatL2`; this is L2 distance, not cosine similarity.
- If cosine similarity is desired, embeddings should be normalized and FAISS should use inner product, or distances should be interpreted carefully.
- The index is loaded once at import time, so rebuilt indexes may not be picked up during the same Streamlit session.
- Missing index/metadata currently calls `exit()`, which can crash the app import. Prefer raising an exception that the UI can display.
- No score threshold.
- No reranking.
- No metadata filtering by document.
- No fallback message that guides the user to rebuild the index.

---

## Phase 1 — PDF parsing and text extraction

### Goal

Extract text from PDFs while preserving enough location metadata to support citations.

### Current status

Partially implemented with `pypdf`.

### Required features

- [x] Read PDFs from `data/`.
- [x] Extract text page by page.
- [x] Preserve page number.
- [x] Preserve approximate line range.
- [ ] Rename/standardize metadata fields:
  - `source_file`
  - `page_number`
  - `chunk_id`
  - `line_start`
  - `line_end`
  - `text`
- [ ] Add `chunk_id` per document.
- [ ] Replace character-based chunking with token-aware chunking.
- [ ] Use 300–500 token chunks as the default.
- [ ] Add chunk overlap, recommended default: 50–100 tokens.
- [ ] Make `chunk_size` and `chunk_overlap` configurable.
- [ ] Keep chunks page-aware; avoid merging across pages unless explicitly tracked.
- [ ] Add parser abstraction:
  - `pypdf` for text-native PDFs
  - `docling` or `marker` for complex layouts
  - OCR option later for scanned documents
- [ ] Detect low-text/scanned PDFs and warn the user.
- [ ] Store extracted text artifacts for debugging.

### Acceptance criteria

- Given a text-native PDF, the system extracts non-empty page-level text.
- Every chunk has source file, page number, chunk ID, and text.
- A retrieved chunk can be traced back to a source page.
- Chunking uses token count, not character count.
- Overlap is present and configurable.

---

## Phase 2 — Embedding and vector index

### Goal

Embed all chunks and store them with metadata in a vector index.

### Current status

Partially implemented with OpenAI embeddings and FAISS.

### Required features

- [x] Use OpenAI embeddings.
- [x] Save FAISS index locally.
- [x] Save metadata locally with NumPy.
- [ ] Add embedding model configuration via `.env`, for example:
  - `EMBEDDING_MODEL=text-embedding-3-small`
  - `EMBEDDING_MODEL=text-embedding-3-large`
- [ ] Decide default embedding model:
  - Cost-effective hosted default: `text-embedding-3-small`
  - Higher-quality hosted option: `text-embedding-3-large`
  - Local/GPU option: `BAAI/bge-base-en-v1.5`
  - Lightweight local option: `sentence-transformers/all-MiniLM-L6-v2`
- [ ] Add embedding batching.
- [ ] Add retry/backoff for API calls.
- [ ] Normalize embeddings if cosine similarity is intended.
- [ ] Rename search script or change implementation so the distance metric matches the file name.
- [ ] Add index metadata file with:
  - embedding model
  - embedding dimension
  - chunk size
  - chunk overlap
  - parser used
  - indexed timestamp
- [ ] Add incremental indexing by file hash.
- [ ] Add optional ChromaDB backend.
- [ ] Add unit tests for index creation and metadata alignment.

### Acceptance criteria

- Number of metadata records equals number of vectors in FAISS.
- Search refuses to run if embedding model/dimension does not match the index.
- A rebuilt index is immediately searchable in the app.
- The system can index at least three PDFs without exceeding API/request limits.

---

## Phase 3 — Retrieval / RAG

### Goal

Retrieve the most relevant chunks for a user question and provide them as grounded context to the LLM.

### Current status

Basic top-k retrieval is implemented.

### Required features

- [x] Embed the user's question.
- [x] Retrieve top-k chunks.
- [x] Display retrieved source chunks in UI.
- [ ] Default top-k should be 3–5.
- [ ] Add score display and metric explanation.
- [ ] Add minimum score/maximum distance threshold.
- [ ] Add optional document filtering.
- [ ] Add reranking option for difficult questions.
- [ ] Add query rewriting option for evaluation experiments.
- [ ] Add retrieval diagnostics:
  - which documents were searched
  - top-k score distribution
  - retrieved pages
  - whether duplicate/near-duplicate chunks were returned
- [ ] Add context budget control before sending to the LLM.
- [ ] Deduplicate overlapping chunks.

### Acceptance criteria

- A query returns top-k chunks with source metadata.
- Retrieved text is passed into the answer-generation prompt.
- The UI lets the user inspect the retrieved evidence.
- Retrieval behavior can be evaluated independently from generation.

---

## Phase 4 — Answer generation with citations

### Goal

Generate answers using only retrieved chunks and cite the source PDF page.

### Current status

Partially implemented through LangChain and `RAG_prompt`, but the prompt file was not included.

### Required features

- [x] Build a context block from retrieved chunks.
- [x] Call an LLM through LangChain.
- [ ] Add or verify `src/prompt.py`.
- [ ] Make the RAG prompt strict:
  - answer only from provided chunks
  - say when the answer is not found
  - cite every factual claim
  - use `[Source: filename.pdf, p. X]`
- [ ] Include line ranges in citation when useful:
  - `[Source: filename.pdf, p. X, lines Y-Z]`
- [ ] Add structured output option:
  - `answer`
  - `citations`
  - `confidence`
  - `used_chunks`
- [ ] Add hallucination guard:
  - reject answers with citations not present in retrieved metadata
- [ ] Add answer post-processing to verify citation format.
- [ ] Move LLM model to `.env`, for example:
  - `CHAT_MODEL=...`
- [ ] Add local model path later:
  - `Mistral-7B-Instruct`
  - `Phi-3-mini`
  - 4-bit quantization on T4 if needed

### Acceptance criteria

- Answers include source citations.
- If the retrieved context does not contain the answer, the model says so.
- Citations refer only to retrieved chunks.
- The cited page actually contains the answer during manual evaluation.

---

## Phase 5 — Evaluation

### Goal

Evaluate whether the RAG system retrieves and cites correct evidence.

### Current status

Not implemented.

### Test document set

Use at least three PDF types:

- [ ] Research paper, preferably arXiv or conference paper.
- [ ] Textbook chapter.
- [ ] Contract or legal-style document.

### Manual evaluation set

Create 10–20 gold questions across the documents.

For each question, record:

- question
- expected answer
- expected source file
- expected page
- expected text span or paragraph
- retrieved top-k chunks
- generated answer
- generated citation
- whether the cited page contains the answer
- whether the answer is correct
- notes

### Metrics

- [ ] Retrieval hit rate@k: expected page appears in top-k.
- [ ] Citation correctness: cited page contains answer.
- [ ] Answer correctness: answer matches gold answer.
- [ ] Unsupported answer rate: answer contains claims not in retrieved chunks.
- [ ] Refusal correctness: system says “not found” when answer is absent.
- [ ] Latency:
  - indexing time
  - retrieval time
  - answer generation time

### Baselines

Compare against:

- [ ] Simple keyword search over extracted text.
- [ ] FAISS RAG without generation.
- [ ] Current RAG system.
- [ ] Uploading PDFs directly to a state-of-the-art LLM.
- [ ] NotebookLM or similar document QA tool, if allowed.

### Acceptance criteria

- At least 10 manually checked questions.
- Every generated citation is manually verified.
- Report includes examples of success and failure cases.
- Retrieval and generation are evaluated separately.

---

## Phase 6 — Streamlit UI

### Goal

Make the system usable through a lightweight web interface.

### Current status

Basic Streamlit UI is implemented.

### Required features

- [x] PDF upload widget.
- [x] Rebuild vector database button.
- [x] Question input.
- [x] Top-k slider.
- [x] Retrieved chunk viewer.
- [x] Generated answer display.
- [ ] Show indexed document list.
- [ ] Show index status:
  - number of PDFs
  - number of chunks
  - embedding model
  - last indexed time
- [ ] Add reset/delete document button.
- [ ] Add clickable citations.
- [ ] Add PDF page viewer or extracted page text viewer.
- [ ] Add multi-document answer source summary.
- [ ] Add chat history.
- [ ] Add error handling for missing index.
- [ ] Add warning when user uploads a scanned/low-text PDF.
- [ ] Add download/export of evaluation logs.

### Acceptance criteria

- User can upload PDFs, rebuild the index, ask a question, inspect sources, and see a cited answer.
- User can tell which PDF and page an answer came from.
- Missing index/API key/parser errors are shown clearly in the UI.

---

## Suggested immediate next steps

1. Create `src/prompt.py` with a strict RAG prompt.
2. Rename and place files into the expected project structure:
   - `src/create_embedding.py`
   - `src/cosine_similarity.py`
3. Replace `TARGET_CHUNK_SIZE = 200` with token-aware chunking.
4. Add `CHUNK_OVERLAP = 75`.
5. Add `chunk_id` and standardized metadata fields.
6. Add embedding batching.
7. Fix FAISS metric naming:
   - either use L2 and rename the file/function
   - or normalize vectors and use cosine-style inner product search
8. Reload the FAISS index after rebuild in the Streamlit app.
9. Add manual evaluation CSV template.
10. Test on three PDFs and 10–20 gold questions.

---

## Minimal `src/prompt.py` to add

```python
RAG_prompt = '''
You are a careful PDF question-answering assistant.

Use only the provided context chunks to answer the question.

Rules:
1. If the answer is not in the context, say: "I could not find the answer in the provided documents."
2. Do not use outside knowledge.
3. Every factual claim must include a citation.
4. Use this citation format: [Source: <pdf_name>, p. <page>]
5. When line numbers are available, use: [Source: <pdf_name>, p. <page>, lines <lines>]

Context:
{context}

Question:
{question}

Answer:
'''
```

---

## Recommended metadata schema

```python
{
    "source_file": "paper.pdf",
    "doc_id": "paper",
    "page_number": 12,
    "chunk_id": "paper_p12_c003",
    "line_start": 10,
    "line_end": 22,
    "text": "..."
}
```

---

## Recommended environment variables

```bash
OPENAI_API_KEY=...
EMBEDDING_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o-mini
INPUT_PATH=data/
OUTPUT_DIR=src/
CHUNK_SIZE_TOKENS=400
CHUNK_OVERLAP_TOKENS=75
TOP_K=5
```

---

## Definition of done for Phase 1 prototype

The Phase 1 prototype is complete when:

- A user can upload a text-native PDF.
- The system extracts page-preserving text.
- The system chunks the PDF into token-aware chunks with overlap.
- The system embeds and indexes chunks.
- The user can ask a question.
- The system retrieves relevant chunks.
- The model answers only from retrieved chunks.
- The answer includes a citation to the source PDF and page.
- At least 10 manual evaluation questions have been checked.
