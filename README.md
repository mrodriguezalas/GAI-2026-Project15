# GAI-2026-Project15

## Install dependencies
- Conda environment: `conda create --name gai26 python=3.12`
- Activate the environment: `conda activate gai26`
- Install dependencies: `pip install -r requirements.txt`

## Create the PDF embedding
- Run the script: `python create_embedding.py`. This will create a `pdf_chunks.index` and `pdf_metadata.npy` files in the src/ directory.

## Run the app
- Run the app: `streamlit run app.py`

# PDF Question Answering RAG App

A small Streamlit application for asking questions over uploaded PDF documents using retrieval-augmented generation (RAG).

## Current Architecture

The app has three main parts:

1. **PDF indexing** with `create_embedding.py`
2. **Vector retrieval** with `cosine_similarity.py`
3. **Streamlit UI + answer generation** with `app.py`

Uploaded PDFs are stored in the local `data/` folder. The user can rebuild the vector database from the UI, ask a question, inspect retrieved chunks, and compare the top retrieved chunk against the rendered source PDF page.

## Chunking Strategy

The current implementation uses `pypdf.PdfReader` to extract text from PDFs page by page.

For each page:

- Text is split into raw lines.
- Empty lines are skipped.
- Lines are grouped into chunks.
- Each chunk keeps source metadata:
  - PDF filename
  - page number
  - extracted text line range
  - chunk text

Current configuration:

```python
TARGET_CHUNK_SIZE = 500
CHUNK_OVERLAP = 200 
```

Important note: the current chunk size is character-based, not token-based. Although `tiktoken` is imported and used to check whether chunks exceed the OpenAI embedding limit, the actual splitting logic currently uses string length.

Planned improvement:

- Add a stable `chunk_id`.
- Optionally add `doc_id` for cleaner multi-document support.

## Embedding Model and Vector Index

Each extracted chunk is embedded using OpenAI embeddings.

Current embedding model:

```text
text-embedding-3-large
```

The vectors are stored in a local FAISS index.

Current vector backend:

```text
FAISS IndexFlatL2
```

This means retrieval currently uses exact nearest-neighbor search with L2 distance. The UI should label this as:

```text
L2 distance — lower is better
```

Current output files:

```text
src/pdf_chunks.index
src/pdf_metadata.npy
```

`pdf_chunks.index` stores the vectors.  
`pdf_metadata.npy` stores the mapping from each vector back to its source PDF, page, extracted line range, and chunk text.

## Retrieval Flow

At query time:

1. The user question is embedded with the same OpenAI embedding model.
2. The FAISS index is searched for the top-k nearest chunks.
3. The matching metadata records are loaded from `pdf_metadata.npy`.
4. The retrieved chunks are passed into the RAG prompt as context.
5. The generated answer is shown in the Streamlit UI.

Current UI top-k range:

```text
1 to 5 chunks
```

## System Prompt / RAG Prompt

The Streamlit app imports the RAG prompt from:

```python
from src.prompt import RAG_prompt
```

The current app formats the prompt with:

```python
system = RAG_prompt.format(context=context_block, question=user_query)
```

Then it sends the formatted prompt as a `SystemMessage` to the chat model.

The prompt should enforce the following behavior:

- Answer only using the retrieved context.
- Do not invent information outside the chunks.
- Include citations using the available metadata:
  - PDF filename
  - page number
  - extracted text line range
- Say that the answer is not available if the retrieved chunks do not contain enough evidence.

Recommended citation style:

```text
[Source: file.pdf, p. X, extracted lines Y-Z]
```

## Team Observations

- The current app is a good working baseline for local PDF RAG.
- The side-by-side evidence view makes retrieval errors easier to spot.
- Page-level citation works well for manual checking, but extracted line numbers should be treated as approximate.
- The main retrieval-quality improvement will likely come from better chunking: token-aware size, overlap, and stable chunk IDs.
- The current FAISS setup is simple and fast, but metadata filtering would be easier with an optional ChromaDB backend later.
- Evaluation should focus first on whether the expected source page appears in the top-k retrieved chunks.