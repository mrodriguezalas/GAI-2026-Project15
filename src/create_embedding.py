import os
import faiss
import numpy as np
from openai import OpenAI
import dotenv
import tiktoken
from pypdf import PdfReader

# Load environment variables
dotenv.load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    print("API key is missing")  

client = OpenAI()

# Configuration
INPUT_PATH = "data/"
OUTPUT_DIR = "src/"
TARGET_CHUNK_SIZE = 200


def run_embeddings():
    documents = []
    metadata = []  
    enc = tiktoken.get_encoding("cl100k_base")

    print("Cleaning up old index and metadata files...")
    files_to_remove = [
        "pdf_chunks.index", "pdf_metadata.npy",
        os.path.join(OUTPUT_DIR, "pdf_chunks.index"),
        os.path.join(OUTPUT_DIR, "pdf_metadata.npy")
    ]
    for f_path in files_to_remove:
        if os.path.exists(f_path):
            os.remove(f_path)
            print(f"Removed: {f_path}")

    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: Folder path does not exist: {INPUT_PATH}")
        exit()

    # 1. Loop through all files in the folder
    pdf_files = [f for f in os.listdir(INPUT_PATH) if f.endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files to process.")

    for pdf_filename in pdf_files:
        full_pdf_path = os.path.join(INPUT_PATH, pdf_filename)
        print(f"Processing: {pdf_filename}...")
        
        try:
            reader = PdfReader(full_pdf_path)
            
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                raw_lines = page_text.splitlines()
                
                current_chunk_text = ""
                start_line = 1
                
                for line_idx, line in enumerate(raw_lines, start=1):
                    clean_line = line.strip()
                    if not clean_line:
                        continue
                        
                    if len(current_chunk_text) + len(clean_line) > TARGET_CHUNK_SIZE and current_chunk_text:
                        chunk_to_save = current_chunk_text.strip()
                        documents.append(chunk_to_save)
                        
                        metadata.append({
                            "pdf_name": pdf_filename,  
                            "page": page_num,
                            "lines": f"{start_line}-{line_idx-1}" if start_line != (line_idx-1) else f"{start_line}",
                            "text": chunk_to_save  
                        })
                        current_chunk_text = clean_line + "\n"
                        start_line = line_idx
                    else:
                        current_chunk_text += clean_line + "\n"
                
                if current_chunk_text.strip():
                    chunk_to_save = current_chunk_text.strip()
                    documents.append(chunk_to_save)
                    
                    metadata.append({
                        "pdf_name": pdf_filename,
                        "page": page_num,
                        "lines": f"{start_line}-{len(raw_lines)}" if start_line != len(raw_lines) else f"{start_line}",
                        "text": chunk_to_save  
                    })
                    
        except Exception as e:
            print(f"❌ Error reading {pdf_filename}: {e}")
            continue

    print(f"\nTotal line-aware chunks across all files: {len(documents)}")
    if len(documents) == 0:
        print("No text data extracted. Exiting.")
        exit()


    for idx, doc in enumerate(documents):
        tokens = len(enc.encode(doc))
        if tokens > 8192:
            print(f"❌ TOO LARGE: Chunk {idx} in {metadata[idx]['pdf_name']} (Page {metadata[idx]['page']}) → {tokens} tokens")


    def embed_texts(texts):
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=texts
        )
        return np.array([e.embedding for e in response.data])

    print("Generating embeddings for all chunks...")
    embeddings = embed_texts(documents)

    # 3. Setup and Save FAISS Index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings) # pyright: ignore[reportCallIssue]

    # Save index and metadata
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    faiss.write_index(index, os.path.join(OUTPUT_DIR, "pdf_chunks.index"))
    np.save(os.path.join(OUTPUT_DIR, "pdf_metadata.npy"), np.array(metadata, dtype=object))

    return (f"Success! FAISS index + metadata for {len(pdf_files)} files saved completely.")

if __name__ == "__main__":
    print("Running embeddings")
    result = run_embeddings()
    print(result)