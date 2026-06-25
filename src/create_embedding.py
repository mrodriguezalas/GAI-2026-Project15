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
TARGET_CHUNK_SIZE = 2000  # Updated chunk size
CHUNK_OVERLAP = 200       # Added chunk overlap (in characters)


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
                
                current_lines = []
                
                for line_idx, line in enumerate(raw_lines, start=1):
                    clean_line = line.strip()
                    if not clean_line:
                        continue
                        
                    # Calculate potential length if we add the current line
                    current_text = "\n".join([l[1] for l in current_lines])
                    if current_text:
                        potential_len = len(current_text) + 1 + len(clean_line)
                    else:
                        potential_len = len(clean_line)
                        
                    if potential_len > TARGET_CHUNK_SIZE and current_lines:
                        chunk_to_save = current_text.strip()
                        documents.append(chunk_to_save)
                        
                        start_line = current_lines[0][0]
                        end_line = current_lines[-1][0]
                        
                        metadata.append({
                            "pdf_name": pdf_filename,  
                            "page": page_num,
                            "lines": f"{start_line}-{end_line}" if start_line != end_line else f"{start_line}",
                            "text": chunk_to_save  
                        })
                        
                        # Slide the window back to create character overlap
                        while current_lines:
                            current_lines.pop(0)
                            if not current_lines:
                                break
                            remaining_text = "\n".join([l[1] for l in current_lines])
                            if len(remaining_text) <= CHUNK_OVERLAP:
                                break
                                
                        current_lines.append((line_idx, clean_line))
                    else:
                        current_lines.append((line_idx, clean_line))
                
                # Handle remaining trailing text on the page
                if current_lines:
                    chunk_to_save = "\n".join([l[1] for l in current_lines]).strip()
                    documents.append(chunk_to_save)
                    
                    start_line = current_lines[0][0]
                    end_line = current_lines[-1][0]
                    
                    metadata.append({
                        "pdf_name": pdf_filename,
                        "page": page_num,
                        "lines": f"{start_line}-{end_line}" if start_line != end_line else f"{start_line}",
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

    return f"Success! FAISS index + metadata for {len(pdf_files)} files saved completely."

if __name__ == "__main__":
    print("Running embeddings")
    result = run_embeddings()
    print(result)